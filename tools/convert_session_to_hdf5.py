#!/usr/bin/env python3
"""
convert_session_to_hdf5.py — Convert an ML2 recorder VRS session to a
single HDF5 file for robotics manipulation training pipelines
(EgoMimic / OSMO / EgoDex / HOMIE conventions).

RGB frames serve as the temporal anchor: all other streams are
nearest-neighbour matched (or linearly interpolated) to RGB timestamps.

Usage:
    python convert_session_to_hdf5.py /path/to/session.vrs [-o output.h5]
    python convert_session_to_hdf5.py /path/to/session.vrs --interp linear
    python convert_session_to_hdf5.py /path/to/session.vrs --no-images

Requires:
    pip install vrs h5py numpy Pillow
    # For H.264 decode:
    pip install opencv-python
"""

import argparse
import contextlib
import io
import os
import struct
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np


@contextlib.contextmanager
def _quiet_vrs_logs():
    """Redirect C-level stderr while pyvrs opens a VRS file.

    VRS's C++ logger prints info ("Opening ...") and a benign warning ("N
    record(s) not sorted properly. Sorting index.") directly to fd 2 on every
    open. These are not actionable for the researcher and drown out the
    converter's own output. We redirect fd 2 to a temp file for the duration
    of the open, then replay only lines that look like actual errors.
    """
    sys.stderr.flush()
    saved_fd = os.dup(2)
    with tempfile.TemporaryFile(mode="w+b") as tmp:
        os.dup2(tmp.fileno(), 2)
        try:
            yield
        finally:
            sys.stderr.flush()
            os.dup2(saved_fd, 2)
            os.close(saved_fd)
            tmp.seek(0)
            captured = tmp.read().decode("utf-8", errors="replace")
    # Re-emit anything that doesn't match the two known-benign patterns.
    # Strip ANSI escape sequences so we don't leave orphaned color resets.
    import re as _re
    _ansi = _re.compile(r"\x1b\[[0-9;]*m")
    for line in captured.splitlines():
        if ("ProgressLogger" in line and "Opening" in line) or \
           ("VRSIndexRecord" in line and "not sorted properly" in line):
            continue
        clean = _ansi.sub("", line).strip()
        if clean:
            print(line, file=sys.stderr)

try:
    import pyvrs
except ImportError:
    print("ERROR: pyvrs not installed.  Run:  pip install vrs", file=sys.stderr)
    sys.exit(1)

try:
    from PIL import Image as PILImage
except ImportError:
    PILImage = None

# ---------------------------------------------------------------------------
# VRS stream flavors (must match vrs_writer.cpp StreamRecordable tags)
# ---------------------------------------------------------------------------

FL_RGB        = "ml2/rgb"
FL_DEPTH      = "ml2/depth"
FL_WORLD_CAM  = ["ml2/world_cam_0", "ml2/world_cam_1", "ml2/world_cam_2"]
FL_HEAD_POSE  = "ml2/head_pose"
FL_EYE        = "ml2/eye_tracking"
FL_HAND       = "ml2/hand_tracking"
FL_IMU        = "ml2/imu"
FL_AUDIO      = "ml2/audio"
FL_MESH       = "ml2/mesh"


# ---------------------------------------------------------------------------
# VRS reader helpers
# ---------------------------------------------------------------------------

def build_flavor_map(reader: "pyvrs.SyncVRSReader") -> Dict[str, list]:
    """Return {flavor: [stream_id, ...]} for all streams in the file."""
    fm: Dict[str, list] = {}
    for sid in reader.stream_ids:
        try:
            fl = reader.get_stream_info(sid).get("flavor", "")
        except Exception:
            fl = ""
        if fl:
            fm.setdefault(fl, []).append(sid)
    return fm


def _data_records(reader: "pyvrs.SyncVRSReader", stream_ids: list) -> list:
    """Return all DATA records for the given stream IDs, sorted by timestamp."""
    recs = []
    for sid in stream_ids:
        try:
            filtered = reader.filtered_by_fields(stream_ids={sid},
                                                  record_types={"data"})
            for r in filtered:
                recs.append(r)
        except Exception:
            for r in reader:
                if r.stream_id == sid and str(r.record_type).lower() == "data":
                    recs.append(r)
    recs.sort(key=lambda r: r.timestamp)
    return recs


def _content_bytes(rec) -> bytes:
    """Return the raw bytes of the first custom content block in a record."""
    try:
        if rec.n_custom_blocks > 0:
            return bytes(rec.custom_blocks[0])
    except Exception:
        pass
    return b""


def _layout_val(rec, field: str, default=None):
    """Safely read a named field from the first metadata (DataLayout) block."""
    try:
        if rec.n_metadata_blocks > 0:
            return rec.metadata_blocks[0][field]
    except Exception:
        pass
    return default


def _layout_arr(rec, field: str) -> Optional[np.ndarray]:
    """Safely read an array field from the first metadata block as ndarray."""
    try:
        if rec.n_metadata_blocks > 0:
            v = rec.metadata_blocks[0][field]
            return np.array(v, dtype=np.float32)
    except Exception:
        pass
    return None


def load_image_poses(recs: list) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract per-record camera extrinsic (world_from_camera).

    Returns (position[N,3], orientation_wxyz[N,4], valid[N]).
    Missing fields (old VRS v1 sessions) → all invalid, zero-filled.
    Quaternion is reordered from SDK-native x,y,z,w to HDF5 w,x,y,z.
    """
    n = len(recs)
    pos = np.zeros((n, 3), dtype=np.float64)
    ori = np.zeros((n, 4), dtype=np.float64)  # w, x, y, z
    valid = np.zeros(n, dtype=np.uint8)
    for i, r in enumerate(recs):
        v = _layout_val(r, "camera_pose_valid")
        if v is None or int(v) == 0:
            continue
        p = _layout_arr(r, "camera_pose_position")
        q = _layout_arr(r, "camera_pose_orientation")
        if p is None or q is None or p.shape[0] != 3 or q.shape[0] != 4:
            continue
        pos[i] = p.astype(np.float64)
        # SDK-native (x,y,z,w) → Hamilton scalar-first (w,x,y,z)
        ori[i] = np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)
        valid[i] = 1
    return pos, ori, valid


# ---------------------------------------------------------------------------
# Quaternion helpers (Hamilton, scalar-first: w, x, y, z)
# ---------------------------------------------------------------------------

def _quat_conj(q: np.ndarray) -> np.ndarray:
    out = q.copy()
    out[..., 1:] *= -1.0
    return out


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
    ], axis=-1)


def _quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector v by quaternion q (Hamilton scalar-first)."""
    qv = np.concatenate([np.zeros(v.shape[:-1] + (1,)), v], axis=-1)
    return _quat_mul(_quat_mul(q, qv), _quat_conj(q))[..., 1:]


def _quat_avg(qs: np.ndarray) -> np.ndarray:
    """Average unit quaternions with sign consensus against qs[0]. Not
    statistically principled but close to the median for tight clusters, which
    is the regime we expect for a rigid camera-to-head mount."""
    if qs.shape[0] == 0:
        return np.array([1.0, 0.0, 0.0, 0.0])
    ref = qs[0]
    signs = np.sign(np.einsum("ij,j->i", qs, ref))
    signs[signs == 0] = 1.0
    aligned = qs * signs[:, None]
    avg = aligned.mean(axis=0)
    n = np.linalg.norm(avg)
    return avg / n if n > 0 else np.array([1.0, 0.0, 0.0, 0.0])


def _nan_invalid(pos: np.ndarray, ori: np.ndarray,
                 valid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Replace invalid rows with NaN, leaving valid rows untouched."""
    pos_out = pos.astype(np.float64, copy=True)
    ori_out = ori.astype(np.float64, copy=True)
    mask = valid == 0
    pos_out[mask] = np.nan
    ori_out[mask] = np.nan
    return pos_out, ori_out


def _write_pose_group(parent, pos: np.ndarray, ori_wxyz: np.ndarray,
                      valid: np.ndarray) -> None:
    """Write a camera_pose/ subgroup under `parent` (an h5py Group)."""
    pos_nan, ori_nan = _nan_invalid(pos, ori_wxyz, valid)
    g = parent.create_group("camera_pose")
    g.create_dataset("position",    data=pos_nan, dtype=np.float64)
    g.create_dataset("orientation", data=ori_nan, dtype=np.float64)
    g.create_dataset("valid",       data=valid.astype(np.uint8))
    g.attrs["frame"] = (
        "world_from_camera in the ML2 gravity-aligned world frame (same as "
        "head_pose/position + head_pose/orientation)."
    )
    g.attrs["columns_position"]       = "x,y,z"
    g.attrs["units_position"]         = "meters"
    g.attrs["columns_orientation"]    = "w,x,y,z"
    g.attrs["convention_orientation"] = (
        "Hamilton quaternion, scalar-first (w,x,y,z). Reordered by converter "
        "from SDK-native (x,y,z,w)."
    )
    g.attrs["valid_semantics"] = (
        "1 = pose populated by SDK for this frame; 0 = pose unavailable "
        "(position/orientation rows are NaN)."
    )
    if valid.size > 0:
        g.attrs["valid_rate"] = float(valid.mean())


def compute_head_from_camera(
    cam_pos: np.ndarray, cam_ori_wxyz: np.ndarray, cam_valid: np.ndarray,
    head_pos: np.ndarray, head_ori_wxyz: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the rigid head_from_camera transform for each valid frame.

    Both inputs are world_from_body. Returns (position[N,3], orientation_wxyz[N,4],
    valid[N]). For a physically mounted camera the stddev across frames is the
    correctness test.
    """
    n = cam_pos.shape[0]
    pos = np.zeros((n, 3), dtype=np.float64)
    ori = np.zeros((n, 4), dtype=np.float64)
    valid = np.zeros(n, dtype=np.uint8)

    # Per-row: q_hc = conj(qH) * qC ;  t_hc = conj(qH) rot (tC - tH)
    qH_inv = _quat_conj(head_ori_wxyz)
    q_hc = _quat_mul(qH_inv, cam_ori_wxyz)
    t_hc = _quat_rotate(qH_inv, cam_pos - head_pos)

    for i in range(n):
        if cam_valid[i] == 0:
            continue
        # Reject frames with degenerate head quaternion (norm ≈ 0).
        if np.linalg.norm(head_ori_wxyz[i]) < 0.5:
            continue
        pos[i] = t_hc[i]
        ori[i] = q_hc[i]
        valid[i] = 1
    return pos, ori, valid


# ---------------------------------------------------------------------------
# Interpolation helpers
# ---------------------------------------------------------------------------

def _nearest_indices(target_ts: np.ndarray, source_ts: np.ndarray) -> np.ndarray:
    """For each target timestamp, find the index of the nearest source timestamp."""
    indices = np.searchsorted(source_ts, target_ts)
    indices = np.clip(indices, 1, len(source_ts) - 1)
    left = indices - 1
    right = indices
    left_diff = np.abs(target_ts - source_ts[left])
    right_diff = np.abs(target_ts - source_ts[right])
    return np.where(left_diff <= right_diff, left, right)


def interpolate(target_ts: np.ndarray, source_ts: np.ndarray,
                source_data: np.ndarray, method: str = "nearest") -> np.ndarray:
    if method == "linear":
        n_cols = source_data.shape[1]
        result = np.empty((len(target_ts), n_cols), dtype=np.float64)
        for col in range(n_cols):
            result[:, col] = np.interp(
                target_ts.astype(np.float64),
                source_ts.astype(np.float64),
                source_data[:, col].astype(np.float64),
            )
        return result
    # nearest
    idx = _nearest_indices(target_ts, source_ts)
    return source_data[idx]


# ---------------------------------------------------------------------------
# Per-stream loaders
# ---------------------------------------------------------------------------

def load_rgb_records(reader, flavor_map) -> Tuple[np.ndarray, list]:
    """
    Load RGB stream.  Returns (timestamps_s, [record, ...]).
    Records contain JPEG or H.264 NAL content blocks.
    """
    if FL_RGB not in flavor_map:
        return np.array([]), []
    recs = _data_records(reader, flavor_map[FL_RGB])
    ts = np.array([r.timestamp for r in recs], dtype=np.float64)
    return ts, recs


def load_pose_stream(reader, flavor_map, flavor: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load head_pose or eye stream.
    head_pose DataLayout: common_ts, sensor_ts, position[3], orientation[4]  → 7 floats
    eye DataLayout: common_ts, sensor_ts, left_origin[3], left_dir[3], right_origin[3],
                    right_dir[3], fixation[3]  → 15 floats
    """
    if flavor not in flavor_map:
        return np.array([]), np.empty((0, 0))

    recs = _data_records(reader, flavor_map[flavor])
    if not recs:
        return np.array([]), np.empty((0, 0))

    ts = np.array([r.timestamp for r in recs], dtype=np.float64)

    def _arr(r, field, n):
        v = _layout_arr(r, field)
        return v if v is not None else np.zeros(n, dtype=np.float32)

    if flavor == FL_HEAD_POSE:
        def row(r):
            pos = _arr(r, "position", 3)
            ori = _arr(r, "orientation", 4)
            return np.concatenate([pos, ori])
        data = np.array([row(r) for r in recs], dtype=np.float64)  # N×7
    else:  # FL_EYE
        def row(r):
            lo = _arr(r, "left_origin", 3)
            ld = _arr(r, "left_direction", 3)
            ro = _arr(r, "right_origin", 3)
            rd = _arr(r, "right_direction", 3)
            fi = _arr(r, "fixation", 3)
            return np.concatenate([lo, ld, ro, rd, fi])
        data = np.array([row(r) for r in recs], dtype=np.float64)  # N×15

    return ts, data


def load_hand_stream(reader, flavor_map) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load hand tracking stream.
    DataLayout: left_keypoints[84], right_keypoints[84], left_confidence,
                right_confidence, left_valid, right_valid
    Output: N × 170 (84+84+2 conf)
    """
    if FL_HAND not in flavor_map:
        return np.array([]), np.empty((0, 0))

    recs = _data_records(reader, flavor_map[FL_HAND])
    if not recs:
        return np.array([]), np.empty((0, 0))

    ts = np.array([r.timestamp for r in recs], dtype=np.float64)

    def row(r):
        lk_v = _layout_arr(r, "left_keypoints")
        rk_v = _layout_arr(r, "right_keypoints")
        lk = lk_v if lk_v is not None else np.full(84, np.nan)
        rk = rk_v if rk_v is not None else np.full(84, np.nan)
        lc_v = _layout_val(r, "left_confidence")
        rc_v = _layout_val(r, "right_confidence")
        lc = float(lc_v) if lc_v is not None else np.nan
        rc = float(rc_v) if rc_v is not None else np.nan
        lv = bool(_layout_val(r, "left_valid") or False)
        rv = bool(_layout_val(r, "right_valid") or False)
        if not lv:
            lk[:] = np.nan
            lc = 0.0  # not tracked: confidence must be 0, not whatever SDK emitted
        if not rv:
            rk[:] = np.nan
            rc = 0.0
        return np.concatenate([lk[:84], rk[:84], [lc, rc]])

    data = np.array([row(r) for r in recs], dtype=np.float64)  # N×170
    return ts, data


def load_imu_stream(reader, flavor_map) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load IMU stream.
    DataLayout: common_ts, sensor_ts, unit_id, accel[3], gyro[3]
    Output: N × 7 (unit_id, ax, ay, az, gx, gy, gz)
    """
    if FL_IMU not in flavor_map:
        return np.array([]), np.empty((0, 0))

    recs = _data_records(reader, flavor_map[FL_IMU])
    if not recs:
        return np.array([]), np.empty((0, 0))

    ts = np.array([r.timestamp for r in recs], dtype=np.float64)

    def row(r):
        uid_v = _layout_val(r, "unit_id")
        uid = float(uid_v) if uid_v is not None else 0.0
        accel_v = _layout_arr(r, "accel")
        gyro_v  = _layout_arr(r, "gyro")
        accel = accel_v if accel_v is not None else np.zeros(3)
        gyro  = gyro_v  if gyro_v  is not None else np.zeros(3)
        return np.array([uid, *accel[:3], *gyro[:3]], dtype=np.float64)

    data = np.array([row(r) for r in recs], dtype=np.float64)
    return ts, data


# ---------------------------------------------------------------------------
# Calibration loader (CONFIGURATION records)
# ---------------------------------------------------------------------------

def _intrinsics_from_tags(file_tags: dict, tag_prefix: str) -> Optional[dict]:
    """Load intrinsics from ml2.cal.<prefix>.* file tags if fx & fy are present.
    Missing optional fields (cx, cy, width, height, distortion) are filled in
    as None. distortion is loaded from a space-separated tag value if present.
    """
    def _g(k):  return file_tags.get(f"ml2.cal.{tag_prefix}.{k}")
    fx, fy = _g("fx"), _g("fy")
    if fx is None or fy is None:
        return None
    try:
        out = {"fx": float(fx), "fy": float(fy),
               "cx": float(_g("cx")) if _g("cx") is not None else None,
               "cy": float(_g("cy")) if _g("cy") is not None else None,
               "width":  int(float(_g("width")))  if _g("width")  is not None else None,
               "height": int(float(_g("height"))) if _g("height") is not None else None,
               "distortion": None}
        d = _g("distortion")
        if d is not None:
            parts = [p for p in str(d).replace(",", " ").split() if p]
            if parts:
                out["distortion"] = [float(p) for p in parts]
    except (TypeError, ValueError):
        return None
    return out


def _intrinsics_from_config_record(reader, flavor_map, flavor: str) -> Optional[dict]:
    """Read intrinsics from the LAST populated CONFIGURATION record.

    The recorder writes two config records per image stream: one at session
    start (fx=0, intrinsics not yet captured) and one at session close (after
    first-frame SDK values are known). Pre-fix sessions only have the start
    record; post-fix sessions have both. We iterate all config records and
    keep the last one where fx is non-zero.
    """
    if flavor not in flavor_map:
        return None
    best: Optional[dict] = None
    for sid in flavor_map[flavor]:
        try:
            for rec in reader.filtered_by_fields(stream_ids={sid},
                                                  record_types={"configuration"}):
                fx = _layout_val(rec, "fx")
                if fx is None or float(fx) == 0.0:
                    continue
                fy   = _layout_val(rec, "fy")
                cx   = _layout_val(rec, "cx")
                cy   = _layout_val(rec, "cy")
                w    = _layout_val(rec, "width")
                h    = _layout_val(rec, "height")
                dist = _layout_arr(rec, "distortion")
                best = {
                    "fx": float(fx), "fy": float(fy),
                    "cx": float(cx) if cx is not None else None,
                    "cy": float(cy) if cy is not None else None,
                    "width":  int(w) if w is not None else None,
                    "height": int(h) if h is not None else None,
                    "distortion": dist.tolist() if dist is not None else None,
                }
        except Exception:
            pass
    return best


def load_calibration_from_config_records(reader, flavor_map, file_tags) -> Dict[str, dict]:
    """Load camera intrinsics, preferring file tags and falling back to
    CONFIGURATION records. The recorder writes intrinsics as tags on session
    close (once first-frame values are known); CONFIGURATION records are
    written at session start and usually have fx=0.
    """
    result: Dict[str, dict] = {}

    # cam_name → (tag_prefix, vrs_flavor for config-record fallback)
    streams = [
        ("rgb",         "rgb",   FL_RGB),
        ("depth",       "depth", FL_DEPTH),
        ("world_cam_0", "wcam0", FL_WORLD_CAM[0]),
        ("world_cam_1", "wcam1", FL_WORLD_CAM[1]),
        ("world_cam_2", "wcam2", FL_WORLD_CAM[2]),
    ]
    for name, tag_prefix, flavor in streams:
        tag_intr = _intrinsics_from_tags(file_tags, tag_prefix)
        cfg_intr = _intrinsics_from_config_record(reader, flavor_map, flavor)
        # Prefer tags, but backfill fields the tag path doesn't cover.
        if tag_intr is not None and cfg_intr is not None:
            for k in ("cx", "cy", "width", "height", "distortion"):
                if tag_intr.get(k) is None:
                    tag_intr[k] = cfg_intr.get(k)
            result[name] = tag_intr
        elif tag_intr is not None:
            result[name] = tag_intr
        elif cfg_intr is not None:
            result[name] = cfg_intr
    return result


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

def convert_session(vrs_path: Path, output_path: Path,
                    interp_method: str = "nearest",
                    include_images: bool = True,
                    include_depth: bool = True,
                    include_world_cams: bool = True,
                    include_audio: bool = True) -> None:

    print(f"Converting session: {vrs_path}")
    print(f"Output:             {output_path}")
    print(f"Interpolation:      {interp_method}")
    print()

    with _quiet_vrs_logs():
        reader = pyvrs.SyncVRSReader(str(vrs_path))
    file_tags = dict(reader.file_tags)
    flavor_map = build_flavor_map(reader)

    print("Streams found:", ", ".join(sorted(flavor_map.keys())) or "(none)")
    print()

    # Load calibration before the HDF5 write so it's available everywhere
    calibration = load_calibration_from_config_records(reader, flavor_map, file_tags)
    if calibration:
        print(f"Intrinsics:   {', '.join(sorted(calibration.keys()))}")
    else:
        print("Intrinsics:   (none — camera_to_head will still be emitted "
              "from per-frame extrinsics if present)")
    print()

    # ── 1. RGB (temporal anchor) ─────────────────────────────────────────
    if FL_RGB not in flavor_map:
        print("ERROR: ml2/rgb stream not found in VRS — RGB is required.")
        sys.exit(1)

    rgb_ts, rgb_recs = load_rgb_records(reader, flavor_map)
    n_frames = len(rgb_ts)
    print(f"RGB: {n_frames} frames")
    if n_frames == 0:
        print("ERROR: No RGB data records found.")
        sys.exit(1)

    # Sort by timestamp (should already be sorted, but guard against it)
    sort_idx = np.argsort(rgb_ts)
    rgb_ts   = rgb_ts[sort_idx]
    rgb_recs = [rgb_recs[i] for i in sort_idx]

    # ── 2. Other streams ─────────────────────────────────────────────────
    head_ts, head_data = load_pose_stream(reader, flavor_map, FL_HEAD_POSE)
    print(f"Head pose:    {len(head_ts)} samples")

    eye_ts, eye_data = load_pose_stream(reader, flavor_map, FL_EYE)
    print(f"Eye tracking: {len(eye_ts)} samples")

    hand_ts, hand_data = load_hand_stream(reader, flavor_map)
    print(f"Hand tracking:{len(hand_ts)} samples")

    imu_ts, imu_data = load_imu_stream(reader, flavor_map)
    print(f"IMU:          {len(imu_ts)} samples")

    # Depth records
    depth_recs: list = []
    depth_ts = np.array([])
    if include_depth and FL_DEPTH in flavor_map:
        depth_recs = _data_records(reader, flavor_map[FL_DEPTH])
        depth_ts   = np.array([r.timestamp for r in depth_recs], dtype=np.float64)
        print(f"Depth:        {len(depth_recs)} frames")

    # World camera records
    world_cam_recs: Dict[str, Tuple[np.ndarray, list]] = {}
    if include_world_cams:
        for fl in FL_WORLD_CAM:
            if fl in flavor_map:
                recs = _data_records(reader, flavor_map[fl])
                wts  = np.array([r.timestamp for r in recs], dtype=np.float64)
                world_cam_recs[fl] = (wts, recs)
                print(f"{fl.split('/')[-1]}: {len(recs)} frames")

    # Mesh records (low-rate spatial snapshots; not interpolated to RGB)
    mesh_recs: list = []
    mesh_ts_raw = np.array([], dtype=np.float64)
    if FL_MESH in flavor_map:
        mesh_recs    = _data_records(reader, flavor_map[FL_MESH])
        mesh_ts_raw  = np.array([r.timestamp for r in mesh_recs], dtype=np.float64)
        print(f"Mesh:         {len(mesh_recs)} snapshots")

    print()

    # ── 3. Interpolate to RGB timestamps ─────────────────────────────────
    print("Interpolating to RGB timestamps...")

    head_interp = (interpolate(rgb_ts, head_ts, head_data, interp_method)
                   if len(head_ts) > 0 else None)
    eye_interp  = (interpolate(rgb_ts, eye_ts, eye_data, interp_method)
                   if len(eye_ts) > 0 else None)
    hand_interp = (interpolate(rgb_ts, hand_ts, hand_data, interp_method)
                   if len(hand_ts) > 0 else None)

    depth_nearest = (_nearest_indices(rgb_ts, depth_ts)
                     if len(depth_ts) > 0 else None)

    world_cam_nearest: Dict[str, Tuple[np.ndarray, list]] = {}
    for fl, (wts, wrecs) in world_cam_recs.items():
        if len(wts) > 0:
            world_cam_nearest[fl] = (_nearest_indices(rgb_ts, wts), wrecs)

    print()

    # ── 4. Write HDF5 ────────────────────────────────────────────────────
    print(f"Writing HDF5: {output_path}")

    with h5py.File(output_path, "w") as hf:
        # File attributes from VRS tags
        hf.attrs["device"]          = file_tags.get("ml2.session_name",
                                                     vrs_path.stem)
        hf.attrs["capture_profile"] = file_tags.get("ml2.profile", "unknown")
        hf.attrs["start_time_ns"]   = int(file_tags.get("ml2.start_time_ns", 0))
        hf.attrs["n_frames"]        = n_frames
        hf.attrs["interp_method"]   = interp_method
        hf.attrs["vrs_path"]        = str(vrs_path)

        # Copy all ml2.* tags as attributes
        for k, v in file_tags.items():
            if k.startswith("ml2."):
                try:
                    hf.attrs[k] = v
                except Exception:
                    pass

        # ── Schema / researcher metadata ─────────────────────────────────
        duration_s = float(rgb_ts[-1] - rgb_ts[0]) if n_frames > 1 else 0.0
        fps_hz     = float(n_frames - 1) / duration_s if duration_s > 0 else 0.0

        hf.attrs["schema_version"]   = "1.0"
        hf.attrs["sample_description"] = (
            "One sample = one synchronized frame across all modalities, "
            "resampled via nearest-neighbour interpolation from native sensor "
            "rates to the RGB frame timeline."
        )
        hf.attrs["frame_rate_hz"]    = fps_hz
        hf.attrs["duration_s"]       = duration_s
        hf.attrs["primary_camera"]   = "rgb"
        hf.attrs["clock_domain"]     = (
            "CLOCK_BOOTTIME — nanoseconds since device boot (Android). "
            "start_time_ns is the session-start wall time in this domain. "
            "All timestamps_ns and imu/timestamps_ns are in the same domain."
        )
        hf.attrs["coordinate_system"] = (
            "ML2 SDK native coordinate frames, written as-is without any "
            "post-hoc rotation. World frame is right-handed and gravity-aligned "
            "(ML2 perception world frame). Head pose encodes world-from-device. "
            "Eye and hand data are in their respective ML2 SDK-defined frames "
            "(query MLSnapshotGetTransform for each coordinate frame ID)."
        )
        hf.attrs["quaternion_convention"] = (
            "Hamilton, scalar-first (w, x, y, z). "
            "Rotation semantics: rotation-from-body-to-world (world_R_body). "
            "Norm should be 1.0."
        )
        hf.attrs["missing_data_convention"] = (
            "NaN indicates no valid measurement. "
            "For hand tracking joints: NaN is PER-KEYPOINT — the ML2 SDK can report "
            "high confidence for a detected hand while specific joints are NaN "
            "(occluded or geometrically undetermined). "
            "confidence == 0.0 means the hand was not detected at all "
            "(left_valid=False in VRS); those frames have all joints NaN. "
            "For depth: see depth/confidence@note."
        )

        # Timestamps (seconds as float64)
        hf.create_dataset("timestamps_s",  data=rgb_ts, dtype=np.float64)
        hf.create_dataset("timestamps_ns",
                          data=(rgb_ts * 1e9).astype(np.int64), dtype=np.int64)

        # Per-camera world_from_camera poses, already aligned to RGB timestamps
        # (RGB: direct, others: nearest-matched). Collected here so the
        # calibration block at the end can derive camera_to_head.
        aligned_poses: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

        # ── RGB images ───────────────────────────────────────────────────
        if include_images:
            print("  Writing RGB images...")
            dt = h5py.vlen_dtype(np.dtype("uint8"))
            rgb_ds = hf.create_dataset("rgb/images", shape=(n_frames,), dtype=dt)

            written = 0
            for i, rec in enumerate(rgb_recs):
                raw = _content_bytes(rec)
                if raw:
                    rgb_ds[i] = np.frombuffer(raw, dtype=np.uint8)
                    written += 1
                if (i + 1) % 100 == 0:
                    print(f"    {i + 1}/{n_frames}")

            hf["rgb"].attrs["encoding"] = "h264_nal_or_jpeg"
            hf["rgb"].attrs["note"] = (
                "Each record is either a JPEG image byte blob or an H.264 NAL "
                "unit (Annex B).  Config/SPS/PPS NALs have is_config=1 in the "
                "DataLayout.  Decode with ffmpeg or a standard H.264 parser."
            )
            print(f"    {written}/{n_frames} written")

            # Per-record world_from_camera extrinsics. RGB is the temporal
            # anchor, so these are already at the root-level timestamps.
            rgb_pos, rgb_ori, rgb_valid = load_image_poses(rgb_recs)
            _write_pose_group(hf["rgb"], rgb_pos, rgb_ori, rgb_valid)
            aligned_poses["rgb"] = (rgb_pos, rgb_ori, rgb_valid)
            print(f"    rgb pose valid: {int(rgb_valid.sum())}/{n_frames}")

        # ── Head pose ────────────────────────────────────────────────────
        if head_interp is not None:
            grp = hf.create_group("head_pose")
            grp.create_dataset("position",    data=head_interp[:, :3], dtype=np.float64)
            grp.create_dataset("orientation", data=head_interp[:, 3:], dtype=np.float64)
            grp.attrs["columns_position"]    = "x,y,z"
            grp.attrs["units_position"]      = "meters"
            grp.attrs["frame_position"]      = "ML2 gravity-aligned world frame"
            grp.attrs["columns_orientation"] = "w,x,y,z"
            grp.attrs["convention_orientation"] = (
                "Hamilton quaternion, scalar-first (w,x,y,z), world_R_device "
                "(rotates device-frame vectors into world frame)"
            )
            grp.attrs["original_samples"]    = len(head_ts)
            print(f"  head_pose: {head_interp.shape[0]} frames")

        # ── Eye tracking ─────────────────────────────────────────────────
        if eye_interp is not None:
            grp = hf.create_group("eye_tracking")
            grp.create_dataset("left_origin",    data=eye_interp[:, 0:3],  dtype=np.float64)
            grp.create_dataset("left_direction",  data=eye_interp[:, 3:6],  dtype=np.float64)
            grp.create_dataset("right_origin",    data=eye_interp[:, 6:9],  dtype=np.float64)
            grp.create_dataset("right_direction", data=eye_interp[:, 9:12], dtype=np.float64)
            grp.create_dataset("fixation_point",  data=eye_interp[:, 12:15],dtype=np.float64)
            grp.attrs["units_origin"]      = "meters"
            grp.attrs["units_direction"]   = "unit vector"
            grp.attrs["units_fixation"]    = "meters"
            grp.attrs["frame"]             = "ML2 SDK per-eye coordinate frames (device-relative)"
            grp.attrs["original_samples"]  = len(eye_ts)
            print(f"  eye_tracking: {eye_interp.shape[0]} frames")

        # ── Hand tracking ────────────────────────────────────────────────
        if hand_interp is not None:
            grp = hf.create_group("hand_tracking")
            n_kp = 28
            left_joints  = hand_interp[:, :n_kp * 3].reshape(n_frames, n_kp, 3)
            right_joints = hand_interp[:, n_kp * 3:n_kp * 6].reshape(n_frames, n_kp, 3)
            grp.create_dataset("left_joints",  data=left_joints,  dtype=np.float64)
            grp.create_dataset("right_joints", data=right_joints, dtype=np.float64)
            if hand_interp.shape[1] > n_kp * 6:
                grp.create_dataset("left_confidence",
                                   data=hand_interp[:, n_kp * 6],     dtype=np.float64)
                grp.create_dataset("right_confidence",
                                   data=hand_interp[:, n_kp * 6 + 1], dtype=np.float64)
            grp.attrs["n_keypoints"]    = n_kp
            grp.attrs["keypoint_names"] = ",".join([
                "thumb_tip", "thumb_ip", "thumb_mcp", "thumb_cmc",
                "index_tip", "index_dip", "index_pip", "index_mcp",
                "middle_tip", "middle_dip", "middle_pip", "middle_mcp",
                "ring_tip", "ring_dip", "ring_pip", "ring_mcp",
                "pinky_tip", "pinky_dip", "pinky_pip", "pinky_mcp",
                "wrist_center", "wrist_ulnar", "wrist_radial",
                "hand_center", "index_meta", "middle_meta", "ring_meta", "pinky_meta",
            ])
            grp.attrs["units_joints"]   = "meters"
            grp.attrs["frame_joints"]   = "ML2 SDK per-keypoint coordinate frames (world-relative)"
            grp.attrs["confidence_range"] = "[0.0, 1.0]; 0.0 means hand not detected (joints are NaN)"
            grp.attrs["missing_data"]   = (
                "NaN is PER-KEYPOINT. The ML2 SDK can report confidence > 0 "
                "for a detected hand while specific joints are NaN (occluded or "
                "geometrically undetermined). confidence == 0.0 means hand not "
                "detected at all (left_valid=False in VRS); those frames have all "
                "joints NaN. Use confidence > 0 to identify detected-hand frames, "
                "then check individual joints for NaN before use."
            )
            grp.attrs["original_samples"] = len(hand_ts)
            print(f"  hand_tracking: {left_joints.shape}")

        # ── IMU (full rate) ───────────────────────────────────────────────
        if len(imu_ts) > 0:
            grp = hf.create_group("imu")
            grp.create_dataset("timestamps_s",  data=imu_ts, dtype=np.float64)
            grp.create_dataset("timestamps_ns",
                               data=(imu_ts * 1e9).astype(np.int64), dtype=np.int64)
            grp.create_dataset("unit_id",      data=imu_data[:, 0].astype(np.int32))
            grp.create_dataset("accelerometer",data=imu_data[:, 1:4], dtype=np.float64)
            grp.create_dataset("gyroscope",    data=imu_data[:, 4:7], dtype=np.float64)
            grp.attrs["columns_accel"]    = "x,y,z"
            grp.attrs["units_accel"]      = "m/s^2 (Android ASENSOR_TYPE_ACCELEROMETER convention)"
            grp.attrs["columns_gyro"]     = "x,y,z"
            grp.attrs["units_gyro"]       = "rad/s (Android ASENSOR_TYPE_GYROSCOPE convention)"
            grp.attrs["unit_id_note"]     = (
                "unit_id distinguishes separate IMU sensor chips or sensor groups "
                "as reported by Android SensorManager. Values observed: 0, 1, 2. "
                "Each unit_id represents an independent physical IMU on the ML2 device."
            )
            grp.attrs["clock_domain"]     = "CLOCK_BOOTTIME, same as root timestamps_ns"
            grp.attrs["n_samples"]        = len(imu_ts)
            print(f"  imu: {len(imu_ts)} samples (full rate)")

        # ── Depth ─────────────────────────────────────────────────────────
        if include_depth and depth_nearest is not None and len(depth_recs) > 0:
            print("  Writing depth frames...")
            # Peek at first frame to get dimensions
            first_raw = _content_bytes(depth_recs[0])
            width  = int(_layout_val(depth_recs[0], "width") or 0)
            height = int(_layout_val(depth_recs[0], "height") or 0)

            if width > 0 and height > 0 and first_raw:
                n_pixels = width * height
                grp = hf.create_group("depth")
                depth_ds = grp.create_dataset(
                    "images", shape=(n_frames, height, width), dtype=np.uint16,
                    chunks=(1, height, width), compression="gzip", compression_opts=4)

                # Optional confidence (packed after depth in content block)
                has_conf = len(first_raw) >= n_pixels * 2 * 2
                conf_ds = None
                if has_conf:
                    conf_ds = grp.create_dataset(
                        "confidence", shape=(n_frames, height, width), dtype=np.uint16,
                        chunks=(1, height, width), compression="gzip", compression_opts=4)

                for i in range(n_frames):
                    di = depth_nearest[i]
                    raw = _content_bytes(depth_recs[di])
                    if raw:
                        arr = np.frombuffer(raw, dtype=np.uint16)
                        depth_ds[i] = arr[:n_pixels].reshape(height, width)
                        if conf_ds is not None and len(arr) >= n_pixels * 2:
                            conf_ds[i] = arr[n_pixels:n_pixels * 2].reshape(height, width)
                    if (i + 1) % 50 == 0:
                        print(f"    {i + 1}/{n_frames}")

                grp.attrs["encoding"]         = "uint16_mm"
                grp.attrs["units"]            = "millimeters (uint16); 0 = invalid/no return"
                grp.attrs["width"]            = width
                grp.attrs["height"]           = height
                if conf_ds is not None:
                    grp["confidence"].attrs["note"] = (
                        "uint16 confidence packed after depth pixels in the VRS content block. "
                        "Interpretation of values is device-defined; all-zero values are expected "
                        "when the ML2 depth confidence signal is not populated by the firmware."
                    )
                print(f"    {n_frames}/{n_frames} done")

                # Depth extrinsics, nearest-matched to RGB timestamps.
                d_pos, d_ori, d_valid = load_image_poses(depth_recs)
                _write_pose_group(grp,
                                   d_pos[depth_nearest],
                                   d_ori[depth_nearest],
                                   d_valid[depth_nearest])
                aligned_poses["depth"] = (d_pos[depth_nearest],
                                           d_ori[depth_nearest],
                                           d_valid[depth_nearest])

        # ── World cameras ─────────────────────────────────────────────────
        if include_world_cams:
            for fl, (widx, wrecs) in world_cam_nearest.items():
                cam_name = fl.split("/")[-1]  # e.g. "world_cam_0"
                print(f"  Writing {cam_name}...")
                dt = h5py.vlen_dtype(np.dtype("uint8"))
                wgrp = hf.create_group(cam_name)
                w_ds = wgrp.create_dataset("images", shape=(n_frames,), dtype=dt)

                width  = int(_layout_val(wrecs[0], "width") or 0) if wrecs else 0
                height = int(_layout_val(wrecs[0], "height") or 0) if wrecs else 0

                for i in range(n_frames):
                    wi  = widx[i]
                    raw = _content_bytes(wrecs[wi])
                    if raw:
                        w_ds[i] = np.frombuffer(raw, dtype=np.uint8)

                wgrp.attrs["encoding"] = "jpeg"
                wgrp.attrs["width"]    = width
                wgrp.attrs["height"]   = height
                print(f"    {n_frames}/{n_frames} done")

                # World-cam extrinsics, nearest-matched to RGB timestamps.
                w_pos, w_ori, w_valid = load_image_poses(wrecs)
                _write_pose_group(wgrp,
                                   w_pos[widx], w_ori[widx], w_valid[widx])
                aligned_poses[cam_name] = (w_pos[widx], w_ori[widx], w_valid[widx])

        # ── Audio ─────────────────────────────────────────────────────────
        if include_audio and FL_AUDIO in flavor_map:
            print("  Writing audio...")
            audio_recs = _data_records(reader, flavor_map[FL_AUDIO])
            if audio_recs:
                # Collect all PCM chunks and concatenate
                chunks = []
                n_channels = int(_layout_val(audio_recs[0], "num_channels") or 4)
                for rec in audio_recs:
                    raw = _content_bytes(rec)
                    if raw:
                        n_samples = len(raw) // (2 * n_channels)
                        chunks.append(
                            np.frombuffer(raw[:n_samples * 2 * n_channels],
                                         dtype=np.int16).reshape(-1, n_channels))

                if chunks:
                    audio_data = np.concatenate(chunks, axis=0)
                    grp = hf.create_group("audio")
                    grp.create_dataset("pcm", data=audio_data, dtype=np.int16,
                                       compression="gzip", compression_opts=4)
                    # Sample rate from file tag if available
                    sample_rate = int(file_tags.get("ml2.streams.audio.sample_rate", 192000))
                    grp.attrs["sample_rate"]     = sample_rate
                    grp.attrs["n_channels"]      = n_channels
                    grp.attrs["n_frames"]        = len(audio_data)
                    grp.attrs["units"]           = "int16 PCM, full-scale range [-32768, 32767]"
                    grp.attrs["channel_mapping"] = (
                        "4-channel interleaved PCM from ML2 microphone array. "
                        "Channel layout is device-defined; no per-channel labeling "
                        "is provided by the ML2 audio API."
                    )
                    print(f"    {len(audio_data)} samples, {n_channels} ch, {sample_rate} Hz")

        # ── Mesh (spatial snapshots, not frame-aligned) ───────────────────
        if mesh_recs:
            print("  Writing mesh snapshots...")
            n_snaps = len(mesh_recs)
            mgrp = hf.create_group("mesh")
            mgrp.create_dataset("timestamps_s",
                                data=mesh_ts_raw, dtype=np.float64)
            mgrp.create_dataset("timestamps_ns",
                                data=(mesh_ts_raw * 1e9).astype(np.int64),
                                dtype=np.int64)

            snap_indices = np.zeros(n_snaps, dtype=np.uint32)
            vert_counts  = np.zeros(n_snaps, dtype=np.uint32)
            idx_counts   = np.zeros(n_snaps, dtype=np.uint32)

            vlen_f32 = h5py.vlen_dtype(np.dtype("float32"))
            vlen_u32 = h5py.vlen_dtype(np.dtype("uint32"))
            verts_ds = mgrp.create_dataset("vertices", shape=(n_snaps,), dtype=vlen_f32)
            idx_ds   = mgrp.create_dataset("indices",  shape=(n_snaps,), dtype=vlen_u32)

            # Create normals dataset only if the first record has normals
            first_has_normals = bool(_layout_val(mesh_recs[0], "has_normals") or False)
            norm_ds = (mgrp.create_dataset("normals", shape=(n_snaps,), dtype=vlen_f32)
                       if first_has_normals else None)

            for i, rec in enumerate(mesh_recs):
                vc = int(_layout_val(rec, "vertex_count") or 0)
                ic = int(_layout_val(rec, "index_count")  or 0)
                hn = bool(_layout_val(rec, "has_normals") or False)
                si = int(_layout_val(rec, "snapshot_index") or i)

                snap_indices[i] = si
                vert_counts[i]  = vc
                idx_counts[i]   = ic

                raw = _content_bytes(rec)
                vert_bytes = vc * 3 * 4  # float32
                idx_bytes  = ic * 4      # uint32
                norm_bytes = vc * 3 * 4 if hn else 0

                if raw and len(raw) >= vert_bytes + idx_bytes:
                    p = 0
                    verts_ds[i] = np.frombuffer(raw[p:p + vert_bytes], dtype=np.float32)
                    p += vert_bytes
                    idx_ds[i]   = np.frombuffer(raw[p:p + idx_bytes],  dtype=np.uint32)
                    p += idx_bytes
                    if norm_ds is not None:
                        if hn and len(raw) >= p + norm_bytes:
                            norm_ds[i] = np.frombuffer(raw[p:p + norm_bytes], dtype=np.float32)
                        else:
                            norm_ds[i] = np.empty(0, dtype=np.float32)
                else:
                    verts_ds[i] = np.empty(0, dtype=np.float32)
                    idx_ds[i]   = np.empty(0, dtype=np.uint32)
                    if norm_ds is not None:
                        norm_ds[i] = np.empty(0, dtype=np.float32)

            mgrp.create_dataset("snapshot_index", data=snap_indices)
            mgrp.create_dataset("vertex_counts",  data=vert_counts)
            mgrp.create_dataset("index_counts",   data=idx_counts)

            mgrp.attrs["n_snapshots"]   = n_snaps
            mgrp.attrs["note"] = (
                "Spatial mesh snapshots at native ML2 meshing rate (~1 Hz). "
                "Not aligned to RGB frame timestamps — use timestamps_s to "
                "correlate with head_pose or rgb/timestamps_s. "
                "Each snapshot is a complete triangulated world mesh within "
                "the capture query bounds. "
                "vertices[i]: flat float32 x,y,z triples — reshape(-1,3) for Nx3. "
                "indices[i]:  flat uint32 triangle indices — reshape(-1,3) for Mx3 faces. "
                "normals[i]:  flat float32 x,y,z triples (present when recorded)."
            )
            mgrp.attrs["vertex_layout"] = "float32 [x,y,z, x,y,z, ...]; reshape(-1,3) → (N,3)"
            mgrp.attrs["index_layout"]  = "uint32 [a,b,c, ...]; reshape(-1,3) → (M,3) triangles"
            mgrp.attrs["frame"]         = "ML2 world frame, same as head_pose (gravity-aligned)"
            mgrp.attrs["units"]         = "meters"
            mgrp.attrs["clock_domain"]  = "CLOCK_BOOTTIME, same as root timestamps_ns"
            print(f"    {n_snaps} snapshots written")

        # ── Calibration ───────────────────────────────────────────────────
        # Every cam named in `calibration` or `aligned_poses` gets a group.
        cam_names = sorted(set(calibration.keys()) | set(aligned_poses.keys()))
        if cam_names:
            print("  Writing calibration...")
            cal_grp = hf.create_group("calibration")
            cal_grp.attrs["note"] = (
                "Per-camera intrinsics (from VRS CONFIGURATION records or file tags) "
                "and derived camera_to_head extrinsics. camera_to_head is the "
                "median of head_from_camera across all valid frames and is a "
                "rigid-body constant: the stddev datasets are a correctness check "
                "(low stddev → SDK pose is self-consistent). "
                "IMU-to-camera and camera-to-camera transforms are not "
                "captured — use camera_to_head + head_pose to compose them at "
                "read time."
            )
            cal_grp.attrs["distortion_model"] = (
                "Polynomial radial-tangential (OpenCV convention): "
                "[k1, k2, p1, p2, k3]. Not available for rgb."
            )

            # Head pose in HDF5 is stored as pos[:,:3] + ori[:,3:] (w,x,y,z).
            head_pos_arr = head_interp[:, :3] if head_interp is not None else None
            head_ori_arr = head_interp[:, 3:] if head_interp is not None else None

            for cam_name in cam_names:
                cg = cal_grp.create_group(cam_name)
                intr = calibration.get(cam_name)
                if intr is not None:
                    cg.attrs["fx"] = intr["fx"]
                    cg.attrs["fy"] = intr["fy"]
                    if intr.get("cx") is not None: cg.attrs["cx"] = intr["cx"]
                    if intr.get("cy") is not None: cg.attrs["cy"] = intr["cy"]
                    if intr.get("width")  is not None: cg.attrs["width"]  = intr["width"]
                    if intr.get("height") is not None: cg.attrs["height"] = intr["height"]
                    if intr.get("distortion") is not None:
                        cg.create_dataset("distortion",
                                          data=np.array(intr["distortion"], dtype=np.float32))
                    cg.attrs["units"] = "pixels (fx, fy, cx, cy)"

                # Derived camera_to_head (rigid). Requires head_pose + per-frame
                # camera_pose for this camera, both aligned to RGB timestamps.
                if cam_name not in aligned_poses or head_pos_arr is None:
                    continue
                cam_pos, cam_ori, cam_valid = aligned_poses[cam_name]
                t_hc, q_hc, v_hc = compute_head_from_camera(
                    cam_pos, cam_ori, cam_valid, head_pos_arr, head_ori_arr)
                n_valid = int(v_hc.sum())
                if n_valid == 0:
                    continue

                valid_mask = v_hc == 1
                t_valid = t_hc[valid_mask]
                q_valid = q_hc[valid_mask]

                t_median = np.median(t_valid, axis=0)
                t_std    = t_valid.std(axis=0)
                q_mean   = _quat_avg(q_valid)

                # Rotation stddev (degrees): angle of each frame's rotation
                # relative to the mean.
                q_rel = _quat_mul(q_valid, _quat_conj(q_mean[None, :]))
                w_rel = np.clip(q_rel[:, 0], -1.0, 1.0)
                angles_deg = np.degrees(2.0 * np.arccos(np.abs(w_rel)))
                q_std_deg = float(angles_deg.std())

                ext = cg.create_group("camera_to_head")
                ext.create_dataset("position",    data=t_median, dtype=np.float64)
                ext.create_dataset("orientation", data=q_mean,   dtype=np.float64)
                ext.create_dataset("position_stddev",       data=t_std, dtype=np.float64)
                ext.attrs["orientation_stddev_deg"] = q_std_deg
                ext.attrs["n_valid_frames"]         = n_valid
                ext.attrs["n_total_frames"]         = int(cam_valid.size)
                ext.attrs["columns_position"]       = "x,y,z"
                ext.attrs["units_position"]         = "meters"
                ext.attrs["columns_orientation"]    = "w,x,y,z"
                ext.attrs["convention_orientation"] = (
                    "Hamilton quaternion, scalar-first (w,x,y,z), "
                    "head_from_camera. Apply as: p_head = rotate(q, p_cam) + t."
                )
                ext.attrs["rigidity_check"] = (
                    "For a hardware-mounted camera these stddev values should be "
                    "near zero (millimetres for position, <1 deg for orientation). "
                    "Large values indicate SDK pose drift or a bug."
                )
            print(f"    cameras: {', '.join(cam_names)}")

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\nDone! Output: {output_path} ({size_mb:.1f} MB)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert ML2 recorder VRS session to HDF5")
    parser.add_argument("vrs_path", type=Path,
                        help="Path to session .vrs file")
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="Output HDF5 path (default: <session>.h5)")
    parser.add_argument("--interp", choices=["nearest", "linear"],
                        default="nearest",
                        help="Interpolation method (default: nearest)")
    parser.add_argument("--no-images",     action="store_true",
                        help="Skip RGB/depth/world images")
    parser.add_argument("--no-depth",      action="store_true")
    parser.add_argument("--no-world-cams", action="store_true")
    parser.add_argument("--no-audio",      action="store_true")

    args = parser.parse_args()

    vrs_path = args.vrs_path.resolve()

    # Accept a bare session name or directory — resolve to the .vrs file.
    if vrs_path.is_dir() or (not vrs_path.exists() and vrs_path.suffix != ".vrs"):
        candidate = vrs_path.with_suffix(".vrs")
        if candidate.exists():
            vrs_path = candidate
        else:
            # Maybe the directory contains a single .vrs file
            matches = list(vrs_path.parent.glob(vrs_path.name + "*.vrs")) if not vrs_path.is_dir() \
                      else list(vrs_path.glob("*.vrs"))
            if len(matches) == 1:
                vrs_path = matches[0]
            else:
                print(f"ERROR: File not found: {vrs_path}")
                print(f"  Expected a .vrs file.  Tried: {vrs_path.with_suffix('.vrs')}")
                sys.exit(1)

    if not vrs_path.exists():
        print(f"ERROR: File not found: {vrs_path}")
        sys.exit(1)

    output = args.output or vrs_path.with_suffix(".h5")

    convert_session(
        vrs_path=vrs_path,
        output_path=output,
        interp_method=args.interp,
        include_images=not args.no_images,
        include_depth=not args.no_depth,
        include_world_cams=not args.no_world_cams,
        include_audio=not args.no_audio,
    )


if __name__ == "__main__":
    main()

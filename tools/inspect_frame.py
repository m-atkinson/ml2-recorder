#!/usr/bin/env python3
"""
inspect_frame.py — Point-in-time data inspector for ML2 session HDF5 files.

Given a timestamp (or frame index), prints the values of every sensor stream
at that moment — head pose, eye tracking, hand keypoints, IMU, and depth stats.
All non-RGB streams are nearest-neighbour matched to the requested time, with
the temporal delta shown so you know how stale each reading is.

The timestamps is seconds since device boot, not seconds since the start of the session.

Usage:
    python inspect_frame.py session.h5 --timestamp 47.832
    python inspect_frame.py session.h5 --frame 384
    python inspect_frame.py session.h5 -t 47.832 --streams head eye imu
    python inspect_frame.py session.h5 -t 47.832 --json

Requires:
    pip install h5py numpy
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np

# ── ANSI colours ──────────────────────────────────────────────────────────────

BOLD  = "\033[1m"
DIM   = "\033[2m"
CYAN  = "\033[36m"
GREEN = "\033[32m"
YELLOW= "\033[33m"
RED   = "\033[31m"
RESET = "\033[0m"

HAND_KEYPOINT_NAMES = [
    "thumb_tip",    "thumb_ip",      "thumb_mcp",     "thumb_cmc",
    "index_tip",    "index_dip",     "index_pip",     "index_mcp",
    "middle_tip",   "middle_dip",    "middle_pip",     "middle_mcp",
    "ring_tip",     "ring_dip",      "ring_pip",       "ring_mcp",
    "pinky_tip",    "pinky_dip",     "pinky_pip",      "pinky_mcp",
    "wrist_center", "wrist_ulnar",   "wrist_radial",
    "hand_center",  "index_meta",    "middle_meta",    "ring_meta",  "pinky_meta",
]


# ── helpers ───────────────────────────────────────────────────────────────────

def _nearest(target: float, ts: np.ndarray) -> Tuple[int, float]:
    """Return (index, delta_s) of the nearest timestamp to target."""
    if len(ts) == 0:
        return -1, float("nan")
    idx = int(np.searchsorted(ts, target))
    idx = min(max(idx, 1), len(ts) - 1)
    left, right = idx - 1, idx
    if abs(target - ts[left]) <= abs(target - ts[right]):
        idx = left
    else:
        idx = right
    return idx, float(target - ts[idx])


def _fmt_delta(delta_s: float) -> str:
    ms = delta_s * 1000
    sign = "+" if ms >= 0 else ""
    col = GREEN if abs(ms) < 5 else YELLOW if abs(ms) < 50 else RED
    return f"{col}{sign}{ms:.1f} ms{RESET}"


def _vec3(arr) -> str:
    if arr is None or np.any(np.isnan(arr)):
        return f"{DIM}nan{RESET}"
    return f"({arr[0]:+.4f}, {arr[1]:+.4f}, {arr[2]:+.4f})"


def _section(title: str) -> None:
    print(f"\n{BOLD}{CYAN}── {title} {RESET}" + "─" * max(0, 60 - len(title)))


# ── per-stream printers ───────────────────────────────────────────────────────

def show_head_pose(f: h5py.File, frame_idx: int) -> Dict[str, Any]:
    if "head_pose" not in f:
        print(f"  {DIM}stream not present{RESET}")
        return {}
    pos = f["head_pose/position"][frame_idx]
    ori = f["head_pose/orientation"][frame_idx]
    print(f"  Position   (x,y,z m):   {_vec3(pos)}")
    print(f"  Quaternion (w,x,y,z):    ({ori[0]:+.5f}, {ori[1]:+.5f},"
          f" {ori[2]:+.5f}, {ori[3]:+.5f})")
    return {"position_m": pos.tolist(), "orientation_wxyz": ori.tolist()}


def show_eye_tracking(f: h5py.File, frame_idx: int) -> Dict[str, Any]:
    if "eye_tracking" not in f:
        print(f"  {DIM}stream not present{RESET}")
        return {}
    lo  = f["eye_tracking/left_origin"][frame_idx]
    ld  = f["eye_tracking/left_direction"][frame_idx]
    ro  = f["eye_tracking/right_origin"][frame_idx]
    rd  = f["eye_tracking/right_direction"][frame_idx]
    fp  = f["eye_tracking/fixation_point"][frame_idx]

    print(f"  Left  origin    (m):  {_vec3(lo)}")
    print(f"  Left  direction (unit): {_vec3(ld)}")
    print(f"  Right origin    (m):  {_vec3(ro)}")
    print(f"  Right direction (unit): {_vec3(rd)}")
    fp_valid = not np.any(np.isnan(fp)) and np.any(fp != 0)
    print(f"  Fixation point  (m):  {_vec3(fp)}"
          + (f"  {DIM}(no valid fixation){RESET}" if not fp_valid else ""))
    return {
        "left_origin_m":     lo.tolist(),
        "left_direction":    ld.tolist(),
        "right_origin_m":    ro.tolist(),
        "right_direction":   rd.tolist(),
        "fixation_point_m":  fp.tolist(),
    }


def show_hand_tracking(f: h5py.File, frame_idx: int,
                       compact: bool = False) -> Dict[str, Any]:
    if "hand_tracking" not in f:
        print(f"  {DIM}stream not present{RESET}")
        return {}

    lc = float(f["hand_tracking/left_confidence"][frame_idx])  if "hand_tracking/left_confidence"  in f else float("nan")
    rc = float(f["hand_tracking/right_confidence"][frame_idx]) if "hand_tracking/right_confidence" in f else float("nan")
    lj = f["hand_tracking/left_joints"][frame_idx]   # 28×3
    rj = f["hand_tracking/right_joints"][frame_idx]  # 28×3

    def conf_str(c: float) -> str:
        col = GREEN if c > 0.7 else YELLOW if c > 0 else RED
        return f"{col}{c:.3f}{RESET}"

    print(f"  Left  confidence: {conf_str(lc)}"
          + (f"  {DIM}(not detected){RESET}" if lc == 0 else ""))
    print(f"  Right confidence: {conf_str(rc)}"
          + (f"  {DIM}(not detected){RESET}" if rc == 0 else ""))

    out: Dict[str, Any] = {"left_confidence": lc, "right_confidence": rc,
                           "left_joints_m": {}, "right_joints_m": {}}

    for side, joints, conf in [("Left", lj, lc), ("Right", rj, rc)]:
        if conf == 0:
            print(f"\n  {side} hand: {DIM}not detected — all joints NaN{RESET}")
            continue
        n_valid  = int(np.sum(~np.any(np.isnan(joints), axis=1)))
        n_total  = len(joints)
        print(f"\n  {side} hand keypoints  ({n_valid}/{n_total} valid):")
        for i, (name, pt) in enumerate(zip(HAND_KEYPOINT_NAMES, joints)):
            is_nan = np.any(np.isnan(pt))
            if compact and is_nan:
                continue
            val = f"{DIM}NaN{RESET}" if is_nan else _vec3(pt)
            print(f"    {name:<18} {val}")
        key = "left_joints_m" if side == "Left" else "right_joints_m"
        out[key] = {name: pt.tolist() for name, pt in zip(HAND_KEYPOINT_NAMES, joints)}

    return out


def show_imu(f: h5py.File, query_ts: float) -> Dict[str, Any]:
    if "imu" not in f or "imu/timestamps_s" not in f:
        print(f"  {DIM}stream not present{RESET}")
        return {}

    imu_ts = f["imu/timestamps_s"][:]
    idx, delta = _nearest(query_ts, imu_ts)
    if idx < 0:
        print(f"  {DIM}no IMU samples{RESET}")
        return {}

    acc = f["imu/accelerometer"][idx]
    gyr = f["imu/gyroscope"][idx]
    uid = int(f["imu/unit_id"][idx]) if "imu/unit_id" in f else -1

    print(f"  Nearest sample: index {idx}  Δ = {_fmt_delta(delta)}"
          + (f"  unit_id={uid}" if uid >= 0 else ""))
    print(f"  Accelerometer (m/s²): ax={acc[0]:+.4f}  ay={acc[1]:+.4f}  az={acc[2]:+.4f}"
          f"  |a|={np.linalg.norm(acc):.3f}")
    print(f"  Gyroscope (rad/s):    gx={gyr[0]:+.4f}  gy={gyr[1]:+.4f}  gz={gyr[2]:+.4f}"
          f"  |g|={np.linalg.norm(gyr):.4f}")

    return {
        "imu_index":       idx,
        "delta_s":         delta,
        "unit_id":         uid,
        "accelerometer":   acc.tolist(),
        "gyroscope":       gyr.tolist(),
        "accel_norm_mps2": float(np.linalg.norm(acc)),
    }


def show_depth(f: h5py.File, frame_idx: int) -> Dict[str, Any]:
    if "depth" not in f or "depth/images" not in f:
        print(f"  {DIM}stream not present{RESET}")
        return {}

    frame = f["depth/images"][frame_idx]
    h, w  = frame.shape
    valid = frame[frame > 0].astype(np.float32)
    valid_pct = 100.0 * len(valid) / (h * w) if h * w > 0 else 0.0

    print(f"  Resolution: {w}×{h}  ({valid_pct:.1f}% valid pixels)")
    if len(valid) > 0:
        print(f"  Depth range (mm): {valid.min():.0f} – {valid.max():.0f}"
              f"   median={np.median(valid):.0f}   mean={valid.mean():.0f}")
        p10, p90 = np.percentile(valid, [10, 90])
        print(f"  10th/90th pct:    {p10:.0f} / {p90:.0f} mm")
    else:
        print(f"  {RED}All pixels invalid (0){RESET}")

    out: Dict[str, Any] = {
        "resolution": [w, h],
        "valid_pct": valid_pct,
    }
    if len(valid) > 0:
        out.update({
            "min_mm": float(valid.min()),
            "max_mm": float(valid.max()),
            "median_mm": float(np.median(valid)),
            "mean_mm": float(valid.mean()),
            "p10_mm": float(p10),
            "p90_mm": float(p90),
        })
    return out


def show_mesh(f: h5py.File, query_ts: float) -> Dict[str, Any]:
    if "mesh" not in f or "mesh/timestamps_s" not in f:
        print(f"  {DIM}stream not present{RESET}")
        return {}

    mesh_ts = f["mesh/timestamps_s"][:]
    idx, delta = _nearest(query_ts, mesh_ts)
    if idx < 0:
        print(f"  {DIM}no mesh snapshots{RESET}")
        return {}

    snap_idx  = int(f["mesh/snapshot_index"][idx])
    n_verts   = int(f["mesh/vertex_counts"][idx])
    n_idx     = int(f["mesh/index_counts"][idx])
    n_tris    = n_idx // 3
    has_norms = "mesh/normals" in f

    print(f"  Nearest snapshot: #{snap_idx}  Δ = {_fmt_delta(delta)}"
          f"  (t = {float(mesh_ts[idx]):.3f} s)")
    print(f"  Vertices: {n_verts:,}   Triangles: {n_tris:,}"
          + (f"   normals: {GREEN}yes{RESET}" if has_norms else f"   normals: {DIM}no{RESET}"))

    out: Dict[str, Any] = {
        "snapshot_index": snap_idx,
        "delta_s":        delta,
        "n_vertices":     n_verts,
        "n_triangles":    n_tris,
        "has_normals":    has_norms,
    }

    verts_flat = np.array(f["mesh/vertices"][idx], dtype=np.float32)
    if len(verts_flat) > 0:
        verts  = verts_flat.reshape(-1, 3)
        mins   = verts.min(axis=0)
        maxs   = verts.max(axis=0)
        extent = maxs - mins
        center = (mins + maxs) / 2.0
        print(f"  Bounding box (m):")
        print(f"    min:    {_vec3(mins)}")
        print(f"    max:    {_vec3(maxs)}")
        print(f"    extent: ({extent[0]:.3f}, {extent[1]:.3f}, {extent[2]:.3f})")
        print(f"    center: {_vec3(center)}")
        out.update({
            "bbox_min_m":    mins.tolist(),
            "bbox_max_m":    maxs.tolist(),
            "bbox_extent_m": extent.tolist(),
            "bbox_center_m": center.tolist(),
        })

    return out


# ── main inspector ────────────────────────────────────────────────────────────

ALL_STREAMS = ["head", "eye", "hand", "imu", "depth", "mesh"]


def inspect(h5_path: Path, query_ts: Optional[float], frame_idx: Optional[int],
            streams: List[str], compact_hands: bool, as_json: bool) -> None:

    with h5py.File(h5_path, "r") as f:
        if "timestamps_s" not in f:
            print("ERROR: timestamps_s not found — is this a valid ML2 HDF5 file?")
            sys.exit(1)

        rgb_ts   = f["timestamps_s"][:]
        n_frames = len(rgb_ts)

        # ── resolve frame index ───────────────────────────────────────────
        if frame_idx is not None:
            if not (0 <= frame_idx < n_frames):
                print(f"ERROR: frame {frame_idx} out of range [0, {n_frames - 1}]")
                sys.exit(1)
            fidx  = frame_idx
            qts   = float(rgb_ts[fidx])
            delta = 0.0
        else:
            # query_ts guaranteed to be set (argparse enforces one of the two)
            qts   = query_ts
            fidx, delta = _nearest(qts, rgb_ts)

        actual_ts = float(rgb_ts[fidx])

        # ── header ────────────────────────────────────────────────────────
        if not as_json:
            t0 = float(rgb_ts[0])
            t1 = float(rgb_ts[-1])
            print(f"\n{BOLD}ML2 Frame Inspector — {h5_path.name}{RESET}")
            print(f"Session: {t0:.3f} s → {t1:.3f} s   ({n_frames} RGB frames)")
            if frame_idx is not None:
                print(f"Query:   frame #{fidx}   t = {actual_ts:.6f} s")
            else:
                print(f"Query:   t = {qts:.6f} s")
                print(f"Nearest: frame #{fidx} / {n_frames - 1}   "
                      f"Δ = {_fmt_delta(delta)}   "
                      f"(t = {actual_ts:.6f} s)")

        result: Dict[str, Any] = {
            "query_timestamp_s":  qts if frame_idx is None else actual_ts,
            "frame_index":        fidx,
            "actual_timestamp_s": actual_ts,
            "delta_s":            delta,
            "n_frames":           n_frames,
            "streams":            {},
        }

        # ── streams ───────────────────────────────────────────────────────
        if "head" in streams:
            if not as_json:
                _section("HEAD POSE")
            result["streams"]["head_pose"] = show_head_pose(f, fidx) if not as_json else _json_head(f, fidx)

        if "eye" in streams:
            if not as_json:
                _section("EYE TRACKING")
            result["streams"]["eye_tracking"] = show_eye_tracking(f, fidx) if not as_json else _json_eye(f, fidx)

        if "hand" in streams:
            if not as_json:
                _section("HAND TRACKING")
            result["streams"]["hand_tracking"] = show_hand_tracking(f, fidx, compact=compact_hands) if not as_json else _json_hand(f, fidx)

        if "imu" in streams:
            if not as_json:
                _section("IMU (full-rate stream, nearest sample)")
            result["streams"]["imu"] = show_imu(f, actual_ts) if not as_json else _json_imu(f, actual_ts)

        if "depth" in streams:
            if not as_json:
                _section("DEPTH")
            result["streams"]["depth"] = show_depth(f, fidx) if not as_json else _json_depth(f, fidx)

        if "mesh" in streams:
            if not as_json:
                _section("MESH (nearest snapshot)")
            result["streams"]["mesh"] = show_mesh(f, actual_ts) if not as_json else _json_mesh(f, actual_ts)

        if not as_json:
            print()
        else:
            print(json.dumps(result, indent=2, default=_json_default))


# ── JSON-only helpers (collect data without printing) ─────────────────────────

def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _json_head(f: h5py.File, i: int) -> Dict:
    if "head_pose" not in f:
        return {}
    return {
        "position_m":      f["head_pose/position"][i].tolist(),
        "orientation_wxyz": f["head_pose/orientation"][i].tolist(),
    }


def _json_eye(f: h5py.File, i: int) -> Dict:
    if "eye_tracking" not in f:
        return {}
    return {
        "left_origin_m":    f["eye_tracking/left_origin"][i].tolist(),
        "left_direction":   f["eye_tracking/left_direction"][i].tolist(),
        "right_origin_m":   f["eye_tracking/right_origin"][i].tolist(),
        "right_direction":  f["eye_tracking/right_direction"][i].tolist(),
        "fixation_point_m": f["eye_tracking/fixation_point"][i].tolist(),
    }


def _json_hand(f: h5py.File, i: int) -> Dict:
    if "hand_tracking" not in f:
        return {}
    lc = float(f["hand_tracking/left_confidence"][i])  if "hand_tracking/left_confidence"  in f else None
    rc = float(f["hand_tracking/right_confidence"][i]) if "hand_tracking/right_confidence" in f else None
    lj = f["hand_tracking/left_joints"][i]
    rj = f["hand_tracking/right_joints"][i]
    return {
        "left_confidence":  lc,
        "right_confidence": rc,
        "left_joints_m":  {n: p.tolist() for n, p in zip(HAND_KEYPOINT_NAMES, lj)},
        "right_joints_m": {n: p.tolist() for n, p in zip(HAND_KEYPOINT_NAMES, rj)},
    }


def _json_imu(f: h5py.File, query_ts: float) -> Dict:
    if "imu" not in f or "imu/timestamps_s" not in f:
        return {}
    imu_ts = f["imu/timestamps_s"][:]
    idx, delta = _nearest(query_ts, imu_ts)
    if idx < 0:
        return {}
    acc = f["imu/accelerometer"][idx]
    gyr = f["imu/gyroscope"][idx]
    uid = int(f["imu/unit_id"][idx]) if "imu/unit_id" in f else -1
    return {
        "imu_index": idx, "delta_s": delta, "unit_id": uid,
        "accelerometer": acc.tolist(), "gyroscope": gyr.tolist(),
        "accel_norm_mps2": float(np.linalg.norm(acc)),
    }


def _json_depth(f: h5py.File, i: int) -> Dict:
    if "depth" not in f or "depth/images" not in f:
        return {}
    frame = f["depth/images"][i]
    h, w  = frame.shape
    valid = frame[frame > 0].astype(np.float32)
    valid_pct = 100.0 * len(valid) / (h * w) if h * w > 0 else 0.0
    out: Dict[str, Any] = {"resolution": [w, h], "valid_pct": valid_pct}
    if len(valid) > 0:
        p10, p90 = np.percentile(valid, [10, 90])
        out.update({
            "min_mm": float(valid.min()), "max_mm": float(valid.max()),
            "median_mm": float(np.median(valid)), "mean_mm": float(valid.mean()),
            "p10_mm": float(p10), "p90_mm": float(p90),
        })
    return out


def _json_mesh(f: h5py.File, query_ts: float) -> Dict:
    if "mesh" not in f or "mesh/timestamps_s" not in f:
        return {}
    mesh_ts = f["mesh/timestamps_s"][:]
    idx, delta = _nearest(query_ts, mesh_ts)
    if idx < 0:
        return {}
    snap_idx  = int(f["mesh/snapshot_index"][idx])
    n_verts   = int(f["mesh/vertex_counts"][idx])
    n_idx     = int(f["mesh/index_counts"][idx])
    has_norms = "mesh/normals" in f
    out: Dict[str, Any] = {
        "snapshot_index": snap_idx,
        "delta_s":        delta,
        "n_vertices":     n_verts,
        "n_triangles":    n_idx // 3,
        "has_normals":    has_norms,
    }
    verts_flat = np.array(f["mesh/vertices"][idx], dtype=np.float32)
    if len(verts_flat) > 0:
        verts  = verts_flat.reshape(-1, 3)
        mins   = verts.min(axis=0)
        maxs   = verts.max(axis=0)
        out.update({
            "bbox_min_m":    mins.tolist(),
            "bbox_max_m":    maxs.tolist(),
            "bbox_extent_m": (maxs - mins).tolist(),
            "bbox_center_m": ((mins + maxs) / 2.0).tolist(),
        })
    return out


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect all sensor data at a precise timestamp in an ML2 HDF5 session")
    parser.add_argument("h5_path", type=Path, help="Path to session .h5 file")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-t", "--timestamp", type=float, metavar="SEC",
                       help="Query timestamp in seconds (matched to nearest RGB frame)")
    group.add_argument("-f", "--frame", type=int, metavar="IDX",
                       help="Frame index (0-based)")

    parser.add_argument("--streams", nargs="+", choices=ALL_STREAMS,
                        default=ALL_STREAMS,
                        help="Which streams to show (default: all)")
    parser.add_argument("--compact-hands", action="store_true",
                        help="Skip NaN keypoints in hand output")
    parser.add_argument("--json", action="store_true",
                        help="Output machine-readable JSON instead of formatted text")

    args = parser.parse_args()
    path = args.h5_path.resolve()

    if not path.exists():
        print(f"ERROR: File not found: {path}")
        sys.exit(1)

    inspect(
        h5_path=path,
        query_ts=args.timestamp,
        frame_idx=args.frame,
        streams=args.streams,
        compact_hands=args.compact_hands,
        as_json=args.json,
    )


if __name__ == "__main__":
    main()

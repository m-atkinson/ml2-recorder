#!/usr/bin/env python3
"""
Visualize an ML2 session HDF5 file in Rerun.

Requires: pip install rerun-sdk av h5py numpy

Usage:
    python tools/view_rerun.py pulled_sessions/session.h5
    python tools/view_rerun.py session.h5 --save output.rrd
    python tools/view_rerun.py session.h5 --frames 0:200
    python tools/view_rerun.py session.h5 --no-world-cams --no-imu
"""

import argparse
import sys
from pathlib import Path

import numpy as np


def _check_deps():
    missing = []
    for mod, pkg in [("h5py", "h5py"), ("rerun", "rerun-sdk"), ("av", "av")]:
        try:
            __import__(mod)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"Missing packages: {', '.join(missing)}")
        print(f"Install: pip install {' '.join(missing)}")
        sys.exit(1)


_check_deps()

import av  # noqa: E402
import h5py  # noqa: E402
import rerun as rr  # noqa: E402

# ── Hand skeleton ───────────────────────────────────────────────────────────
# 28 keypoints (ML2 SDK order):
#  0-3:  thumb  (tip, ip, mcp, cmc)
#  4-7:  index  (tip, dip, pip, mcp)
#  8-11: middle (tip, dip, pip, mcp)
# 12-15: ring   (tip, dip, pip, mcp)
# 16-19: pinky  (tip, dip, pip, mcp)
# 20:wrist_center 21:wrist_ulnar 22:wrist_radial 23:hand_center
# 24:index_meta 25:middle_meta 26:ring_meta 27:pinky_meta
HAND_BONES = [
    (0, 1), (1, 2), (2, 3),               # thumb
    (4, 5), (5, 6), (6, 7), (7, 24),      # index
    (8, 9), (9, 10), (10, 11), (11, 25),  # middle
    (12, 13), (13, 14), (14, 15), (15, 26),  # ring
    (16, 17), (17, 18), (18, 19), (19, 27),  # pinky
    (3, 24), (24, 25), (25, 26), (26, 27),   # palm arch
    (20, 24), (20, 27),                       # wrist to palm
]


def _hand_strips(joints):
    """Return valid bone segments as a list of 2-point strips (NaN joints skipped)."""
    strips = []
    for a, b in HAND_BONES:
        if not (np.any(np.isnan(joints[a])) or np.any(np.isnan(joints[b]))):
            strips.append([joints[a].tolist(), joints[b].tolist()])
    return strips


def _decode_h264(rgb_ds, start, end):
    """
    Yield (frame_index, rgb_ndarray) pairs from H.264 NAL units stored in HDF5.
    NAL units are Annex B formatted; SPS/PPS config NALs produce no frames and
    are consumed silently by PyAV.

    frame_index is the HDF5 index i of the NAL that produced the frame, so
    ts[frame_index] is always the correct timestamp — even when config NALs
    (SPS/PPS-only entries) are interspersed among the keyframes.
    """
    codec = av.CodecContext.create("h264", "r")
    last_i = start
    for i in range(start, end):
        nal = bytes(rgb_ds[i])
        if not nal:
            continue
        try:
            for pkt in codec.parse(nal):
                for frame in codec.decode(pkt):
                    yield i, frame.to_ndarray(format="rgb24")
                    last_i = i
        except Exception:
            pass
    # flush any frames the decoder held back (rare for I/P-only streams)
    try:
        for pkt in codec.parse(b""):
            for frame in codec.decode(pkt):
                yield last_i, frame.to_ndarray(format="rgb24")
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="Visualize ML2 HDF5 session in Rerun")
    parser.add_argument("hdf5", type=Path, help="Path to session .h5 file")
    parser.add_argument("--save", type=Path, default=None,
                        help="Save to .rrd file instead of launching viewer")
    parser.add_argument("--frames", type=str, default=None,
                        help="Frame range (e.g. 0:500)")
    parser.add_argument("--no-world-cams", action="store_true",
                        help="Skip world cameras")
    parser.add_argument("--no-imu", action="store_true",
                        help="Skip IMU timeseries")
    args = parser.parse_args()

    with h5py.File(args.hdf5, "r") as hf:
        n_frames = int(hf.attrs["n_frames"])

    start, end = 0, n_frames
    if args.frames:
        parts = args.frames.split(":")
        start = int(parts[0]) if parts[0] else 0
        end   = int(parts[1]) if len(parts) > 1 and parts[1] else n_frames
        end   = min(end, n_frames)
    n = end - start

    print(f"Session: {args.hdf5.name}  ({n_frames} frames total, logging {start}:{end})")

    rr.init("ml2_session", spawn=(args.save is None))
    if args.save:
        rr.save(str(args.save))

    with h5py.File(args.hdf5, "r") as hf:
        ts = hf["timestamps_s"][:]  # (N,)

        # ── RGB ─────────────────────────────────────────────────────────────
        print(f"[1/5] Decoding H.264 RGB ({n} frames)...")
        for frame_i, rgb in _decode_h264(hf["rgb/images"], start, end):
            rr.set_time("frame", sequence=frame_i)
            rr.set_time("time", duration=ts[frame_i])
            rr.log("camera/rgb", rr.Image(rgb))
            if (frame_i - start) % 100 == 0:
                print(f"      {frame_i - start}/{n}", end="\r")
        print(f"      {n}/{n}")

        # ── Load frame-aligned arrays ────────────────────────────────────────
        head_pos   = hf["head_pose/position"][:]      # (N, 3)
        head_ori   = hf["head_pose/orientation"][:]   # (N, 4) w,x,y,z
        depth_ds   = hf["depth/images"] if "depth" in hf else None

        has_eye = "eye_tracking" in hf
        if has_eye:
            lo = hf["eye_tracking/left_origin"][:]
            ld = hf["eye_tracking/left_direction"][:]
            ro = hf["eye_tracking/right_origin"][:]
            rd = hf["eye_tracking/right_direction"][:]
            fp = hf["eye_tracking/fixation_point"][:]
        else:
            print("      (eye_tracking not present in this session)")

        has_hands = "hand_tracking" in hf
        if has_hands:
            lj   = hf["hand_tracking/left_joints"][:]    # (N, 28, 3)
            rj   = hf["hand_tracking/right_joints"][:]
            lc   = hf["hand_tracking/left_confidence"][:]
            rc_  = hf["hand_tracking/right_confidence"][:]

        world_cams = []
        if not args.no_world_cams:
            for key, path in [("world_cam_0", "camera/world_0"),
                               ("world_cam_1", "camera/world_1"),
                               ("world_cam_2", "camera/world_2")]:
                if key in hf:
                    world_cams.append((hf[key + "/images"], path))

        # ── Per-frame spatial streams ────────────────────────────────────────
        print(f"[2/5] Logging spatial streams ({n} frames)...")
        for i in range(start, end):
            rr.set_time("frame", sequence=i)
            rr.set_time("time", duration=ts[i])

            # Depth (uint16 mm)
            if depth_ds is not None:
                rr.log("camera/depth", rr.DepthImage(depth_ds[i], meter=0.001))

            # Head pose — quaternion stored as w,x,y,z; Rerun needs x,y,z,w
            q = head_ori[i]
            rr.log("world/head", rr.Transform3D(
                translation=head_pos[i],
                rotation=rr.datatypes.Quaternion(xyzw=[q[1], q[2], q[3], q[0]]),
            ))

            # Eye gaze arrows
            if has_eye:
                if not np.any(np.isnan(lo[i])):
                    rr.log("world/gaze/left", rr.Arrows3D(
                        origins=[lo[i]], vectors=[ld[i] * 0.5],
                        colors=[[0, 200, 80, 255]],
                    ))
                if not np.any(np.isnan(ro[i])):
                    rr.log("world/gaze/right", rr.Arrows3D(
                        origins=[ro[i]], vectors=[rd[i] * 0.5],
                        colors=[[200, 50, 50, 255]],
                    ))
                if not np.any(np.isnan(fp[i])):
                    rr.log("world/gaze/fixation", rr.Points3D(
                        [fp[i]], radii=0.02, colors=[[255, 220, 0, 200]],
                    ))

            # Hands
            if has_hands:
                if lc[i] > 0:
                    valid = ~np.any(np.isnan(lj[i]), axis=1)
                    if valid.any():
                        rr.log("world/hands/left", rr.Points3D(
                            lj[i][valid], radii=0.008, colors=[[255, 220, 0, 220]],
                        ))
                    strips = _hand_strips(lj[i])
                    if strips:
                        rr.log("world/hands/left/skeleton", rr.LineStrips3D(
                            strips, colors=[[255, 200, 0, 180]], radii=0.003,
                        ))

                if rc_[i] > 0:
                    valid = ~np.any(np.isnan(rj[i]), axis=1)
                    if valid.any():
                        rr.log("world/hands/right", rr.Points3D(
                            rj[i][valid], radii=0.008, colors=[[0, 200, 255, 220]],
                        ))
                    strips = _hand_strips(rj[i])
                    if strips:
                        rr.log("world/hands/right/skeleton", rr.LineStrips3D(
                            strips, colors=[[0, 180, 255, 180]], radii=0.003,
                        ))

            # World cameras (JPEG, passed through as encoded blobs)
            for cam_ds, cam_path in world_cams:
                blob = bytes(cam_ds[i])
                if blob:
                    rr.log(cam_path, rr.EncodedImage(
                        contents=blob, media_type="image/jpeg",
                    ))

            if (i - start) % 50 == 0:
                print(f"      {i - start}/{n}", end="\r")
        print(f"      {n}/{n}")

        # ── IMU (batched via send_columns) ───────────────────────────────────
        if not args.no_imu:
            print("[3/5] Logging IMU...")
            imu_ts  = hf["imu/timestamps_s"][:]
            accel   = hf["imu/accelerometer"][:]
            gyro    = hf["imu/gyroscope"][:]
            mask    = (imu_ts >= ts[start]) & (imu_ts <= ts[end - 1])
            imu_ts  = imu_ts[mask]
            accel   = accel[mask]
            gyro    = gyro[mask]
            rr.send_columns(
                "imu/accel_magnitude",
                indexes=[rr.TimeColumn("time", duration=imu_ts)],
                columns=[rr.components.ScalarBatch(np.linalg.norm(accel, axis=1))],
            )
            rr.send_columns(
                "imu/gyro_magnitude",
                indexes=[rr.TimeColumn("time", duration=imu_ts)],
                columns=[rr.components.ScalarBatch(np.linalg.norm(gyro, axis=1))],
            )
            print(f"      {len(imu_ts)} samples")
        else:
            print("[3/5] IMU skipped.")

        # ── Mesh (if present) ────────────────────────────────────────────────
        if "mesh" in hf:
            mesh_ts   = hf["mesh/timestamps_s"][:]
            n_snaps   = len(mesh_ts)
            has_normals = "normals" in hf["mesh"]
            print(f"[4/5] Logging mesh ({n_snaps} snapshots)...")
            for s in range(n_snaps):
                verts = hf["mesh/vertices"][s].reshape(-1, 3)
                tris  = hf["mesh/indices"][s].reshape(-1, 3)
                normals = hf["mesh/normals"][s].reshape(-1, 3) if has_normals else None
                rr.set_time("time", duration=mesh_ts[s])
                rr.log("world/mesh", rr.Mesh3D(
                    vertex_positions=verts,
                    triangle_indices=tris,
                    vertex_normals=normals,
                ))
                print(f"      {s + 1}/{n_snaps}", end="\r")
            print(f"      {n_snaps}/{n_snaps}")
        else:
            print("[4/5] Mesh: not present in this session.")

        print("[5/5] Done.")

    if args.save:
        print(f"Saved: {args.save}")
    else:
        print("Viewer launched — close the Rerun window to exit.")


if __name__ == "__main__":
    main()

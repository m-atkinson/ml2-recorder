#!/usr/bin/env python3
"""
load_sample.py — Minimal example showing how to open a session HDF5 file
and read one frame from every stream.

Usage:
    python load_sample.py session_20260410_173748.h5

Requirements:
    pip install h5py numpy
"""

import sys
from pathlib import Path

import h5py
import numpy as np


def main():
    if len(sys.argv) < 2:
        print("Usage: python load_sample.py <session.h5>")
        sys.exit(1)

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(1)

    with h5py.File(path, "r") as f:
        # -- Primary timeline --
        timestamps = f["timestamps_s"][:]
        n_frames = len(timestamps)
        duration = timestamps[-1] - timestamps[0]
        print(f"Session: {n_frames} frames, {duration:.1f} s")

        # Pick a frame in the middle
        idx = n_frames // 2
        t = timestamps[idx]
        print(f"\nReading frame {idx} (t = {t:.3f} s):\n")

        # -- Head pose --
        pos = f["head_pose/position"][idx]       # (3,) meters
        ori = f["head_pose/orientation"][idx]     # (4,) quaternion w,x,y,z
        print(f"Head position (m):    x={pos[0]:.3f}  y={pos[1]:.3f}  z={pos[2]:.3f}")
        print(f"Head orientation:     w={ori[0]:.4f} x={ori[1]:.4f} y={ori[2]:.4f} z={ori[3]:.4f}")

        # -- Eye tracking --
        if "eye_tracking/fixation_point" in f:
            fix = f["eye_tracking/fixation_point"][idx]  # (3,) meters
            print(f"Eye fixation (m):     x={fix[0]:.3f}  y={fix[1]:.3f}  z={fix[2]:.3f}")

        # -- Hand tracking --
        if "hand_tracking" in f:
            for side in ["left", "right"]:
                conf = float(f[f"hand_tracking/{side}_confidence"][idx])
                if conf > 0:
                    joints = f[f"hand_tracking/{side}_joints"][idx]  # (28, 3) meters
                    valid = np.sum(~np.any(np.isnan(joints), axis=1))
                    print(f"{side.capitalize()} hand:  conf={conf:.2f}  valid keypoints={valid}/28")
                else:
                    print(f"{side.capitalize()} hand:  not detected")

        # -- Depth --
        depth_frame = f["depth/images"][idx]  # (480, 544) uint16 mm
        valid_mask = depth_frame > 0
        if valid_mask.any():
            print(f"Depth:   {valid_mask.sum()} valid pixels, "
                  f"range {depth_frame[valid_mask].min()}-{depth_frame[valid_mask].max()} mm")

        # -- RGB (H.264 NAL unit — needs a decoder for pixel access) --
        rgb_nal = f["rgb/images"][idx]
        print(f"RGB NAL: {len(rgb_nal)} bytes")

        # -- World cameras (JPEG) --
        for cam in ["world_cam_0", "world_cam_1", "world_cam_2"]:
            jpeg = f[f"{cam}/images"][idx]
            print(f"{cam}: {len(jpeg)} bytes JPEG")

        # -- Camera extrinsics (post-2026-04-22 sessions) --
        # `world_from_camera` per frame, plus a derived rigid camera_to_head
        # under /calibration. On pre-extrinsics sessions these fields are
        # present but valid=0 everywhere; detect with the `valid` mask.
        if "camera_pose" in f["rgb"]:
            print("\nCamera extrinsics:")
            for cam in ["rgb", "world_cam_0", "world_cam_1", "world_cam_2"]:
                cp = f[f"{cam}/camera_pose"]
                if int(cp["valid"][idx]) == 0:
                    print(f"  {cam}: no pose for this frame")
                    continue
                p = cp["position"][idx]
                q = cp["orientation"][idx]  # w, x, y, z
                print(f"  {cam}: pos=({p[0]:+.3f}, {p[1]:+.3f}, {p[2]:+.3f}) m  "
                      f"quat=({q[0]:+.3f}, {q[1]:+.3f}, {q[2]:+.3f}, {q[3]:+.3f})")

            # Static camera_to_head (rigid mount transform).
            for cam in ["rgb", "world_cam_0", "world_cam_1", "world_cam_2"]:
                ext_path = f"calibration/{cam}/camera_to_head"
                if ext_path not in f:
                    continue
                ext = f[ext_path]
                c2h_pos = ext["position"][:]
                c2h_std = ext["position_stddev"][:]
                deg_std = ext.attrs.get("orientation_stddev_deg", float("nan"))
                print(f"  {cam} camera_to_head: pos={c2h_pos.round(3).tolist()} m  "
                      f"pos_std={c2h_std.round(4).tolist()} m  "
                      f"ori_std={deg_std:.2f} deg")

        # -- IMU (nearest sample to this frame's timestamp) --
        imu_ts = f["imu/timestamps_s"][:]
        imu_idx = int(np.argmin(np.abs(imu_ts - t)))
        acc = f["imu/accelerometer"][imu_idx]
        gyr = f["imu/gyroscope"][imu_idx]
        print(f"IMU (nearest): accel=({acc[0]:+.2f}, {acc[1]:+.2f}, {acc[2]:+.2f}) m/s^2  "
              f"gyro=({gyr[0]:+.4f}, {gyr[1]:+.4f}, {gyr[2]:+.4f}) rad/s")

        # -- Mesh (nearest snapshot) --
        if "mesh/timestamps_s" in f:
            mesh_ts = f["mesh/timestamps_s"][:]
            mesh_idx = int(np.argmin(np.abs(mesh_ts - t)))
            n_verts = int(f["mesh/vertex_counts"][mesh_idx])
            n_tris = int(f["mesh/index_counts"][mesh_idx]) // 3
            print(f"Mesh (nearest): {n_verts:,} vertices, {n_tris:,} triangles")

        # -- Audio --
        audio = f["audio/pcm"]
        print(f"Audio: {audio.shape[0]:,} samples x {audio.shape[1]} channels @ 192 kHz")


if __name__ == "__main__":
    main()

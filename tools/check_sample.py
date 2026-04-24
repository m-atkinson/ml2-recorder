#!/usr/bin/env python3
"""
check_sample.py — quick hand-tracking coverage report for a session HDF5.

Usage:
    python tools/check_sample.py path/to/session.h5
"""
import argparse
import sys

import h5py
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hand-tracking coverage check for a session HDF5 file.")
    parser.add_argument("h5_path", help="Path to session HDF5 file")
    args = parser.parse_args()

    print("=" * 50)
    print("Hand-Tracking Coverage Report")
    print(f"Target file: {args.h5_path}")
    print("=" * 50)

    try:
        with h5py.File(args.h5_path, "r") as f:
            left = f["/hand_tracking/left_joints"][:]
            right = f["/hand_tracking/right_joints"][:]
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    total_frames = left.shape[0]
    total_floats_per_frame = 28 * 3

    left_nans_per_frame = np.isnan(left).sum(axis=(1, 2))
    right_nans_per_frame = np.isnan(right).sum(axis=(1, 2))

    left_fully_missing = (left_nans_per_frame == total_floats_per_frame).sum()
    right_fully_missing = (right_nans_per_frame == total_floats_per_frame).sum()

    left_pct = np.sum(left_nans_per_frame) / (total_frames * total_floats_per_frame) * 100
    right_pct = np.sum(right_nans_per_frame) / (total_frames * total_floats_per_frame) * 100

    print(f"Total frames:                     {total_frames}")
    print(f"Left hand fully missing frames:   {left_fully_missing} / {total_frames}")
    print(f"Right hand fully missing frames:  {right_fully_missing} / {total_frames}")
    print(f"Left hand overall missing joints: {left_pct:.1f}%")
    print(f"Right hand overall missing joints:{right_pct:.1f}%")
    print()

    if left_fully_missing == total_frames:
        print("CRITICAL: 100% of left hand tracking data is missing (NaN).")
    if right_fully_missing == total_frames:
        print("CRITICAL: 100% of right hand tracking data is missing (NaN).")
    print("=" * 50)


if __name__ == "__main__":
    main()

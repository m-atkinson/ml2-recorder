#!/usr/bin/env python3
"""
Test suite for validate_session.py

Creates synthetic session directories (valid and invalid) in a temp directory
and runs validate_session.py against them to verify correctness.

Run with:
    python -m pytest tools/test_validate_session.py -v
    python tools/test_validate_session.py
"""

import csv
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

VALIDATE_SCRIPT = Path(__file__).resolve().parent / "validate_session.py"


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def make_minimal_jpeg(path: Path) -> None:
    """Write a minimal valid JPEG file (valid SOI + APP0 marker + EOI)."""
    # SOI (FF D8) + APP0 marker (FF E0) with minimal JFIF header + EOI (FF D9)
    soi = b"\xff\xd8"
    # APP0 marker: FF E0, length 0x0010 (16 bytes), JFIF\0, version 1.1, etc.
    app0 = (
        b"\xff\xe0"
        b"\x00\x10"          # length = 16
        b"JFIF\x00"          # identifier
        b"\x01\x01"          # version 1.1
        b"\x00"              # aspect ratio units
        b"\x00\x01\x00\x01"  # x/y density
        b"\x00\x00"          # thumbnail size
    )
    eoi = b"\xff\xd9"
    # Add some padding to reach ~100 bytes
    padding = b"\x00" * (100 - len(soi) - len(app0) - len(eoi))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(soi + app0 + padding + eoi)


def make_minimal_png(path: Path, width: int = 16, height: int = 16, fill: int = 128) -> None:
    """Write a valid PNG file. Use width/height > 1 and varied fill to pass
    depth variance checks (which flag files < 200 bytes as empty)."""
    import zlib

    signature = b"\x89PNG\r\n\x1a\n"

    # IHDR: 16-bit grayscale (color type 0, bit depth 16) for depth-like images
    ihdr_data = struct.pack(">IIBBBBB", width, height, 16, 0, 0, 0, 0)
    ihdr_crc = zlib.crc32(b"IHDR" + ihdr_data) & 0xFFFFFFFF
    ihdr_chunk = struct.pack(">I", 13) + b"IHDR" + ihdr_data + struct.pack(">I", ihdr_crc)

    # Build raw scanlines: filter byte 0 + 2 bytes per pixel (16-bit)
    raw = b""
    for row in range(height):
        raw += b"\x00"  # filter byte
        for col in range(width):
            val = (fill + row * 17 + col * 13) & 0xFFFF  # varied pixel values
            raw += struct.pack(">H", val)

    compressed = zlib.compress(raw)
    idat_crc = zlib.crc32(b"IDAT" + compressed) & 0xFFFFFFFF
    idat_chunk = struct.pack(">I", len(compressed)) + b"IDAT" + compressed + struct.pack(">I", idat_crc)

    # IEND
    iend_crc = zlib.crc32(b"IEND") & 0xFFFFFFFF
    iend_chunk = struct.pack(">I", 0) + b"IEND" + struct.pack(">I", iend_crc)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(signature + ihdr_chunk + idat_chunk + iend_chunk)


def make_minimal_wav(path: Path) -> None:
    """Write a minimal valid WAV/RIFF file header (no actual audio data)."""
    # RIFF header
    sample_rate = 48000
    num_channels = 4
    bits_per_sample = 16
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    data_size = 0  # empty audio

    fmt_chunk = struct.pack(
        "<4sIHHIIHH",
        b"fmt ",
        16,              # chunk size
        1,               # PCM format
        num_channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
    )
    data_chunk = struct.pack("<4sI", b"data", data_size)
    riff_size = 4 + len(fmt_chunk) + len(data_chunk)  # 'WAVE' + chunks
    riff_header = struct.pack("<4sI4s", b"RIFF", riff_size, b"WAVE")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(riff_header + fmt_chunk + data_chunk)


def write_csv(path: Path, headers: list[str], rows: list[list]) -> None:
    """Write a CSV file with the given headers and data rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for row in rows:
            writer.writerow(row)


def run_validate(session_dir: str) -> subprocess.CompletedProcess:
    """Run validate_session.py against a session directory and return the result."""
    return subprocess.run(
        [sys.executable, str(VALIDATE_SCRIPT), session_dir],
        capture_output=True,
        text=True,
    )


# ---------------------------------------------------------------------------
# Reusable session builders
# ---------------------------------------------------------------------------

def _monotonic_timestamps(start: int, count: int, interval_ns: int) -> list[int]:
    """Generate monotonically increasing timestamps."""
    return [start + i * interval_ns for i in range(count)]


def _build_rgb(session: Path, num_frames: int = 5, fps: int = 15) -> None:
    """Create rgb/ directory, images, and rgb.csv."""
    interval_ns = int(1e9 / fps)  # ~66_666_666 ns for 15 fps
    timestamps = _monotonic_timestamps(1_000_000_000, num_frames, interval_ns)
    rgb_dir = session / "rgb"
    rgb_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, ts in enumerate(timestamps):
        fname = f"frame_{i:06d}.jpg"
        make_minimal_jpeg(rgb_dir / fname)
        rows.append([ts, ts, fname])

    write_csv(session / "rgb.csv", ["timestamp_ns", "sensor_timestamp_ns", "filename"], rows)


def _build_rgb_h264(session: Path, num_frames: int = 5, fps: int = 15) -> None:
    """Create rgb.h264 and an H.264-style rgb.csv manifest."""
    interval_ns = int(1e9 / fps)
    timestamps = _monotonic_timestamps(1_000_000_000, num_frames, interval_ns)
    rows = [[ts, ts, i] for i, ts in enumerate(timestamps)]

    (session / "rgb.h264").write_bytes(b"\x00\x00\x00\x01\x67\x64\x00\x1f")
    write_csv(session / "rgb.csv", ["timestamp_ns", "sensor_timestamp_ns", "frame_index"], rows)


def _build_depth(session: Path, num_frames: int = 3, fps: int = 15) -> None:
    """Create depth/ + depth_confidence/ directories, images, and depth.csv."""
    interval_ns = int(1e9 / fps)
    timestamps = _monotonic_timestamps(1_000_000_000, num_frames, interval_ns)

    depth_dir = session / "depth"
    depth_conf_dir = session / "depth_confidence"
    depth_dir.mkdir(parents=True, exist_ok=True)
    depth_conf_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, ts in enumerate(timestamps):
        fname = f"frame_{i:06d}.png"
        conf_fname = f"frame_{i:06d}.png"
        make_minimal_png(depth_dir / fname)
        make_minimal_png(depth_conf_dir / conf_fname)
        rows.append([ts, ts, fname, conf_fname])

    write_csv(session / "depth.csv", ["timestamp_ns", "sensor_timestamp_ns", "filename", "confidence_filename"], rows)


def _build_world_cams(session: Path, num_frames: int = 3) -> None:
    """Create world_cam_0/1/2 directories with CSVs."""
    interval_ns = int(1e9 / 15)
    timestamps = _monotonic_timestamps(1_000_000_000, num_frames, interval_ns)

    for cam_idx in range(3):
        cam_dir = session / f"world_cam_{cam_idx}"
        cam_dir.mkdir(parents=True, exist_ok=True)
        rows = [[ts, ts, f"frame_{i:06d}.png"] for i, ts in enumerate(timestamps)]
        # Also create the image files
        for i, ts in enumerate(timestamps):
            make_minimal_png(cam_dir / f"frame_{i:06d}.png")
        write_csv(
            session / f"world_cam_{cam_idx}.csv",
            ["timestamp_ns", "sensor_timestamp_ns", "filename"],
            rows,
        )


def _build_head_pose(session: Path, num_rows: int = 3) -> None:
    """Create head_pose.csv with valid position/quaternion data."""
    interval_ns = int(1e9 / 15)
    timestamps = _monotonic_timestamps(1_000_000_000, num_rows, interval_ns)
    headers = [
        "timestamp_ns", "sensor_timestamp_ns", "pos_x", "pos_y", "pos_z",
        "quat_w", "quat_x", "quat_y", "quat_z",
    ]
    rows = []
    for i, ts in enumerate(timestamps):
        rows.append([
            ts, ts,
            0.1 * (i + 1), 0.2 * (i + 1), 0.3 * (i + 1),  # position
            1.0, 0.0, 0.0, 0.0,                              # quaternion (identity)
        ])
    write_csv(session / "head_pose.csv", headers, rows)


def _build_eye_tracking(session: Path, num_rows: int = 3) -> None:
    """Create eye_tracking.csv."""
    interval_ns = int(1e9 / 15)
    timestamps = _monotonic_timestamps(1_000_000_000, num_rows, interval_ns)
    headers = [
        "timestamp_ns", "sensor_timestamp_ns",
        "left_origin_x", "left_origin_y", "left_origin_z",
        "left_dir_x", "left_dir_y", "left_dir_z",
        "right_origin_x", "right_origin_y", "right_origin_z",
        "right_dir_x", "right_dir_y", "right_dir_z",
        "fixation_x", "fixation_y", "fixation_z",
    ]
    rows = []
    for i, ts in enumerate(timestamps):
        rows.append([
            ts, ts,
            0.0, 0.0, 0.0,           # left origin
            0.1 * i, 0.2 * i, 0.5,   # left direction
            0.0, 0.0, 0.0,           # right origin
            0.1 * i, 0.2 * i, 0.5,   # right direction
            0.5, 0.3, 1.0,           # fixation point
        ])
    write_csv(session / "eye_tracking.csv", headers, rows)


def _build_hand_tracking(session: Path, num_rows: int = 3) -> None:
    """Create hand_tracking.csv with expanded keypoint columns."""
    interval_ns = int(1e9 / 15)
    timestamps = _monotonic_timestamps(1_000_000_000, num_rows, interval_ns)
    # 28 keypoint names matching the ML2 hand tracking API
    kp_names = [
        "thumb_tip", "thumb_ip", "thumb_mcp", "thumb_cmc",
        "index_tip", "index_dip", "index_pip", "index_mcp",
        "middle_tip", "middle_dip", "middle_pip", "middle_mcp",
        "ring_tip", "ring_dip", "ring_pip", "ring_mcp",
        "pinky_tip", "pinky_dip", "pinky_pip", "pinky_mcp",
        "wrist_center", "wrist_ulnar", "wrist_radial", "hand_center",
        "index_meta", "middle_meta", "ring_meta", "pinky_meta",
    ]
    headers = ["timestamp_ns", "sensor_timestamp_ns"]
    for side in ["left", "right"]:
        for kp in kp_names:
            headers.extend([f"{side}_{kp}_x", f"{side}_{kp}_y", f"{side}_{kp}_z"])
    headers.extend(["left_confidence", "right_confidence"])

    rows = []
    for i, ts in enumerate(timestamps):
        row = [ts, ts]
        # Generate 28 * 2 * 3 = 168 joint values with some variance
        for side_offset in [0.1, 0.2]:
            for j in range(28):
                for axis in range(3):
                    row.append(round(side_offset * (i + 1) + j * 0.01 + axis * 0.001, 6))
        row.extend([0.95, 0.90])
        rows.append(row)
    write_csv(session / "hand_tracking.csv", headers, rows)


def _build_imu(session: Path, num_rows: int = 15, all_zeros: bool = False) -> None:
    """Create imu.csv with accelerometer/gyroscope data."""
    interval_ns = int(1e9 / 200)  # 200 Hz IMU
    timestamps = _monotonic_timestamps(1_000_000_000, num_rows, interval_ns)
    headers = [
        "timestamp_ns", "sensor_timestamp_ns", "unit_id",
        "accel_x", "accel_y", "accel_z",
        "gyro_x", "gyro_y", "gyro_z",
    ]
    rows = []
    for i, ts in enumerate(timestamps):
        if all_zeros:
            rows.append([ts, ts, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        else:
            rows.append([
                ts, ts, i % 4,  # unit_id cycles 0-3
                0.01 * i, -9.81 + 0.005 * i, 0.02 * i,  # accel (gravity on y)
                0.001 * i, -0.002 * i, 0.003 * i,          # gyro
            ])
    write_csv(session / "imu.csv", headers, rows)


def _build_audio(session: Path) -> None:
    """Create audio/raw_4ch.wav."""
    make_minimal_wav(session / "audio" / "raw_4ch.wav")


def _build_metadata(session: Path, enabled_streams=None) -> None:
    """Write metadata.json."""
    if enabled_streams is None:
        enabled_streams = ["rgb"]
    meta = {
        "enabled_streams": enabled_streams,
        "capture_profile": "full",
        "device": "ML2",
        "rgb_fps": 15,
    }
    (session / "metadata.json").write_text(json.dumps(meta, indent=2))


def _build_calibration(session: Path) -> None:
    """Write calibration.json."""
    cal = {
        "rgb_intrinsics": {
            "fx": 500, "fy": 500, "cx": 704, "cy": 704,
        }
    }
    (session / "calibration.json").write_text(json.dumps(cal, indent=2))


def _build_minimal_rgb_session(session: Path) -> None:
    """Build a minimal valid session with only RGB."""
    _build_rgb(session, num_frames=5)
    _build_metadata(session, enabled_streams=["rgb"])
    _build_calibration(session)


def _build_minimal_h264_session(session: Path) -> None:
    """Build a minimal valid session with RGB stored as raw H.264."""
    _build_rgb_h264(session, num_frames=5)
    _build_metadata(session, enabled_streams=["rgb"])
    _build_calibration(session)


def _build_full_session(session: Path) -> None:
    """Build a valid full session with all streams."""
    all_streams = [
        "rgb", "depth", "depth_confidence",
        "world_cam_0", "world_cam_1", "world_cam_2",
        "head_pose", "eye_tracking", "hand_tracking",
        "imu", "audio",
    ]
    _build_rgb(session, num_frames=3)
    _build_depth(session, num_frames=3)
    _build_world_cams(session, num_frames=3)
    _build_head_pose(session, num_rows=3)
    _build_eye_tracking(session, num_rows=3)
    _build_hand_tracking(session, num_rows=3)
    _build_imu(session, num_rows=15)
    _build_audio(session)
    _build_metadata(session, enabled_streams=all_streams)
    _build_calibration(session)


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

class TestValidateSession(unittest.TestCase):
    """Tests for validate_session.py using synthetic session directories."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="test_validate_session_")
        self.session = Path(self.tmpdir) / "session"
        self.session.mkdir()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    # -- 1. Valid minimal session (RGB only) --------------------------------

    def test_valid_minimal_session(self):
        """A minimal session with only RGB + metadata should pass validation."""
        _build_minimal_rgb_session(self.session)

        result = run_validate(str(self.session))
        self.assertEqual(
            result.returncode, 0,
            f"Expected PASS (exit 0) for valid minimal session.\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}",
        )

    # -- 2. Valid minimal H.264 session -------------------------------------

    def test_valid_minimal_h264_session(self):
        """A minimal session with raw H.264 RGB should pass validation."""
        _build_minimal_h264_session(self.session)

        result = run_validate(str(self.session))
        self.assertEqual(
            result.returncode, 0,
            f"Expected PASS (exit 0) for valid H.264 session.\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}",
        )

    # -- 3. Valid full session (all streams) --------------------------------

    def test_valid_full_session(self):
        """A full session with all streams present and correct should pass."""
        _build_full_session(self.session)

        result = run_validate(str(self.session))
        self.assertEqual(
            result.returncode, 0,
            f"Expected PASS (exit 0) for valid full session.\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}",
        )

    # -- 4. Corrupt JPEG ----------------------------------------------------

    def test_corrupt_jpeg(self):
        """A session with a corrupt JPEG (bad magic bytes) should fail."""
        _build_minimal_rgb_session(self.session)

        # Overwrite the first JPEG with random bytes (no valid JPEG header)
        corrupt_file = self.session / "rgb" / "frame_000000.jpg"
        corrupt_file.write_bytes(b"\x00\x01\x02\x03" * 25)  # 100 bytes of junk

        result = run_validate(str(self.session))
        self.assertNotEqual(
            result.returncode, 0,
            f"Expected FAIL (non-zero exit) for corrupt JPEG.\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}",
        )

    # -- 5. Non-monotonic timestamps ----------------------------------------

    def test_non_monotonic_timestamps(self):
        """Timestamps that go backwards in rgb.csv should fail validation."""
        _build_minimal_rgb_session(self.session)

        # Rewrite rgb.csv with a backwards timestamp
        interval_ns = int(1e9 / 15)
        ts = _monotonic_timestamps(1_000_000_000, 5, interval_ns)
        # Swap timestamps 2 and 3 so order breaks
        ts[2], ts[3] = ts[3], ts[2]

        rows = [[t, t, f"frame_{i:06d}.jpg"] for i, t in enumerate(ts)]
        write_csv(self.session / "rgb.csv", ["timestamp_ns", "sensor_timestamp_ns", "filename"], rows)

        result = run_validate(str(self.session))
        self.assertNotEqual(
            result.returncode, 0,
            f"Expected FAIL for non-monotonic timestamps.\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}",
        )

    # -- 6. Large frame gap -------------------------------------------------

    def test_large_frame_gap(self):
        """A 500ms gap at 15fps (expected ~66ms, threshold ~200ms) should warn/fail."""
        _build_minimal_rgb_session(self.session)

        # Rewrite rgb.csv with a 500ms gap between frame 2 and 3
        interval_ns = int(1e9 / 15)
        ts = _monotonic_timestamps(1_000_000_000, 5, interval_ns)
        # Insert a 500ms gap after frame 2
        gap_ns = 500_000_000
        ts[3] = ts[2] + gap_ns
        ts[4] = ts[3] + interval_ns

        rows = [[t, t, f"frame_{i:06d}.jpg"] for i, t in enumerate(ts)]
        write_csv(self.session / "rgb.csv", ["timestamp_ns", "sensor_timestamp_ns", "filename"], rows)

        result = run_validate(str(self.session))
        # Accept either non-zero exit (FAIL) or a warning in output
        failed_or_warned = (
            result.returncode != 0
            or "warn" in result.stdout.lower()
            or "gap" in result.stdout.lower()
            or "jitter" in result.stdout.lower()
        )
        self.assertTrue(
            failed_or_warned,
            f"Expected WARN or FAIL for large frame gap.\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}",
        )

    # -- 7. All-zero IMU data -----------------------------------------------

    def test_all_zero_imu(self):
        """IMU data with all zeros should fail the data variance check."""
        _build_full_session(self.session)

        # Overwrite imu.csv with all-zero sensor values
        _build_imu(self.session, num_rows=15, all_zeros=True)

        result = run_validate(str(self.session))
        self.assertNotEqual(
            result.returncode, 0,
            f"Expected FAIL for all-zero IMU data.\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}",
        )

    # -- 8. Missing CSV for existing directory ------------------------------

    def test_missing_csv_for_existing_dir(self):
        """rgb/ directory exists with JPEGs but rgb.csv is missing -> FAIL."""
        _build_minimal_rgb_session(self.session)

        # Remove rgb.csv
        (self.session / "rgb.csv").unlink()

        result = run_validate(str(self.session))
        self.assertNotEqual(
            result.returncode, 0,
            f"Expected FAIL when rgb.csv is missing.\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}",
        )

    # -- 9. Wrong CSV schema ------------------------------------------------

    def test_wrong_csv_schema(self):
        """rgb.csv with wrong column headers should fail schema validation."""
        _build_minimal_rgb_session(self.session)

        # Overwrite rgb.csv with wrong headers
        interval_ns = int(1e9 / 15)
        ts = _monotonic_timestamps(1_000_000_000, 5, interval_ns)
        rows = [[t, t, f"frame_{i:06d}.jpg"] for i, t in enumerate(ts)]
        write_csv(
            self.session / "rgb.csv",
            ["time", "sensor_time", "file"],  # wrong headers
            rows,
        )

        result = run_validate(str(self.session))
        self.assertNotEqual(
            result.returncode, 0,
            f"Expected FAIL for wrong CSV schema.\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}",
        )


if __name__ == "__main__":
    unittest.main()

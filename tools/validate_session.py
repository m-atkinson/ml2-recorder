#!/usr/bin/env python3
"""
validate_session.py — Acceptance gate for ML2 recorder VRS sessions.

Validates a pulled .vrs file and gives a pass/fail verdict without
visual inspection.  Works without any third-party packages by scanning
the VRS binary for known markers.  If pyvrs is available it performs
deeper record-level checks.

Usage:
    python validate_session.py /path/to/session_YYYYMMDD_HHMMSS.vrs
    python validate_session.py --strict /path/to/session.vrs
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Result tracking
# ---------------------------------------------------------------------------

PASS = "PASS"
FAIL = "FAIL"
WARN = "WARN"

_results: List[Tuple[str, str, str]] = []


def record(status: str, name: str, detail: str = "") -> None:
    _results.append((status, name, detail))
    colour = {PASS: "\033[32m", FAIL: "\033[31m", WARN: "\033[33m"}.get(status, "")
    msg = f"  [{colour}{status}\033[0m] {name}"
    if detail:
        msg += f" -- {detail}"
    print(msg)


def print_summary(strict: bool) -> int:
    counts = {PASS: 0, FAIL: 0, WARN: 0}
    for status, _, _ in _results:
        counts[status] = counts.get(status, 0) + 1

    effective_fail = counts[FAIL] + (counts[WARN] if strict else 0)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  PASS: {counts[PASS]}")
    print(f"  FAIL: {counts[FAIL]}")
    print(f"  WARN: {counts[WARN]}")
    if strict and counts[WARN]:
        print(f"  (--strict: {counts[WARN]} warnings treated as failures)")

    if effective_fail == 0:
        print("\n  *** SESSION VALID ***")
    else:
        print(f"\n  *** SESSION INVALID ({effective_fail} issue(s)) ***")
    print("=" * 60)

    return 0 if effective_fail == 0 else 1


# ---------------------------------------------------------------------------
# Stream definitions
# ---------------------------------------------------------------------------

REQUIRED_STREAMS = ["ml2/rgb", "ml2/imu", "ml2/head_pose"]

OPTIONAL_STREAMS = [
    "ml2/depth",
    "ml2/eye_tracking",
    "ml2/hand_tracking",
    "ml2/world_cam_0",
    "ml2/world_cam_1",
    "ml2/world_cam_2",
    "ml2/audio",
    "ml2/mesh",
]

REQUIRED_TAGS = ["ml2.session_name", "ml2.start_time_ns"]

# Minimum file size for a 10-second full-quality session (bytes)
MIN_SIZE_BYTES = 1 * 1024 * 1024  # 1 MB

# VRS file magic — the file header begins with this string
VRS_MAGIC = b"VisionRe"


# ---------------------------------------------------------------------------
# Pure-Python binary checks (no dependencies)
# ---------------------------------------------------------------------------

def check_file(path: Path) -> Optional[bytes]:
    """Read the whole file; return bytes or None on error."""
    if not path.exists():
        record(FAIL, "file_exists", str(path))
        return None
    record(PASS, "file_exists", str(path))

    size = path.stat().st_size
    size_mb = size / (1024 * 1024)
    if size < MIN_SIZE_BYTES:
        record(FAIL, "file_size", f"{size_mb:.3f} MB — suspiciously small")
    else:
        record(PASS, "file_size", f"{size_mb:.1f} MB")

    try:
        data = path.read_bytes()
    except OSError as e:
        record(FAIL, "file_readable", str(e))
        return None
    record(PASS, "file_readable")
    return data


def check_magic(data: bytes) -> bool:
    if data[:len(VRS_MAGIC)] == VRS_MAGIC:
        record(PASS, "vrs_magic", "VRS file header detected")
        return True
    else:
        record(FAIL, "vrs_magic",
               f"expected {VRS_MAGIC!r}, got {data[:8]!r}")
        return False


def check_stream_flavors(data: bytes) -> Dict[str, bool]:
    """Scan raw bytes for each expected stream flavor string."""
    found: Dict[str, bool] = {}
    all_streams = REQUIRED_STREAMS + OPTIONAL_STREAMS
    for flavor in all_streams:
        found[flavor] = flavor.encode() in data

    for fl in REQUIRED_STREAMS:
        if found[fl]:
            record(PASS, f"stream:{fl}")
        else:
            record(FAIL, f"stream:{fl}", "flavor string not found in file")

    for fl in OPTIONAL_STREAMS:
        if found[fl]:
            record(PASS, f"stream:{fl}")
        else:
            record(WARN, f"stream:{fl}", "not present")

    return found


def check_file_tags(data: bytes) -> None:
    """Scan for required file-level tag key strings."""
    for tag in REQUIRED_TAGS:
        tag_bytes = tag.encode()
        pos = data.find(tag_bytes)
        if pos >= 0:
            # Extract the value: skip the key and any length/separator bytes,
            # then read until the next null or newline (VRS stores as C-strings).
            after = data[pos + len(tag_bytes):]
            # Find the next printable run of ASCII after the key
            val_start = 0
            for i, b in enumerate(after[:32]):
                if 0x20 <= b <= 0x7E:
                    val_start = i
                    break
            val_bytes = after[val_start:]
            val_end = 0
            for i, b in enumerate(val_bytes[:80]):
                if b < 0x20 or b > 0x7E:
                    val_end = i
                    break
            else:
                val_end = 80
            val = val_bytes[:val_end].decode("ascii", errors="replace").strip()
            record(PASS, f"tag:{tag}", val[:60] if val else "(present)")
        else:
            record(WARN, f"tag:{tag}", "not found in file")


def check_not_truncated(data: bytes) -> None:
    """Heuristic: last 64 bytes should be readable and non-zero."""
    tail = data[-64:]
    if len(tail) < 64:
        record(WARN, "not_truncated", "file too small to check tail")
        return
    if all(b == 0 for b in tail):
        record(FAIL, "not_truncated", "last 64 bytes are all zeros — possible truncation")
    else:
        record(PASS, "not_truncated", "file tail looks intact")


# ---------------------------------------------------------------------------
# Deep checks via pyvrs (optional)
# ---------------------------------------------------------------------------

def deep_check_pyvrs(path: Path) -> None:
    """If pyvrs is importable, run deeper record-level checks."""
    try:
        import pyvrs  # type: ignore
    except ImportError:
        record(WARN, "pyvrs_deep_check",
               "pyvrs not installed — skipping record-level checks (pip install vrs)")
        return

    try:
        reader = pyvrs.SyncVRSReader(str(path))
    except Exception as e:
        record(FAIL, "pyvrs_open", str(e))
        return
    record(PASS, "pyvrs_open")

    # Build flavor → stream_id map (matches convert_session_to_hdf5.py)
    flavor_map: Dict[str, list] = {}
    for sid in reader.stream_ids:
        try:
            fl = reader.get_stream_info(sid).get("flavor", "")
        except Exception:
            fl = ""
        if fl:
            flavor_map.setdefault(fl, []).append(sid)

    # Count DATA records per required stream
    min_records = {"ml2/rgb": 10, "ml2/imu": 500, "ml2/head_pose": 10}
    for fl, min_n in min_records.items():
        if fl not in flavor_map:
            continue
        n = 0
        for sid in flavor_map[fl]:
            try:
                n += sum(1 for _ in reader.filtered_by_fields(
                    stream_ids={sid}, record_types={"data"}))
            except Exception:
                pass
        if n >= min_n:
            record(PASS, f"pyvrs:record_count:{fl.split('/')[-1]}", f"{n} records")
        elif n > 0:
            record(WARN, f"pyvrs:record_count:{fl.split('/')[-1]}",
                   f"{n} records (expected ≥ {min_n})")
        else:
            record(FAIL, f"pyvrs:record_count:{fl.split('/')[-1]}",
                   f"0 records (expected ≥ {min_n})")

    # ------------------------------------------------------------------
    # Extrinsics (per-frame camera_pose) health checks
    # ------------------------------------------------------------------
    _check_extrinsics(reader, flavor_map)


def _layout_field(rec, field: str, default=None):
    try:
        if rec.n_metadata_blocks > 0:
            return rec.metadata_blocks[0][field]
    except Exception:
        pass
    return default


def _check_extrinsics(reader, flavor_map: Dict[str, list]) -> None:
    """Validate per-frame camera_pose population and quaternion sanity for
    every image stream (RGB, depth, world cams)."""
    import math

    image_streams = [
        ("ml2/rgb",          "rgb",          0.60),  # allow startup misses for RGB
        ("ml2/depth",        "depth",        0.90),
        ("ml2/world_cam_0",  "world_cam_0",  0.90),
        ("ml2/world_cam_1",  "world_cam_1",  0.90),
        ("ml2/world_cam_2",  "world_cam_2",  0.90),
    ]

    for flavor, label, valid_threshold in image_streams:
        if flavor not in flavor_map:
            continue
        n_total = 0
        n_data  = 0  # records that carry a data frame (exclude RGB config NALs)
        n_valid = 0
        worst_q_err = 0.0  # largest deviation from unit norm
        worst_p_mag = 0.0  # largest position magnitude
        schema_missing = False
        for sid in flavor_map[flavor]:
            try:
                it = reader.filtered_by_fields(stream_ids={sid},
                                                record_types={"data"})
            except Exception:
                continue
            for r in it:
                n_total += 1
                # For RGB: skip config NALs (SPS/PPS) — they don't have a pose.
                if flavor == "ml2/rgb":
                    is_cfg = _layout_field(r, "is_config_nal")
                    if is_cfg is not None and int(is_cfg) == 1:
                        continue
                n_data += 1
                v = _layout_field(r, "camera_pose_valid")
                if v is None:
                    schema_missing = True
                    continue
                if int(v) == 0:
                    continue
                n_valid += 1
                q = _layout_field(r, "camera_pose_orientation")
                p = _layout_field(r, "camera_pose_position")
                if q is None or p is None:
                    continue
                # Quaternion unit-norm check (SDK-native x,y,z,w).
                qn = math.sqrt(sum(float(x)*float(x) for x in q))
                q_err = abs(qn - 1.0)
                if q_err > worst_q_err:
                    worst_q_err = q_err
                pm = math.sqrt(sum(float(x)*float(x) for x in p))
                if pm > worst_p_mag:
                    worst_p_mag = pm

        if n_data == 0:
            continue
        if schema_missing and n_valid == 0:
            record(WARN, f"pyvrs:extrinsics:{label}",
                   "camera_pose fields not in schema (old recorder build?)")
            continue

        valid_rate = n_valid / n_data if n_data else 0.0
        detail = (f"{n_valid}/{n_data} frames valid "
                  f"({valid_rate*100:.1f}%), "
                  f"max |q|-1 = {worst_q_err:.4f}, "
                  f"max |t| = {worst_p_mag:.2f} m")

        if valid_rate >= valid_threshold:
            status = PASS
        elif valid_rate > 0:
            status = WARN
        else:
            status = FAIL
        record(status, f"pyvrs:extrinsics:{label}:valid_rate", detail)

        # Quaternion norm sanity: each quaternion within 1% of unit.
        if worst_q_err > 0.01:
            record(FAIL, f"pyvrs:extrinsics:{label}:quat_norm",
                   f"max |q|-1 = {worst_q_err:.4f} (expect < 0.01)")
        else:
            record(PASS, f"pyvrs:extrinsics:{label}:quat_norm",
                   f"max |q|-1 = {worst_q_err:.4f}")

        # Position sanity: worst magnitude < 10 m in a sane session.
        if worst_p_mag > 10.0:
            record(WARN, f"pyvrs:extrinsics:{label}:position_range",
                   f"max |t| = {worst_p_mag:.2f} m (unusual — user moved far?)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def validate(vrs_path: Path, strict: bool) -> int:
    print(f"\nValidating: {vrs_path}\n")

    data = check_file(vrs_path)
    if data is None:
        return print_summary(strict)

    print("\n--- VRS header ---")
    check_magic(data)
    check_not_truncated(data)

    print("\n--- File tags ---")
    check_file_tags(data)

    print("\n--- Streams ---")
    check_stream_flavors(data)

    print("\n--- Deep checks (pyvrs) ---")
    deep_check_pyvrs(vrs_path)

    return print_summary(strict)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate an ML2 recorder VRS session file")
    parser.add_argument("vrs_path", type=Path,
                        help="Path to session .vrs file")
    parser.add_argument("--strict", action="store_true",
                        help="Treat warnings as failures")
    args = parser.parse_args()

    path = args.vrs_path.resolve()
    if path.is_dir() or not path.suffix:
        candidate = path.with_suffix(".vrs")
        if candidate.exists():
            path = candidate
        else:
            print(f"ERROR: Expected a .vrs file, got: {path}")
            print(f"  Hint: file should be at {path}.vrs")
            sys.exit(1)

    sys.exit(validate(path, strict=args.strict))


if __name__ == "__main__":
    main()

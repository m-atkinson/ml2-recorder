#!/usr/bin/env python3
"""
quality_check.py — Robot-training data quality evaluator for ML2 sessions.

Reads an HDF5 session file (produced by convert_session_to_hdf5.py) and
scores each sensor stream against thresholds designed for manipulation-learning
pipelines (EgoMimic / ACT / Diffusion Policy).

Outputs a human-readable report with PASS / WARN / FAIL per check and an
overall 0–100 suitability score.  Returns exit code 0 if score >= 60,
1 if below (suitable for CI gates).

Usage:
    python quality_check.py session.h5
    python quality_check.py session.h5 --min-score 70
    python quality_check.py session.h5 --json report.json
    python quality_check.py session.vrs          # auto-converts first

Requires: numpy, h5py  (pip install numpy h5py)
Optional: Pillow, opencv-python-headless  (for image blur checks)
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np

# ── Optional image deps ───────────────────────────────────────────────────────
try:
    import cv2 as _cv2  # type: ignore
    _HAVE_CV2 = True
except ImportError:
    _HAVE_CV2 = False

try:
    from PIL import Image as _PILImage  # type: ignore
    import io as _io
    _HAVE_PIL = True
except ImportError:
    _HAVE_PIL = False

# =============================================================================
# Thresholds for robot-training use (manipulation / ego-centric tasks)
# =============================================================================

# RGB — primary ego-centric view
RGB_FPS_TARGET   = 30.0   # Hz expected from hardware
RGB_FPS_WARN     = 20.0   # WARN if below
RGB_FPS_FAIL     = 10.0   # FAIL if below

RGB_JITTER_WARN  = 0.20   # CV (std/mean) of frame intervals: warn if > 20 %
RGB_JITTER_FAIL  = 0.40   # fail if > 40 %

RGB_MAX_GAP_WARN = 0.100  # seconds — single-frame dropout threshold
RGB_MAX_GAP_FAIL = 0.500  # seconds

RGB_BLUR_WARN    = 100.0  # Laplacian variance — warn if below (blurry)
RGB_BLUR_FAIL    = 30.0   # fail if below

# IMU
IMU_HZ_WARN      = 200.0  # Hz
IMU_HZ_FAIL      = 100.0  # Hz
IMU_GAP_WARN     = 0.020  # seconds
IMU_GAP_FAIL     = 0.100  # seconds
IMU_GRAVITY      = 9.81   # m/s²  expected accel norm at rest
IMU_GRAVITY_TOL  = 0.5    # |norm - 9.81| > this → warn (possible calibration issue)

# Head pose (bounding-box volume = product of X/Y/Z ranges)
POSE_WORKSPACE_WARN = 0.005  # m³ — very little movement
POSE_WORKSPACE_FAIL = 0.001  # m³ — almost no movement

# Hand tracking (critical for manipulation)
HAND_DETECT_WARN = 0.70   # fraction of frames
HAND_DETECT_FAIL = 0.40

# Eye tracking
EYE_DETECT_WARN  = 0.50
EYE_DETECT_FAIL  = 0.20

# Session duration
DURATION_WARN    = 20.0   # seconds
DURATION_FAIL    = 10.0   # seconds

# World cameras blur (JPEG — can actually decode)
WC_BLUR_WARN     = 80.0
WC_BLUR_FAIL     = 20.0

# =============================================================================
# Result tracking
# =============================================================================

PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"
INFO = "INFO"

@dataclass
class Check:
    status: str
    name: str
    detail: str = ""
    value: Optional[float] = None    # numeric value for JSON export
    weight: float = 1.0              # weight in score

@dataclass
class Dimension:
    name: str
    checks: List[Check] = field(default_factory=list)
    score: float = 100.0             # 0–100

    def add(self, status: str, name: str, detail: str = "",
            value: Optional[float] = None, weight: float = 1.0) -> Check:
        c = Check(status, name, detail, value, weight)
        self.checks.append(c)
        return c

    def compute_score(self) -> float:
        """Weighted average: PASS=100, WARN=50, FAIL=0."""
        if not self.checks:
            return 100.0
        total_w = sum(c.weight for c in self.checks if c.status != INFO)
        if total_w == 0:
            return 100.0
        weighted = sum(
            (100 if c.status == PASS else 50 if c.status == WARN else 0) * c.weight
            for c in self.checks if c.status != INFO
        )
        self.score = weighted / total_w
        return self.score


def _colour(status: str) -> str:
    return {PASS: "\033[32m", WARN: "\033[33m", FAIL: "\033[31m", INFO: "\033[36m"}.get(status, "")

RESET = "\033[0m"
BOLD  = "\033[1m"


def _print_check(c: Check) -> None:
    col = _colour(c.status)
    msg = f"    [{col}{c.status}{RESET}] {c.name}"
    if c.detail:
        msg += f" — {c.detail}"
    print(msg)


def _pf(val: float, warn_thresh: float, fail_thresh: float,
        low_is_bad: bool = True) -> str:
    """Return PASS/WARN/FAIL given value and thresholds."""
    if low_is_bad:
        if val >= warn_thresh:
            return PASS
        if val >= fail_thresh:
            return WARN
        return FAIL
    else:
        if val <= warn_thresh:
            return PASS
        if val <= fail_thresh:
            return WARN
        return FAIL


# =============================================================================
# Helpers
# =============================================================================

def _laplacian_var_from_jpeg(data: bytes) -> Optional[float]:
    """Decode a JPEG byte blob and return Laplacian variance (blur metric)."""
    if _HAVE_CV2:
        arr = np.frombuffer(data, dtype=np.uint8)
        img = _cv2.imdecode(arr, _cv2.IMREAD_GRAYSCALE)
        if img is not None:
            return float(_cv2.Laplacian(img, _cv2.CV_64F).var())
    if _HAVE_PIL:
        try:
            img = _PILImage.open(_io.BytesIO(data)).convert("L")
            g = np.array(img, dtype=np.float32)
            pad = np.pad(g, 1, mode="edge")
            lap = (-pad[:-2,:-2]-pad[:-2,1:-1]-pad[:-2,2:]-pad[1:-1,:-2]
                   +8*pad[1:-1,1:-1]-pad[1:-1,2:]-pad[2:,:-2]-pad[2:,1:-1]-pad[2:,2:])
            return float(lap.var())
        except Exception:
            pass
    return None


def _is_jpeg(data: bytes) -> bool:
    return len(data) >= 2 and data[0] == 0xFF and data[1] == 0xD8


def _sample_blur(images_ds, n_samples: int = 10) -> Tuple[Optional[float], int]:
    """Return (mean blur score, n_decoded) for a sample of frames."""
    n = len(images_ds)
    if n == 0:
        return None, 0
    indices = np.linspace(0, n - 1, min(n_samples, n), dtype=int)
    scores = []
    for i in indices:
        data = bytes(images_ds[i])
        if _is_jpeg(data):
            s = _laplacian_var_from_jpeg(data)
            if s is not None:
                scores.append(s)
    if not scores:
        return None, 0
    return float(np.mean(scores)), len(scores)


# =============================================================================
# Per-dimension checks
# =============================================================================

def check_metadata(f: h5py.File) -> Dimension:
    d = Dimension("Session metadata")
    attrs = dict(f.attrs)

    dur = attrs.get("n_frames", 0)
    ts  = f["timestamps_s"][:] if "timestamps_s" in f else None
    if ts is not None and len(ts) > 1:
        duration = float(ts[-1] - ts[0])
    else:
        duration = 0.0

    st = _pf(duration, DURATION_WARN, DURATION_FAIL)
    d.add(st, "duration", f"{duration:.1f} s", value=duration, weight=2)

    profile = attrs.get("ml2.profile", attrs.get("capture_profile", "unknown"))
    d.add(INFO, "profile", str(profile))

    n_frames = int(attrs.get("n_frames", 0))
    d.add(INFO, "rgb_frames", f"{n_frames} frames")

    # Check which streams are present
    streams = {
        "rgb": "rgb" in f,
        "imu": "imu" in f,
        "head_pose": "head_pose" in f,
        "depth": "depth" in f,
        "hand_tracking": "hand_tracking" in f,
        "eye_tracking": "eye_tracking" in f,
        "world_cam_0": "world_cam_0" in f,
        "audio": "audio" in f,
    }
    present = [k for k, v in streams.items() if v]
    missing = [k for k, v in streams.items() if not v]
    d.add(INFO, "streams_present", ", ".join(present) if present else "none")
    if missing:
        d.add(WARN, "streams_missing", ", ".join(missing), weight=0.5)

    d.compute_score()
    return d


def check_rgb(f: h5py.File) -> Dimension:
    d = Dimension("RGB camera")
    if "timestamps_s" not in f:
        d.add(FAIL, "rgb_timestamps", "timestamps_s missing from file", weight=3)
        d.compute_score()
        return d

    ts = f["timestamps_s"][:]
    if len(ts) < 2:
        d.add(FAIL, "rgb_frames", f"Only {len(ts)} frame(s)", weight=3)
        d.compute_score()
        return d

    diffs = np.diff(ts)
    fps   = 1.0 / np.mean(diffs)
    jitter_cv = np.std(diffs) / np.mean(diffs)
    max_gap   = float(np.max(diffs))
    n_dropped = int(np.sum(diffs > 2 * np.mean(diffs)))
    drop_pct  = n_dropped / len(ts) * 100

    # FPS
    st = _pf(fps, RGB_FPS_WARN, RGB_FPS_FAIL)
    d.add(st, "fps",
          f"{fps:.1f} Hz  (target {RGB_FPS_TARGET} Hz)", value=fps, weight=3)
    if fps < RGB_FPS_TARGET * 0.9:
        d.add(WARN, "fps_note",
              "RGB is below hardware max. Possible C++ encode backpressure or VRS queue drop.")

    # Jitter
    st = _pf(jitter_cv, RGB_JITTER_WARN, RGB_JITTER_FAIL, low_is_bad=False)
    d.add(st, "jitter_cv",
          f"CV={jitter_cv:.1%}  std={np.std(diffs)*1000:.1f} ms", value=jitter_cv, weight=2)

    # Max gap
    st = _pf(max_gap, RGB_MAX_GAP_WARN, RGB_MAX_GAP_FAIL, low_is_bad=False)
    d.add(st, "max_gap",
          f"{max_gap*1000:.0f} ms", value=max_gap, weight=2)

    # Dropped frames
    if n_dropped == 0:
        d.add(PASS, "dropped_frames", "0")
    elif drop_pct < 5:
        d.add(WARN, "dropped_frames", f"{n_dropped} ({drop_pct:.1f}%)", weight=1)
    else:
        d.add(FAIL, "dropped_frames", f"{n_dropped} ({drop_pct:.1f}%)", weight=1)

    # Image blur (world cams are JPEG; RGB is H.264 — requires video decoder)
    if "rgb/images" in f:
        images_ds = f["rgb/images"]
        blur, n_decoded = _sample_blur(images_ds)
        if blur is not None:
            st = _pf(blur, RGB_BLUR_WARN, RGB_BLUR_FAIL)
            d.add(st, "blur_score", f"{blur:.0f} (higher=sharper)", value=blur, weight=1)
        else:
            d.add(INFO, "blur_check",
                  "RGB frames are H.264 — pixel-level blur check requires a video decoder. "
                  "Install opencv-python and pass full H.264 stream (future work).")

    d.compute_score()
    return d


def check_imu(f: h5py.File) -> Dimension:
    d = Dimension("IMU")
    if "imu" not in f:
        d.add(FAIL, "imu_present", "stream missing", weight=3)
        d.compute_score()
        return d

    ts  = f["imu/timestamps_s"][:]
    acc = f["imu/accelerometer"][:]
    gyr = f["imu/gyroscope"][:]

    if len(ts) < 2:
        d.add(FAIL, "imu_samples", f"Only {len(ts)} sample(s)", weight=3)
        d.compute_score()
        return d

    diffs  = np.diff(ts)
    hz     = 1.0 / np.mean(diffs)
    max_gap = float(np.max(diffs))
    n_gaps  = int(np.sum(diffs > IMU_GAP_WARN))

    st = _pf(hz, IMU_HZ_WARN, IMU_HZ_FAIL)
    d.add(st, "sample_rate", f"{hz:.0f} Hz", value=hz, weight=2)

    st = _pf(max_gap, IMU_GAP_WARN, IMU_GAP_FAIL, low_is_bad=False)
    d.add(st, "max_gap", f"{max_gap*1000:.1f} ms", value=max_gap, weight=2)

    if n_gaps > 0:
        d.add(WARN, "gap_count",
              f"{n_gaps} gaps >{IMU_GAP_WARN*1000:.0f} ms  "
              f"(can cause IMU integration drift in training)", weight=1)
    else:
        d.add(PASS, "gap_count", "none")

    # Accelerometer norm at rest ≈ 9.81 m/s²
    acc_norms = np.linalg.norm(acc, axis=1)
    mean_norm = float(np.mean(acc_norms))
    norm_err  = abs(mean_norm - IMU_GRAVITY)
    if norm_err < IMU_GRAVITY_TOL:
        d.add(PASS, "accel_norm",
              f"{mean_norm:.3f} m/s²  (|err|={norm_err:.3f})", value=mean_norm)
    else:
        d.add(WARN, "accel_norm",
              f"{mean_norm:.3f} m/s²  (expected ~{IMU_GRAVITY}, |err|={norm_err:.3f}) "
              f"— possible calibration or sustained motion during recording",
              value=mean_norm)

    # Gyro range sanity
    gyr_max = float(np.max(np.abs(gyr)))
    d.add(INFO, "gyro_max", f"{np.degrees(gyr_max):.1f} deg/s")

    d.compute_score()
    return d


def check_head_pose(f: h5py.File) -> Dimension:
    d = Dimension("Head pose")
    if "head_pose" not in f:
        d.add(FAIL, "head_pose_present", "stream missing", weight=2)
        d.compute_score()
        return d

    pos  = f["head_pose/position"][:]
    quat = f["head_pose/orientation"][:]

    if len(pos) < 2:
        d.add(FAIL, "head_pose_frames", f"Only {len(pos)} frame(s)", weight=2)
        d.compute_score()
        return d

    # Workspace volume — product of X/Y/Z axis ranges (bounding box)
    rng = pos.max(axis=0) - pos.min(axis=0)
    workspace_vol = float(np.prod(rng))

    st = _pf(workspace_vol, POSE_WORKSPACE_WARN, POSE_WORKSPACE_FAIL)
    d.add(st, "workspace_volume",
          f"{workspace_vol:.4f} m³ (bounding box X×Y×Z={rng[0]:.2f}×{rng[1]:.2f}×{rng[2]:.2f} m)",
          value=workspace_vol, weight=1)

    # Smoothness — velocity
    vel = np.diff(pos, axis=0)
    speed = np.linalg.norm(vel, axis=1)
    d.add(INFO, "mean_speed", f"{np.mean(speed):.3f} m/frame")

    # Orientation validity (unit quaternions)
    qnorms = np.linalg.norm(quat, axis=1)
    bad_quat = int(np.sum(np.abs(qnorms - 1.0) > 0.01))
    if bad_quat == 0:
        d.add(PASS, "quaternion_validity", "all unit quaternions")
    else:
        d.add(WARN, "quaternion_validity",
              f"{bad_quat}/{len(quat)} non-unit quaternions", weight=1)

    d.compute_score()
    return d


def check_hand_tracking(f: h5py.File) -> Dimension:
    d = Dimension("Hand tracking")
    if "hand_tracking" not in f:
        d.add(WARN, "hand_present", "stream missing — required for manipulation tasks",
              weight=3)
        d.compute_score()
        return d

    lc = f["hand_tracking/left_confidence"][:]
    rc = f["hand_tracking/right_confidence"][:]
    lj = f["hand_tracking/left_joints"][:]
    rj = f["hand_tracking/right_joints"][:]

    left_rate  = float(np.mean(lc > 0))
    right_rate = float(np.mean(rc > 0))
    both_rate  = float(np.mean((lc > 0) & (rc > 0)))

    for side, rate in [("left", left_rate), ("right", right_rate)]:
        st = _pf(rate, HAND_DETECT_WARN, HAND_DETECT_FAIL)
        d.add(st, f"{side}_detection",
              f"{rate:.1%} of frames", value=rate, weight=2)

    d.add(INFO, "both_hands_detected", f"{both_rate:.1%} of frames")

    # Joint position sanity (all zeros = invalid / not tracked)
    l_valid = float(np.mean(np.any(lj != 0, axis=(1, 2))))
    r_valid = float(np.mean(np.any(rj != 0, axis=(1, 2))))
    d.add(INFO, "left_joints_nonzero",  f"{l_valid:.1%}")
    d.add(INFO, "right_joints_nonzero", f"{r_valid:.1%}")

    if right_rate < HAND_DETECT_WARN:
        d.add(WARN, "manipulation_note",
              "Low right-hand detection. During recording, ensure the hand "
              "performing manipulation is visible in the RGB/depth field of view.")

    d.compute_score()
    return d


def check_eye_tracking(f: h5py.File) -> Dimension:
    d = Dimension("Eye tracking")
    if "eye_tracking" not in f:
        d.add(WARN, "eye_present", "stream missing", weight=1)
        d.compute_score()
        return d

    fp = f["eye_tracking/fixation_point"][:]
    # Valid = fixation point is not all-zero
    valid = np.any(fp != 0, axis=1)
    rate  = float(np.mean(valid))

    st = _pf(rate, EYE_DETECT_WARN, EYE_DETECT_FAIL)
    d.add(st, "gaze_valid", f"{rate:.1%} of frames", value=rate, weight=2)

    # Gaze dispersion — how much of the scene is being looked at
    fp_valid = fp[valid]
    if len(fp_valid) > 1:
        fp_std = np.std(fp_valid, axis=0)
        d.add(INFO, "gaze_dispersion",
              f"std: X={fp_std[0]:.3f}  Y={fp_std[1]:.3f}  Z={fp_std[2]:.3f} m")

    d.compute_score()
    return d


def check_depth(f: h5py.File) -> Dimension:
    d = Dimension("Depth")
    if "depth" not in f or "depth/images" not in f:
        d.add(WARN, "depth_present", "stream missing", weight=2)
        d.compute_score()
        return d

    imgs = f["depth/images"]
    n = len(imgs)
    d.add(INFO, "depth_frames", str(n))

    if n == 0:
        d.add(FAIL, "depth_data", "0 frames", weight=2)
        d.compute_score()
        return d

    # Sample a subset for valid pixel ratio
    indices = np.linspace(0, n - 1, min(20, n), dtype=int)
    valid_ratios = []
    for i in indices:
        frame = imgs[i]
        valid_ratios.append(float(np.mean(frame > 0)))

    mean_valid = float(np.mean(valid_ratios))
    if mean_valid >= 0.7:
        d.add(PASS, "valid_pixels",
              f"{mean_valid:.1%} non-zero across sampled frames", value=mean_valid)
    elif mean_valid >= 0.4:
        d.add(WARN, "valid_pixels",
              f"{mean_valid:.1%} non-zero — significant invalid depth regions", value=mean_valid)
    else:
        d.add(FAIL, "valid_pixels",
              f"{mean_valid:.1%} non-zero — mostly invalid depth", value=mean_valid)

    # Depth range
    sample_frame = imgs[n // 2]
    valid_depths = sample_frame[sample_frame > 0].astype(np.float32)
    if len(valid_depths):
        d.add(INFO, "depth_range_mm",
              f"{valid_depths.min():.0f}–{valid_depths.max():.0f} mm "
              f"(median {np.median(valid_depths):.0f} mm)")

    d.compute_score()
    return d


def check_world_cams(f: h5py.File) -> Dimension:
    d = Dimension("World cameras")
    cam_keys = [k for k in ["world_cam_0", "world_cam_1", "world_cam_2"] if k in f]

    if not cam_keys:
        d.add(WARN, "world_cams_present", "no world cameras in file", weight=1)
        d.compute_score()
        return d

    d.add(INFO, "cameras_present", ", ".join(cam_keys))

    for cam in cam_keys:
        images_ds = f[f"{cam}/images"]
        n = len(images_ds)
        d.add(INFO, f"{cam}_frames", str(n))

        if n == 0:
            d.add(FAIL, f"{cam}_data", "0 frames", weight=1)
            continue

        blur, n_dec = _sample_blur(images_ds, n_samples=8)
        if blur is not None and n_dec > 0:
            st = _pf(blur, WC_BLUR_WARN, WC_BLUR_FAIL)
            d.add(st, f"{cam}_blur", f"{blur:.0f} (n={n_dec})", value=blur, weight=1)
        else:
            d.add(INFO, f"{cam}_blur", "could not decode sample frames")

    d.compute_score()
    return d


def check_audio(f: h5py.File) -> Dimension:
    d = Dimension("Audio")
    if "audio" not in f or "audio/pcm" not in f:
        d.add(INFO, "audio_present", "stream not recorded")
        d.compute_score()
        return d

    pcm = f["audio/pcm"]
    n_samples, n_ch = pcm.shape
    d.add(INFO, "channels", str(n_ch))
    d.add(INFO, "samples",  str(n_samples))

    # Infer true sample rate from stored attr + cross-check against session length
    stored_rate = int(f["audio"].attrs.get("sample_rate", 48000))
    dur_at_stored = n_samples / stored_rate

    # Snap n_samples to a standard audio rate to find true hardware rate
    STANDARD_RATES = [8000, 16000, 22050, 32000, 44100, 48000, 96000, 192000]
    # For each candidate rate, compute what session duration it would imply
    # then find the rate whose implied duration best matches session_dur
    if "timestamps_s" in f:
        rgb_ts = f["timestamps_s"][:]
        session_dur = float(rgb_ts[-1] - rgb_ts[0]) if len(rgb_ts) > 1 else 0.0
        # Also check the nominal recording duration from attrs
        nominal_dur = float(f.attrs.get("ml2.duration_s", 0) or session_dur)
        ref_dur = nominal_dur if nominal_dur > 0 else session_dur

        best_rate  = stored_rate
        best_error = abs(n_samples / stored_rate - ref_dur)
        for rate in STANDARD_RATES:
            err = abs(n_samples / rate - ref_dur)
            if err < best_error:
                best_error = err
                best_rate  = rate

        if best_rate != stored_rate:
            d.add(WARN, "sample_rate_mismatch",
                  f"stored metadata says {stored_rate} Hz, but {n_samples} samples "
                  f"over nominal {ref_dur:.0f}s matches {best_rate} Hz (err={best_error:.2f}s). "
                  f"Fix: set ml2.streams.audio.sample_rate VRS tag to {best_rate}.",
                  weight=0.5)
            dur_s = n_samples / best_rate
            d.add(INFO, "duration_s", f"{dur_s:.1f} s @ {best_rate} Hz (corrected)")
        else:
            dur_s = dur_at_stored
            d.add(INFO, "duration_s", f"{dur_s:.1f} s @ {stored_rate} Hz")
    else:
        dur_s = dur_at_stored
        d.add(INFO, "duration_s", f"{dur_s:.1f} s @ {stored_rate} Hz")

    # Clipping check (int16 max = 32767)
    sample_data = pcm[::1000, :]  # every 1000th sample
    clip_frac = float(np.mean(np.abs(sample_data) >= 32000))
    if clip_frac < 0.001:
        d.add(PASS, "clipping", f"{clip_frac:.4%} clipped")
    elif clip_frac < 0.01:
        d.add(WARN, "clipping", f"{clip_frac:.3%} clipped", weight=0.5)
    else:
        d.add(FAIL, "clipping", f"{clip_frac:.2%} clipped — distorted audio", weight=0.5)

    d.compute_score()
    return d


def check_multimodal_sync(f: h5py.File) -> Dimension:
    """Check timestamp alignment across streams."""
    d = Dimension("Multi-modal synchronisation")
    if "timestamps_s" not in f:
        d.add(WARN, "rgb_timestamps", "missing — cannot check sync", weight=2)
        d.compute_score()
        return d

    rgb_ts  = f["timestamps_s"][:]
    rgb_t0, rgb_t1 = float(rgb_ts[0]), float(rgb_ts[-1])

    # IMU start lag — IMU stream may start up to ~1s after RGB due to hw init
    if "imu/timestamps_s" in f:
        imu_ts = f["imu/timestamps_s"][:]
        imu_t0, imu_t1 = float(imu_ts[0]), float(imu_ts[-1])
        start_lag = imu_t0 - rgb_t0  # positive = IMU starts after RGB
        rgb_dur = rgb_t1 - rgb_t0
        imu_dur = imu_t1 - imu_t0

        if abs(start_lag) < 1.0:
            d.add(PASS, "imu_start_lag",
                  f"{start_lag*1000:.0f} ms (hardware init offset, normal)",
                  value=start_lag, weight=0.5)
        else:
            d.add(WARN, "imu_start_lag",
                  f"{start_lag*1000:.0f} ms — large lag may indicate IMU missed early frames",
                  value=start_lag, weight=0.5)

        # End coverage — IMU should not stop significantly before RGB
        end_gap = rgb_t1 - imu_t1  # positive = IMU ended before RGB
        if end_gap < 3.0:
            d.add(PASS, "imu_end_coverage",
                  f"IMU={imu_dur:.1f}s  RGB={rgb_dur:.1f}s",
                  value=end_gap, weight=1)
        else:
            d.add(WARN, "imu_end_coverage",
                  f"IMU stopped {end_gap:.1f}s before RGB end — "
                  f"{end_gap/rgb_dur:.0%} of session has no IMU data",
                  value=end_gap, weight=1)

    d.compute_score()
    return d


# =============================================================================
# Report rendering
# =============================================================================

DIMENSION_WEIGHTS = {
    "Session metadata":           1.0,
    "RGB camera":                 3.0,   # primary observation for policy
    "IMU":                        2.0,
    "Head pose":                  1.5,
    "Hand tracking":              2.5,   # critical for manipulation
    "Eye tracking":               0.5,
    "Depth":                      1.5,
    "World cameras":              0.5,
    "Audio":                      0.2,
    "Multi-modal synchronisation":1.0,
}


def overall_score(dims: List[Dimension]) -> float:
    total_w = 0.0
    weighted = 0.0
    for d in dims:
        w = DIMENSION_WEIGHTS.get(d.name, 1.0)
        weighted += d.score * w
        total_w  += w
    return weighted / total_w if total_w else 0.0


def render_report(dims: List[Dimension], h5_path: Path) -> float:
    print(f"\n{'=' * 70}")
    print(f"  ML2 ROBOT-TRAINING DATA QUALITY REPORT")
    print(f"  {h5_path.name}")
    print(f"{'=' * 70}")

    for d in dims:
        score_col = "\033[32m" if d.score >= 75 else "\033[33m" if d.score >= 50 else "\033[31m"
        print(f"\n{BOLD}── {d.name}{RESET}  "
              f"[{score_col}{d.score:.0f}/100{RESET}]")
        for c in d.checks:
            _print_check(c)

    score = overall_score(dims)
    score_col = "\033[32m" if score >= 75 else "\033[33m" if score >= 50 else "\033[31m"

    print(f"\n{'=' * 70}")
    print(f"  OVERALL SUITABILITY SCORE: {score_col}{BOLD}{score:.0f} / 100{RESET}")

    if score >= 80:
        verdict = f"\033[32m{BOLD}READY for robot training{RESET}"
    elif score >= 60:
        verdict = f"\033[33m{BOLD}MARGINAL — review warnings before training{RESET}"
    else:
        verdict = f"\033[31m{BOLD}NOT READY — too many quality issues{RESET}"

    print(f"  Verdict: {verdict}")
    print(f"{'=' * 70}\n")

    # Summary table
    print("  Stream scores:")
    for d in dims:
        bar_len = int(d.score / 100 * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        sc = d.score
        col = "\033[32m" if sc >= 75 else "\033[33m" if sc >= 50 else "\033[31m"
        print(f"    {d.name:<32} {col}{bar}{RESET} {sc:5.0f}")
    print()

    return score


def to_dict(dims: List[Dimension], score: float) -> dict:
    return {
        "overall_score": round(score, 1),
        "dimensions": [
            {
                "name": d.name,
                "score": round(d.score, 1),
                "checks": [
                    {"status": c.status, "name": c.name,
                     "detail": c.detail, "value": c.value}
                    for c in d.checks
                ],
            }
            for d in dims
        ],
    }


# =============================================================================
# Main
# =============================================================================

def quality_check(h5_path: Path) -> Tuple[List[Dimension], float]:
    with h5py.File(h5_path, "r") as f:
        dims = [
            check_metadata(f),
            check_rgb(f),
            check_imu(f),
            check_head_pose(f),
            check_hand_tracking(f),
            check_eye_tracking(f),
            check_depth(f),
            check_world_cams(f),
            check_audio(f),
            check_multimodal_sync(f),
        ]
    score = overall_score(dims)
    return dims, score


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate ML2 session quality for robot training")
    parser.add_argument("session", type=Path,
                        help="Path to .h5 session file (or .vrs — will auto-convert)")
    parser.add_argument("--min-score", type=float, default=60.0,
                        help="Exit code 1 if overall score below this (default 60)")
    parser.add_argument("--json", type=Path, default=None,
                        help="Write JSON report to this path")
    args = parser.parse_args()

    path = args.session.resolve()

    # Auto-convert VRS → HDF5 if needed
    if path.suffix == ".vrs":
        h5_path = path.with_suffix(".h5")
        if not h5_path.exists():
            print(f"Converting {path.name} → {h5_path.name} ...")
            import subprocess
            script = Path(__file__).parent / "convert_session_to_hdf5.py"
            result = subprocess.run(
                [sys.executable, str(script), str(path)], check=False)
            if result.returncode != 0 or not h5_path.exists():
                print("ERROR: HDF5 conversion failed.")
                sys.exit(1)
        path = h5_path

    if not path.exists():
        print(f"ERROR: file not found: {path}")
        sys.exit(1)

    dims, score = quality_check(path)
    render_report(dims, path)

    if args.json:
        data = to_dict(dims, score)
        args.json.write_text(json.dumps(data, indent=2))
        print(f"JSON report written to {args.json}")

    sys.exit(0 if score >= args.min_score else 1)


if __name__ == "__main__":
    main()

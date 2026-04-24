#!/usr/bin/env python3
"""
view_session.py — Visual playback of an ML2 recorder VRS session.

Displays synchronized RGB frames with overlaid head pose trajectory,
gaze vectors, and hand keypoints.  Also shows colorized depth maps.

Usage:
    python view_session.py /path/to/session.vrs
    python view_session.py /path/to/session.vrs --fps 10 --start 100

Controls:
    Space    — pause / resume
    → / ←   — step forward / backward (when paused)
    q / Esc  — quit

Requires:
    pip install opencv-python numpy vrs
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    import pyvrs
except ImportError:
    print("ERROR: pyvrs not installed.  Run:  pip install vrs")
    sys.exit(1)

# ---------------------------------------------------------------------------
# VRS stream flavors
# ---------------------------------------------------------------------------

FL_RGB       = "ml2/rgb"
FL_DEPTH     = "ml2/depth"
FL_HEAD_POSE = "ml2/head_pose"
FL_EYE       = "ml2/eye"
FL_HAND      = "ml2/hand"


# ---------------------------------------------------------------------------
# VRS session loader
# ---------------------------------------------------------------------------

class VrsSession:
    """Load and index all sensor streams from a VRS session file."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.reader = pyvrs.SyncVRSReader(str(path))
        self._flavor_map = self._build_flavor_map()

        # Load stream indices (sorted by timestamp)
        self.rgb_ts, self.rgb_recs   = self._load_data_records(FL_RGB)
        self.depth_ts, self.depth_recs = self._load_data_records(FL_DEPTH)
        self.head_ts, self.head_recs   = self._load_data_records(FL_HEAD_POSE)
        self.eye_ts,  self.eye_recs    = self._load_data_records(FL_EYE)
        self.hand_ts, self.hand_recs   = self._load_data_records(FL_HAND)

        self.n_frames = len(self.rgb_ts)

    # -- internals --

    def _build_flavor_map(self) -> Dict[str, list]:
        fm: Dict[str, list] = {}
        for sid in self.reader.stream_ids:
            try:
                fl = self.reader.get_stream_tags(sid).get("flavor", "")
            except Exception:
                fl = ""
            if fl:
                fm.setdefault(fl, []).append(sid)
        return fm

    def _load_data_records(self, flavor: str) -> Tuple[np.ndarray, list]:
        sids = self._flavor_map.get(flavor, [])
        if not sids:
            return np.array([], dtype=np.float64), []

        recs = []
        for sid in sids:
            try:
                filtered = self.reader.filtered_by_fields(stream_ids={sid},
                                                           record_types={"data"})
                recs.extend(list(filtered))
            except Exception:
                for r in self.reader:
                    if r.stream_id == sid and str(r.record_type).lower() == "data":
                        recs.append(r)

        recs.sort(key=lambda r: r.timestamp)
        ts = np.array([r.timestamp for r in recs], dtype=np.float64)
        return ts, recs

    # -- public frame access --

    def _content_bytes(self, rec) -> bytes:
        try:
            blocks = rec.read_content_blocks()
            if blocks:
                b = blocks[0]
                return bytes(b) if isinstance(b, (bytes, bytearray, memoryview)) else b.get_buffer()
        except Exception:
            pass
        return b""

    def _layout_val(self, rec, field, default=None):
        try:
            return rec.data_layout[field].get()
        except Exception:
            pass
        return default

    def _layout_arr(self, rec, field) -> Optional[np.ndarray]:
        try:
            return np.array(rec.data_layout[field].get(), dtype=np.float32)
        except Exception:
            return None

    def _nearest_idx(self, timestamp: float, source_ts: np.ndarray) -> int:
        if len(source_ts) == 0:
            return -1
        idx = int(np.searchsorted(source_ts, timestamp))
        idx = min(max(idx, 1), len(source_ts) - 1)
        if abs(timestamp - source_ts[idx - 1]) <= abs(timestamp - source_ts[idx]):
            return idx - 1
        return idx

    def get_rgb_frame(self, i: int) -> Optional[np.ndarray]:
        if i < 0 or i >= len(self.rgb_recs):
            return None
        raw = self._content_bytes(self.rgb_recs[i])
        if not raw:
            return None
        buf = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        return img  # may be None if content is not a JPEG (e.g. H.264 config NAL)

    def get_depth_frame(self, timestamp: float) -> Optional[np.ndarray]:
        idx = self._nearest_idx(timestamp, self.depth_ts)
        if idx < 0:
            return None
        raw = self._content_bytes(self.depth_recs[idx])
        if not raw:
            return None
        rec   = self.depth_recs[idx]
        w     = int(self._layout_val(rec, "width") or 0)
        h     = int(self._layout_val(rec, "height") or 0)
        if w > 0 and h > 0 and len(raw) >= w * h * 2:
            return np.frombuffer(raw, dtype=np.uint16)[:w * h].reshape(h, w)
        return None

    def get_head_pose(self, timestamp: float) -> Optional[np.ndarray]:
        idx = self._nearest_idx(timestamp, self.head_ts)
        if idx < 0:
            return None
        rec = self.head_recs[idx]
        pos = self._layout_arr(rec, "position") or np.zeros(3)
        ori = self._layout_arr(rec, "orientation") or np.zeros(4)
        return np.concatenate([pos, ori])  # [x, y, z, w, qx, qy, qz]

    def get_eye_data(self, timestamp: float) -> Optional[np.ndarray]:
        idx = self._nearest_idx(timestamp, self.eye_ts)
        if idx < 0:
            return None
        rec = self.eye_recs[idx]
        lo = self._layout_arr(rec, "left_origin") or np.zeros(3)
        ld = self._layout_arr(rec, "left_direction") or np.zeros(3)
        ro = self._layout_arr(rec, "right_origin") or np.zeros(3)
        rd = self._layout_arr(rec, "right_direction") or np.zeros(3)
        fi = self._layout_arr(rec, "fixation") or np.zeros(3)
        return np.concatenate([lo, ld, ro, rd, fi])  # 15 floats

    def get_hand_data(self, timestamp: float) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        idx = self._nearest_idx(timestamp, self.hand_ts)
        if idx < 0:
            return None
        rec = self.hand_recs[idx]
        lv = bool(self._layout_val(rec, "left_valid") or False)
        rv = bool(self._layout_val(rec, "right_valid") or False)
        lk = (self._layout_arr(rec, "left_keypoints") or np.full(84, np.nan)).reshape(28, 3)
        rk = (self._layout_arr(rec, "right_keypoints") or np.full(84, np.nan)).reshape(28, 3)
        if not lv:
            lk[:] = np.nan
        if not rv:
            rk[:] = np.nan
        return lk, rk  # (28×3, 28×3)


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def colorize_depth(depth_u16: np.ndarray, max_mm: int = 5000) -> np.ndarray:
    depth_f = np.clip(depth_u16.astype(np.float32) / max_mm, 0, 1)
    colored = cv2.applyColorMap((depth_f * 255).astype(np.uint8), cv2.COLORMAP_JET)
    colored[depth_u16 == 0] = [0, 0, 0]
    return colored


def draw_gaze(img: np.ndarray, eye_data: np.ndarray) -> None:
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2
    scale = min(w, h) * 0.3
    # left direction: indices 3-5; right direction: indices 9-11
    for d, color, ox in [(eye_data[3:6], (0, 255, 0), -30),
                         (eye_data[9:12], (0, 0, 255), 30)]:
        if np.any(np.isnan(d)):
            continue
        start = (cx + ox, cy)
        end   = (int(cx + ox + d[0] * scale), int(cy - d[1] * scale))
        cv2.arrowedLine(img, start, end, color, 2, tipLength=0.2)


def draw_hand_keypoints(img: np.ndarray,
                        hand: Tuple[np.ndarray, np.ndarray]) -> None:
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2
    scale = min(w, h) * 2.0
    lk, rk = hand
    for joints, color in [(lk, (0, 255, 255)), (rk, (255, 0, 255))]:
        if np.all(np.isnan(joints)):
            continue
        for j in range(len(joints)):
            if np.isnan(joints[j, 0]):
                continue
            px = int(cx + joints[j, 0] * scale)
            py = int(cy - joints[j, 1] * scale)
            if 0 <= px < w and 0 <= py < h:
                cv2.circle(img, (px, py), 3, color, -1)


def draw_info_bar(img: np.ndarray, frame_idx: int, n_frames: int,
                  timestamp_s: float,
                  head_pose: Optional[np.ndarray],
                  paused: bool) -> np.ndarray:
    h, w = img.shape[:2]
    bar = np.zeros((40, w, 3), dtype=np.uint8)

    label = f"Frame {frame_idx}/{n_frames - 1}"
    if paused:
        label += "  [PAUSED]"
    cv2.putText(bar, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.putText(bar, f"t={timestamp_s:.3f}s", (w // 3, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    if head_pose is not None and not np.any(np.isnan(head_pose[:3])):
        pos_txt = f"Head: ({head_pose[0]:.2f}, {head_pose[1]:.2f}, {head_pose[2]:.2f})"
        cv2.putText(bar, pos_txt, (2 * w // 3, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 255), 1)

    progress = frame_idx / max(n_frames - 1, 1)
    cv2.rectangle(bar, (0, 36), (int(w * progress), 40), (0, 200, 0), -1)

    return np.vstack([img, bar])


# ---------------------------------------------------------------------------
# Main playback loop
# ---------------------------------------------------------------------------

def view_session(session: VrsSession, fps: int = 15, start: int = 0) -> None:
    if session.n_frames == 0:
        print("No RGB frames to display.")
        return

    win = "ML2 Session Viewer"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    frame_idx = min(start, session.n_frames - 1)
    paused    = False
    delay_ms  = max(1, 1000 // fps)

    while True:
        ts = session.rgb_ts[frame_idx]

        # RGB frame
        rgb = session.get_rgb_frame(frame_idx)
        if rgb is None:
            # May be a config NAL — skip forward silently
            if not paused:
                frame_idx = (frame_idx + 1) % session.n_frames
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            continue

        # Overlays
        eye = session.get_eye_data(ts)
        if eye is not None:
            draw_gaze(rgb, eye)

        hand = session.get_hand_data(ts)
        if hand is not None:
            draw_hand_keypoints(rgb, hand)

        head = session.get_head_pose(ts)
        display = draw_info_bar(rgb, frame_idx, session.n_frames, ts, head, paused)

        # Depth side window
        depth = session.get_depth_frame(ts)
        if depth is not None:
            cv2.imshow("Depth", colorize_depth(depth))

        cv2.imshow(win, display)

        key = cv2.waitKey(0 if paused else delay_ms) & 0xFF

        if key in (ord("q"), 27):
            break
        elif key == ord(" "):
            paused = not paused
        elif key in (83, ord("d")):   # Right arrow / d
            frame_idx = min(frame_idx + 1, session.n_frames - 1)
        elif key in (81, ord("a")):   # Left arrow / a
            frame_idx = max(frame_idx - 1, 0)
        elif not paused:
            frame_idx = (frame_idx + 1) % session.n_frames

    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visual playback of ML2 recorder VRS session")
    parser.add_argument("vrs_path", type=Path,
                        help="Path to session .vrs file")
    parser.add_argument("--fps",   type=int, default=15,
                        help="Playback frame rate (default: 15)")
    parser.add_argument("--start", type=int, default=0,
                        help="Start frame index (default: 0)")

    args = parser.parse_args()
    path = args.vrs_path.resolve()

    if not path.exists():
        print(f"ERROR: File not found: {path}")
        sys.exit(1)

    print(f"Loading: {path}")
    session = VrsSession(path)
    print(f"  RGB frames:    {session.n_frames}")
    print(f"  Depth frames:  {len(session.depth_ts)}")
    print(f"  Head pose:     {len(session.head_ts)} samples")
    print(f"  Eye tracking:  {len(session.eye_ts)} samples")
    print(f"  Hand tracking: {len(session.hand_ts)} samples")
    print()
    print("Controls: Space=pause  →/←=step  q=quit")
    print()

    view_session(session, fps=args.fps, start=args.start)


if __name__ == "__main__":
    main()

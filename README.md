# ml2-recorder

Multi-sensor recorder for the Magic Leap 2, producing synchronized datasets for
egocentric robotics and world-model training (EgoMimic / OSMO-style pipelines).
All capture runs on-device in native C++. Sessions are pulled over ADB and
converted to HDF5 on the host.

See [`docs/schema.md`](docs/schema.md) for the VRS-to-HDF5 data flow and
[`docs/testing.md`](docs/testing.md) for what can be checked with and without
ML2 hardware.

## What it records

Every stream is timestamped in a common clock and written to a single VRS file:

| Stream | Format | Rate |
|---|---|---|
| RGB | H.264 Annex B | 15 fps |
| Depth + confidence | uint16 mm | ~5 fps |
| World cameras (×3) | JPEG grayscale | ~8 fps |
| Head pose | position + quaternion | ~30 Hz |
| Eye tracking | binocular rays + fixation | ~30 Hz |
| Hand tracking | 28 keypoints × 2 hands | ~30 Hz |
| IMU | accel + gyro, 4 units | ~1.7 kHz aggregate |
| Audio | 4-channel PCM | 192 kHz |
| Spatial mesh | vertices + indices | 1 Hz |
| **Camera extrinsics** | per-frame `world_from_camera` | matches each image stream |
| Camera intrinsics | fx/fy/cx/cy/distortion | static per camera |

## Prerequisites

- Magic Leap 2 headset
- [MLSDK v1.12.0+](https://ml2-developer.magicleap.com/) (proprietary; requires a free Magic Leap developer account)
- Android NDK 30, CMake 4.1+
- OpenJDK 17+ (`brew install openjdk` or `sudo apt install openjdk-17-jdk`)
- `adb` on PATH
- Python 3.10+ (`pip install -r requirements.txt` for host tools)

## Build

```bash
export MLSDK=$HOME/MagicLeap/mlsdk/v1.12.0     # wherever you unpacked MLSDK
export JAVA_HOME=$(brew --prefix openjdk)      # or your JDK path

./gradlew assembleDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

## Record a session

```bash
adb shell am start -n com.ml2.recorder/.RecorderActivity \
  --es profile full_quality \
  --ei duration 60
```

Profiles: `full_quality` (default, 15 fps RGB), `high_temporal` (30 fps RGB),
`lightweight` (single world cam, reduced depth).

Stop early by pinching both hands together for 2 seconds, or let the duration
expire.

## Pull and convert

```bash
adb pull /storage/emulated/0/Android/data/com.ml2.recorder/files/recordings/session_YYYYMMDD_HHMMSS.vrs
python tools/convert_session_to_hdf5.py session_YYYYMMDD_HHMMSS.vrs
```

The converter produces `session_YYYYMMDD_HHMMSS.h5` with RGB as the temporal
anchor, other streams nearest-matched to RGB timestamps, IMU kept at native
rate. Every image stream carries per-frame `camera_pose/` (world_from_camera),
and `calibration/{cam}/camera_to_head/` holds the rigid head-mount transform
derived from the per-frame poses.

## Visualize

Rerun (recommended):

```bash
python tools/view_rerun.py session.h5
```

Lightweight OpenCV viewer:

```bash
python tools/view_session.py session_YYYYMMDD_HHMMSS.vrs
```

One-command install → record → pull → validate → convert:

```bash
./tools/quicktest.sh 60 --profile full
```

## Session schema

Top-level HDF5 layout (one sample per RGB frame, unless noted):

```
/
├── timestamps_s, timestamps_ns
├── rgb/images, rgb/camera_pose/{position,orientation,valid}
├── depth/images, depth/confidence, depth/camera_pose/{...}
├── world_cam_0/images, world_cam_0/camera_pose/{...}   # + _1, _2
├── head_pose/{position, orientation}
├── eye_tracking/{left_origin, left_direction, right_origin, right_direction, fixation_point}
├── hand_tracking/{left_joints, right_joints, left_confidence, right_confidence}
├── imu/{timestamps_s, timestamps_ns, unit_id, accelerometer, gyroscope}   # native rate
├── audio/pcm
├── mesh/{vertices, indices, normals, timestamps_s, ...}   # native rate
└── calibration/{rgb,depth,world_cam_0,world_cam_1,world_cam_2}/
    ├── (attrs) fx, fy, cx, cy, width, height
    ├── distortion
    └── camera_to_head/{position, orientation, position_stddev, ...}
```

Quaternions are Hamilton, scalar-first (w, x, y, z). Coordinates are in the
ML2 gravity-aligned world frame. Units are metres and radians unless marked
otherwise.

See `sample_dataset/load_sample.py` for a minimal reader that touches every
stream.

## Project layout

```
app/src/main/cpp/          Native C++ recorder
  main.cpp                 Entry point, recording loop
  capture_profile.h        Profile presets
  rgb_capture.{h,cpp}      H.264 + CV camera pose
  depth_capture.{h,cpp}    Depth + confidence
  world_camera_capture.{h,cpp}
  perception_capture.{h,cpp}  Head / eye / hand
  imu_capture.{h,cpp}
  audio_capture.{h,cpp}
  meshing_capture.{h,cpp}
  vrs_writer.{h,cpp}       VRS serialisation
  write_queue.{h,cpp}      Async write pool

tools/
  convert_session_to_hdf5.py   VRS → HDF5, with interpolation
  validate_session.py          Acceptance gate
  view_rerun.py                Rerun visualiser
  view_session.py              OpenCV visualiser
  quality_check.py             Per-stream QC scoring
  quicktest.sh                 One-command end-to-end test
  export_session.sh            Device session management

tests/                     Host-side GTest unit tests
sample_dataset/            Minimal loader example
```

## License

Apache 2.0 — see `LICENSE`. Third-party components and their licences are
listed in `THIRD_PARTY.md`. Magic Leap SDK is proprietary and linked at build
time; it is not redistributed with this project.

## Contributing

Bug reports, patches, and hardware porting notes welcome. See
[`CONTRIBUTING.md`](CONTRIBUTING.md) for the dev loop.

Security issues: please follow [`SECURITY.md`](SECURITY.md) rather than
opening a public issue.

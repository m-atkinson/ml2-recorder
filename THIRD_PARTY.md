# Third-Party Components

This project vendors or fetches several third-party components. Their licenses
apply to those components only.

## Fetched at build time (via CMake `FetchContent`)

| Component | Purpose | License |
|---|---|---|
| [VRS](https://github.com/facebookresearch/vrs) | Sensor-stream container format | Apache 2.0 |
| [fmt](https://github.com/fmtlib/fmt) | C++ string formatting | MIT |
| [lz4](https://github.com/lz4/lz4) | Fast compression | BSD-2-Clause |
| [zstd](https://github.com/facebook/zstd) | Compression | BSD-3-Clause / GPL-2 dual |
| [xxHash](https://github.com/Cyan4973/xxHash) | Non-cryptographic hashing | BSD-2-Clause |
| [RapidJSON](https://github.com/Tencent/rapidjson) | JSON parsing | MIT |

## Vendored

| File | Source | License |
|---|---|---|
| `app/src/main/cpp/stb_image_write.h` | [stb](https://github.com/nothings/stb) | Public domain / MIT (dual) |
| `app/src/main/cpp/boost_shim/` | Minimal shims to replace Boost.Interprocess + Boost.UUID for NDK builds | Authored here, Apache 2.0 with this project |

## Runtime (linked, not redistributed)

- **Magic Leap SDK** (`libcamera.magicleap.so`, `libperception.magicleap.so`,
  etc.) — proprietary, supplied by Magic Leap under their developer license.
  Users must obtain and install it separately from
  <https://ml2-developer.magicleap.com/>.

## Android platform libraries

Linked from the NDK / platform — `android`, `log`, `mediandk`, `OpenSLES`.
Governed by Android's Apache 2.0 license.

## Python host tooling

The Python scripts in `tools/` and `sample_dataset/` rely on:

- `numpy`, `h5py`, `pandas`, `scipy` — BSD-family
- `opencv-python` — Apache 2.0
- `pillow` — HPND
- `pyvrs` (from the VRS project above) — Apache 2.0
- `rerun-sdk` — MIT / Apache 2.0
- `av` (PyAV) — BSD-3-Clause

All are installed by the user via `pip` and are not bundled here.

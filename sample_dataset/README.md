# Sample loader

This directory holds a minimal reader showing how to open a session HDF5
produced by `tools/convert_session_to_hdf5.py` and touch every stream.

```bash
pip install h5py numpy av
python sample_dataset/load_sample.py path/to/session.h5
```

The script prints a one-frame dump: head pose, eye fixation, hand confidence
and keypoint validity, depth coverage, per-camera world_from_camera
extrinsics, and the static `camera_to_head` transforms from `/calibration/`.

It is robust to missing optional streams (sessions recorded with reduced
profiles, or pre-extrinsics sessions that lack `camera_pose/` groups) — use
it as a template when writing your own reader.

## Working on new sessions

If you have an ML2 recorder of your own (see the top-level README), produce
a session with:

```bash
./tools/quicktest.sh 30 --profile full
```

This builds, installs, records a 30-second clip, pulls it over ADB, runs
`validate_session.py`, and converts the VRS to HDF5. The resulting `.h5`
can be fed directly to `load_sample.py` or a Rerun viewer
(`tools/view_rerun.py`).

## Redistributing sample data

This directory does **not** currently ship with pre-recorded sessions.
If you want to publish a sample dataset alongside a fork or application
note, consider:

- **GitHub Releases** — free for files up to 2 GB
- **Hugging Face Datasets** — natural home for robotics corpora, good
  discoverability
- A self-hosted bucket — simplest if you already have cloud storage

Whatever you choose, regenerate `SHA256SUMS.txt` at publish time and
include it in the release description so consumers can verify integrity:

```bash
shasum -a 256 *.h5 > SHA256SUMS.txt
```

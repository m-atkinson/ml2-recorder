# Testing Guide

Some checks require a Magic Leap 2 headset because the recorder depends on ML2
sensor APIs and the proprietary MLSDK. Host-only checks still help verify the
Python tools, documentation, and data-processing path.

## Without ML2 Hardware

These checks do not require a headset:

```bash
python -m pip install -r requirements.txt
python tools/validate_session.py --help
python tools/convert_session_to_hdf5.py --help
python tools/quality_check.py --help
python tools/inspect_frame.py --help
```

Use these to confirm the Python environment can import the required packages and
that command-line entry points are available. Full conversion and visualization
still require a real `.vrs` or `.h5` session file. Once an HDF5 file is
available, the minimal reader can be checked with:

```bash
python sample_dataset/load_sample.py session_YYYYMMDD_HHMMSS.h5
```

## With ML2 Hardware

Build, install, record, pull, validate, convert, and score a short session:

```bash
export MLSDK=$HOME/MagicLeap/mlsdk/v1.12.0
export JAVA_HOME=$(brew --prefix openjdk)

./gradlew assembleDebug
./tools/quicktest.sh 30 --profile full_quality
```

The expected outputs are written under `pulled_sessions/`:

```text
session_YYYYMMDD_HHMMSS.vrs
session_YYYYMMDD_HHMMSS.h5
session_YYYYMMDD_HHMMSS_quality.json
```

## Manual Acceptance Checklist

For a partner-facing capture, keep the raw VRS file and the converted HDF5 file.
Before sharing results, check:

- `validate_session.py` reports a valid VRS file.
- `convert_session_to_hdf5.py` completes without errors.
- `quality_check.py` produces a JSON report.
- `view_rerun.py session_YYYYMMDD_HHMMSS.h5` opens the converted dataset.
- `sample_dataset/load_sample.py session_YYYYMMDD_HHMMSS.h5` prints one frame
  across the expected streams.


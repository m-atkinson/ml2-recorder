# Contributing

Thanks for considering a contribution. This project welcomes bug reports,
pull requests, and notes on porting to other ML2 hardware configurations.

## Dev loop

```bash
# Recorder side (native Android)
export MLSDK=$HOME/MagicLeap/mlsdk/v1.12.0
export JAVA_HOME=$(brew --prefix openjdk)   # or your JDK path
./gradlew assembleDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk

# End-to-end test
./tools/quicktest.sh 15 --profile full

# Host-side unit tests
cmake -S tests -B tests/build
cmake --build tests/build
ctest --test-dir tests/build

# Python tools tests
pytest tools/
```

## Style

- **C++**: C++17, native CAPI (no STL exceptions in hot paths). New files
  should match the naming and include-ordering of existing files.
- **Python**: run `ruff format` and `ruff check` before submitting. Type
  hints welcomed but not required.
- **Commit messages**: short imperative subject (< 70 chars), blank line,
  body that explains *why* the change is needed and what the tradeoffs are.
  Reference the file:line locations you touched when the change is
  non-obvious.

## Pull requests

1. Fork, branch, work in a focused branch (one concern per PR).
2. Include a short test plan in the PR description, and — when applicable —
   a before/after recording or converter output showing the change.
3. For recorder-side changes: build and smoke-test on a real ML2 before
   submitting. CI doesn't have access to ML2 hardware.
4. For data-format changes: update both the converter and
   `sample_dataset/load_sample.py`, and bump `kDataVersion` /
   `kConfigVersion` in `app/src/main/cpp/vrs_writer.cpp` if the VRS
   schema changes.

## Reporting issues

Open an issue with:
- ML2 firmware and MLSDK version
- A minimal reproduction (session file excerpt, command line used)
- Expected vs actual behaviour
- Any relevant `logcat` output from `adb logcat -s ML2Recorder:*`

Security issues: see [`SECURITY.md`](SECURITY.md).

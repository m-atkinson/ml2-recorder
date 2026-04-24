#!/usr/bin/env bash
# =============================================================================
# quicktest.sh — One-command workflow: install → record → pull → convert → QC
#
# Usage:
#   ./tools/quicktest.sh [duration_s] [--profile full|high_temporal|lightweight]
#                        [--apk <path>] [--skip-install] [--min-score N]
#                        [--no-convert] [--no-quality]
#
# Defaults: 30 s, profile=full, min-score=60
#
# Examples:
#   ./tools/quicktest.sh                   # 30s full-quality session, full pipeline
#   ./tools/quicktest.sh 10 --min-score 50 # quick 10s check, lower pass bar
#   ./tools/quicktest.sh --skip-install    # device already has latest build
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── Colour helpers ────────────────────────────────────────────────────────────
if [ -t 1 ]; then
    RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
    CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'
else
    RED='' GREEN='' YELLOW='' CYAN='' BOLD='' RESET=''
fi

pass()    { printf "${GREEN}${BOLD}[PASS]${RESET} %s\n" "$*"; }
fail()    { printf "${RED}${BOLD}[FAIL]${RESET} %s\n" "$*"; }
warn()    { printf "${YELLOW}${BOLD}[WARN]${RESET} %s\n" "$*"; }
info()    { printf "${CYAN}${BOLD}[INFO]${RESET} %s\n" "$*"; }
section() { printf "\n${BOLD}══ %s ══${RESET}\n" "$*"; }

# ── Defaults ──────────────────────────────────────────────────────────────────
DURATION=30
PROFILE="full"
APK_PATH="${PROJECT_ROOT}/app/build/outputs/apk/debug/app-debug.apk"
SKIP_INSTALL=0
MIN_SCORE=60
NO_CONVERT=0
NO_QUALITY=0

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --profile)       PROFILE="$2";    shift 2 ;;
        --apk)           APK_PATH="$2";   shift 2 ;;
        --min-score)     MIN_SCORE="$2";  shift 2 ;;
        --skip-install)  SKIP_INSTALL=1;  shift   ;;
        --no-convert)    NO_CONVERT=1;    shift   ;;
        --no-quality)    NO_QUALITY=1;    shift   ;;
        --help|-h)
            sed -n '2,12p' "$0" | sed 's/^# \?//'
            exit 0 ;;
        -*) fail "Unknown flag: $1"; exit 1 ;;
        *)  DURATION="$1"; shift ;;
    esac
done

if ! [[ "$DURATION" =~ ^[0-9]+$ ]] || [[ "$DURATION" -eq 0 ]]; then
    fail "Duration must be a positive integer (got: $DURATION)"; exit 1
fi

PACKAGE="com.ml2.recorder"
DEVICE_BASE="/storage/emulated/0/Android/data/${PACKAGE}/files/recordings"
PULLED_DIR="${PROJECT_ROOT}/pulled_sessions"
OVERALL_FAILED=0
START_TIME="$(date +%s)"

elapsed() { echo $(( $(date +%s) - START_TIME )); }

# ── Cleanup trap ──────────────────────────────────────────────────────────────
LOGCAT_PID=""
LOGCAT_TMP=""
cleanup() {
    [[ -n "$LOGCAT_PID" ]] && kill "$LOGCAT_PID" 2>/dev/null || true
    [[ -n "$LOGCAT_TMP" && -f "$LOGCAT_TMP" ]] && rm -f "$LOGCAT_TMP"
}
trap cleanup EXIT

# =============================================================================
section "Pre-flight"
# =============================================================================

if ! command -v adb &>/dev/null; then
    fail "adb not in PATH. Install Android SDK platform-tools."; exit 1
fi
pass "adb: $(command -v adb)"

DEVICE_COUNT="$(adb devices 2>/dev/null | grep -cE '\bdevice$' || true)"
if [[ "$DEVICE_COUNT" -eq 0 ]]; then
    fail "No device detected — connect ML2 and enable USB debugging."; exit 1
fi
pass "Device connected (${DEVICE_COUNT} device(s))"

if [[ "$SKIP_INSTALL" -eq 0 && ! -f "$APK_PATH" ]]; then
    fail "APK not found: $APK_PATH"
    fail "Build first:  ./gradlew assembleDebug  (or pass --apk <path> or --skip-install)"
    exit 1
fi

info "Duration=${DURATION}s  Profile=${PROFILE}  MinScore=${MIN_SCORE}"

# =============================================================================
if [[ "$SKIP_INSTALL" -eq 0 ]]; then
section "Install"
# =============================================================================
    info "Installing $APK_PATH ..."
    if ! adb install -r "$APK_PATH" 2>&1; then
        fail "APK install failed."; exit 1
    fi
    pass "APK installed"

    info "Granting permissions..."
    for perm in android.permission.CAMERA \
                android.permission.RECORD_AUDIO \
                android.permission.READ_EXTERNAL_STORAGE \
                android.permission.WRITE_EXTERNAL_STORAGE; do
        adb shell pm grant "$PACKAGE" "$perm" 2>/dev/null || true
    done
    for perm in com.magicleap.permission.EYE_TRACKING \
                com.magicleap.permission.HAND_TRACKING \
                com.magicleap.permission.DEPTH_CAMERA; do
        adb shell pm grant "$PACKAGE" "$perm" 2>/dev/null || true
    done
    pass "Permissions granted"
fi

# =============================================================================
section "Record"
# =============================================================================

LOGCAT_TMP="$(mktemp /tmp/ml2_quicktest_XXXXXX.log)"
adb logcat -c 2>/dev/null || true
adb logcat -s "ML2Recorder:*" "*:E" > "$LOGCAT_TMP" 2>&1 &
LOGCAT_PID=$!

info "Launching recorder (profile=$PROFILE, ${DURATION}s)..."
if ! adb shell am start \
        -n "${PACKAGE}/.RecorderActivity" \
        --ei duration "$DURATION" \
        --es profile "$PROFILE" 2>&1; then
    fail "Failed to start activity."; exit 1
fi
pass "Activity launched"

WAIT_TOTAL=$(( DURATION + 15 ))
info "Waiting up to ${WAIT_TOTAL}s for completion..."
WAITED=0
DONE=0
while [[ $WAITED -lt $WAIT_TOTAL ]]; do
    if grep -qi "Recording complete\|SESSION_COMPLETE" "$LOGCAT_TMP" 2>/dev/null; then
        DONE=1; info "Completion detected at ${WAITED}s."; break
    fi
    sleep 1
    WAITED=$(( WAITED + 1 ))
    (( WAITED % 10 == 0 )) && info "  ${WAITED}s / ${WAIT_TOTAL}s"
done
[[ $DONE -eq 0 ]] && warn "No completion marker; continuing after timeout."

# Stop logcat
kill "$LOGCAT_PID" 2>/dev/null || true; wait "$LOGCAT_PID" 2>/dev/null || true
LOGCAT_PID=""

ERRORS="$(grep -E "FATAL|SIGABRT|SIGSEGV|MLResult_.*[Ee]rror|RuntimeException|ANR" \
          "$LOGCAT_TMP" 2>/dev/null || true)"
if [[ -n "$ERRORS" ]]; then
    fail "Critical errors in logcat:"; echo "$ERRORS"; OVERALL_FAILED=1
else
    pass "No critical errors in logcat"
fi

# =============================================================================
section "Pull"
# =============================================================================

info "Finding latest session on device..."
LATEST="$(adb shell ls -t "$DEVICE_BASE/" 2>/dev/null | tr -d '\r' | grep '\.vrs$' | head -1 | tr -d '\r\n')"
if [[ -z "$LATEST" ]]; then
    fail "No .vrs files under $DEVICE_BASE/"; exit 1
fi
info "Latest: $LATEST"

DEVICE_VRS="${DEVICE_BASE}/${LATEST}"
LOCAL_VRS="${PULLED_DIR}/${LATEST}"
mkdir -p "$PULLED_DIR"

info "Pulling to ${LOCAL_VRS} ..."
if ! adb pull "$DEVICE_VRS" "$LOCAL_VRS"; then
    fail "adb pull failed."; exit 1
fi
[[ ! -f "$LOCAL_VRS" ]] && { fail "File not found after pull."; exit 1; }

SIZE_MB="$(du -sh "$LOCAL_VRS" | awk '{print $1}')"
pass "Pulled (${SIZE_MB})"

# =============================================================================
section "Validate (binary)"
# =============================================================================

VALIDATE="${SCRIPT_DIR}/validate_session.py"
if [[ -f "$VALIDATE" ]]; then
    if python3 "$VALIDATE" "$LOCAL_VRS"; then
        pass "Binary validation passed"
    else
        warn "Binary validation had issues (see above)"
        OVERALL_FAILED=1
    fi
else
    warn "validate_session.py not found — skipping"
fi

# =============================================================================
if [[ "$NO_CONVERT" -eq 0 ]]; then
section "Convert to HDF5"
# =============================================================================
    CONVERT="${SCRIPT_DIR}/convert_session_to_hdf5.py"
    LOCAL_H5="${LOCAL_VRS%.vrs}.h5"

    if [[ -f "$LOCAL_H5" ]]; then
        info "HDF5 already exists: $LOCAL_H5 — skipping conversion"
    elif [[ -f "$CONVERT" ]]; then
        info "Converting $LATEST → HDF5 ..."
        if python3 "$CONVERT" "$LOCAL_VRS"; then
            pass "Conversion complete: $LOCAL_H5"
        else
            fail "HDF5 conversion failed."; OVERALL_FAILED=1; NO_QUALITY=1
        fi
    else
        warn "convert_session_to_hdf5.py not found — skipping"
        NO_QUALITY=1
    fi
fi

# =============================================================================
if [[ "$NO_QUALITY" -eq 0 ]]; then
section "Quality Check"
# =============================================================================
    LOCAL_H5="${LOCAL_VRS%.vrs}.h5"
    QC="${SCRIPT_DIR}/quality_check.py"
    QC_JSON="${LOCAL_VRS%.vrs}_quality.json"

    if [[ ! -f "$LOCAL_H5" ]]; then
        warn "HDF5 not found — skipping quality check"; OVERALL_FAILED=1
    elif [[ -f "$QC" ]]; then
        if python3 "$QC" "$LOCAL_H5" --min-score "$MIN_SCORE" --json "$QC_JSON"; then
            pass "Quality check PASSED (score >= $MIN_SCORE)"
        else
            fail "Quality check FAILED (score < $MIN_SCORE)"
            OVERALL_FAILED=1
        fi
        info "JSON report: $QC_JSON"
    else
        warn "quality_check.py not found — skipping"
    fi
fi

# =============================================================================
section "Summary"
# =============================================================================
ELAPSED="$(elapsed)"
echo ""
info "Profile:       $PROFILE"
info "Duration:      ${DURATION}s"
info "Device VRS:    $DEVICE_VRS"
info "Local VRS:     $LOCAL_VRS"
[[ "$NO_CONVERT" -eq 0 && -f "${LOCAL_VRS%.vrs}.h5" ]] && \
    info "Local HDF5:    ${LOCAL_VRS%.vrs}.h5"
[[ "$NO_QUALITY" -eq 0 && -f "${LOCAL_VRS%.vrs}_quality.json" ]] && \
    info "Quality JSON:  ${LOCAL_VRS%.vrs}_quality.json"
info "Elapsed:       ${ELAPSED}s"
echo ""

if [[ "$OVERALL_FAILED" -eq 0 ]]; then
    pass "ALL STEPS PASSED"
else
    fail "ONE OR MORE STEPS FAILED — see above"
fi

exit "$OVERALL_FAILED"

#!/usr/bin/env bash
set -euo pipefail

# --------------------------------------------------------------------------
# test_on_device.sh — automated on-device test cycle for ML2 multi-sensor
# recorder.  Wraps install, record, pull, and validate into one command.
#
# Usage:
#   ./test_on_device.sh [duration_seconds] [--profile full|high_temporal|lightweight] [--apk <path>]
#
# Defaults: 30 s duration, "full" profile.
# --------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── Colour helpers (degrade gracefully when not a TTY) ────────────────────
if [ -t 1 ]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    BOLD='\033[1m'
    RESET='\033[0m'
else
    RED='' GREEN='' YELLOW='' BOLD='' RESET=''
fi

pass()  { printf "${GREEN}${BOLD}[PASS]${RESET} %s\n" "$*"; }
fail()  { printf "${RED}${BOLD}[FAIL]${RESET} %s\n" "$*"; }
warn()  { printf "${YELLOW}${BOLD}[WARN]${RESET} %s\n" "$*"; }
info()  { printf "${BOLD}[INFO]${RESET} %s\n" "$*"; }

# ── Defaults ──────────────────────────────────────────────────────────────
DURATION=30
PROFILE="full"
APK_PATH="${PROJECT_ROOT}/app/build/outputs/apk/debug/app-debug.apk"

# ── Argument parsing ─────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --profile)
            PROFILE="$2"; shift 2 ;;
        --apk)
            APK_PATH="$2"; shift 2 ;;
        --help|-h)
            echo "Usage: $0 [duration_seconds] [--profile full|high_temporal|lightweight] [--apk <path>]"
            exit 0 ;;
        -*)
            fail "Unknown flag: $1"; exit 1 ;;
        *)
            # Positional: treat as duration
            DURATION="$1"; shift ;;
    esac
done

# Validate duration is a positive integer
if ! [[ "$DURATION" =~ ^[0-9]+$ ]] || [[ "$DURATION" -eq 0 ]]; then
    fail "Duration must be a positive integer (got: $DURATION)"
    exit 1
fi

# Validate profile
case "$PROFILE" in
    full|high_temporal|lightweight) ;;
    *) fail "Unknown profile: $PROFILE (expected full, high_temporal, or lightweight)"; exit 1 ;;
esac

# ── Cleanup trap ─────────────────────────────────────────────────────────
LOGCAT_PID=""
LOGCAT_TMP=""
TEST_START_TIME="$(date +%s)"
TEST_FAILED=0

cleanup() {
    if [[ -n "$LOGCAT_PID" ]] && kill -0 "$LOGCAT_PID" 2>/dev/null; then
        kill "$LOGCAT_PID" 2>/dev/null || true
        wait "$LOGCAT_PID" 2>/dev/null || true
    fi
    if [[ -n "$LOGCAT_TMP" ]] && [[ -f "$LOGCAT_TMP" ]]; then
        rm -f "$LOGCAT_TMP"
    fi
}
trap cleanup EXIT

# ── Helper: elapsed time ─────────────────────────────────────────────────
elapsed() {
    local now
    now="$(date +%s)"
    echo $(( now - TEST_START_TIME ))
}

# ══════════════════════════════════════════════════════════════════════════
# 1. Pre-flight checks
# ══════════════════════════════════════════════════════════════════════════
info "Pre-flight checks"

# 1a. adb in PATH
if ! command -v adb &>/dev/null; then
    fail "adb not found in PATH. Install Android SDK platform-tools."
    exit 1
fi
pass "adb found: $(command -v adb)"

# 1b. Device connected
DEVICE_COUNT="$(adb devices 2>/dev/null | grep -cE '\bdevice$' || true)"
if [[ "$DEVICE_COUNT" -eq 0 ]]; then
    fail "No device detected. Connect the ML2 and enable USB debugging."
    exit 1
fi
if [[ "$DEVICE_COUNT" -gt 1 ]]; then
    warn "Multiple devices detected ($DEVICE_COUNT). Using the default adb device."
fi
pass "Device connected ($DEVICE_COUNT device(s))"

# 1c. APK exists
if [[ ! -f "$APK_PATH" ]]; then
    fail "APK not found at: $APK_PATH"
    fail "Build the project first, or pass --apk <path>."
    exit 1
fi
pass "APK found: $APK_PATH"

info "Config: duration=${DURATION}s  profile=${PROFILE}"
echo ""

# ══════════════════════════════════════════════════════════════════════════
# 2. Install
# ══════════════════════════════════════════════════════════════════════════
info "Installing APK..."
if ! adb install -r "$APK_PATH" 2>&1; then
    fail "APK installation failed."
    exit 1
fi
pass "APK installed"
echo ""

# ══════════════════════════════════════════════════════════════════════════
# 2b. Grant permissions
# ══════════════════════════════════════════════════════════════════════════
info "Granting runtime permissions..."
for perm in android.permission.CAMERA \
            android.permission.RECORD_AUDIO \
            android.permission.READ_EXTERNAL_STORAGE \
            android.permission.WRITE_EXTERNAL_STORAGE; do
    adb shell pm grant com.ml2.recorder "$perm" 2>/dev/null || true
done
for perm in com.magicleap.permission.EYE_TRACKING \
            com.magicleap.permission.HAND_TRACKING \
            com.magicleap.permission.DEPTH_CAMERA; do
    adb shell pm grant com.ml2.recorder "$perm" 2>/dev/null || true
done
pass "Permissions granted"
echo ""

# ══════════════════════════════════════════════════════════════════════════
# 3. Start logcat monitor (background)
# ══════════════════════════════════════════════════════════════════════════
LOGCAT_TMP="$(mktemp /tmp/ml2_logcat_XXXXXX.log)"

# Clear old logcat buffer so we only capture this session
adb logcat -c 2>/dev/null || true

adb logcat -s "ML2Recorder:*" "*:E" > "$LOGCAT_TMP" 2>&1 &
LOGCAT_PID=$!
info "Logcat monitor started (PID $LOGCAT_PID, writing to $LOGCAT_TMP)"
echo ""

# ══════════════════════════════════════════════════════════════════════════
# 4. Start recording
# ══════════════════════════════════════════════════════════════════════════
info "Launching recorder (profile=$PROFILE, duration=${DURATION}s)..."
if ! adb shell am start \
        -n com.ml2.recorder/.RecorderActivity \
        --ei duration "$DURATION" \
        --es profile "$PROFILE" 2>&1; then
    fail "Failed to start NativeActivity."
    exit 1
fi
pass "NativeActivity launched"
echo ""

# ══════════════════════════════════════════════════════════════════════════
# 5. Wait for recording to finish
# ══════════════════════════════════════════════════════════════════════════
WAIT_TOTAL=$(( DURATION + 10 ))  # grace period for startup/shutdown
info "Waiting up to ${WAIT_TOTAL}s for recording to complete..."

SECONDS_WAITED=0
RECORDING_DONE=0
while [[ $SECONDS_WAITED -lt $WAIT_TOTAL ]]; do
    # Check for a "recording complete" marker in logcat
    if grep -qi "Recording complete\|SESSION_COMPLETE" "$LOGCAT_TMP" 2>/dev/null; then
        RECORDING_DONE=1
        info "Detected recording-complete marker in logcat after ${SECONDS_WAITED}s."
        break
    fi
    sleep 1
    SECONDS_WAITED=$(( SECONDS_WAITED + 1 ))

    # Progress every 10 seconds
    if (( SECONDS_WAITED % 10 == 0 )); then
        info "  ... ${SECONDS_WAITED}s / ${WAIT_TOTAL}s"
    fi
done

if [[ $RECORDING_DONE -eq 0 ]]; then
    warn "No recording-complete marker detected; proceeding after timeout."
fi
echo ""

# ══════════════════════════════════════════════════════════════════════════
# 6. Scan logcat for errors
# ══════════════════════════════════════════════════════════════════════════
info "Scanning logcat for errors..."

# Stop logcat collection
if kill -0 "$LOGCAT_PID" 2>/dev/null; then
    kill "$LOGCAT_PID" 2>/dev/null || true
    wait "$LOGCAT_PID" 2>/dev/null || true
fi
LOGCAT_PID=""

ERROR_PATTERNS="FATAL|SIGABRT|SIGSEGV|MLResult_.*[Ee]rror|java\.lang\.RuntimeException|ANR"
ERRORS_FOUND=""
if [[ -s "$LOGCAT_TMP" ]]; then
    ERRORS_FOUND="$(grep -E "$ERROR_PATTERNS" "$LOGCAT_TMP" || true)"
fi

if [[ -n "$ERRORS_FOUND" ]]; then
    fail "Critical errors detected in logcat:"
    echo "--------"
    echo "$ERRORS_FOUND"
    echo "--------"
    TEST_FAILED=1
else
    pass "No critical errors in logcat"
fi
echo ""

# ══════════════════════════════════════════════════════════════════════════
# 7. Pull session
# ══════════════════════════════════════════════════════════════════════════
info "Finding latest session on device..."

# Sessions are a single .vrs file in the app-specific external files dir.
DEVICE_BASE="/storage/emulated/0/Android/data/com.ml2.recorder/files/recordings"

LATEST_SESSION="$(adb shell ls -t "$DEVICE_BASE/" 2>/dev/null | tr -d '\r' | grep '\.vrs$' | head -1 | tr -d '\r\n')"
if [[ -z "$LATEST_SESSION" ]]; then
    fail "No .vrs session files found under $DEVICE_BASE/"
    exit 1
fi
info "Latest session: $LATEST_SESSION"

DEVICE_SESSION_PATH="${DEVICE_BASE}/${LATEST_SESSION}"
LOCAL_VRS="${PROJECT_ROOT}/pulled_sessions/${LATEST_SESSION}"
mkdir -p "${PROJECT_ROOT}/pulled_sessions"

info "Pulling session to ${LOCAL_VRS} ..."
if ! adb pull "$DEVICE_SESSION_PATH" "$LOCAL_VRS" 2>&1; then
    fail "adb pull failed for session $LATEST_SESSION"
    exit 1
fi

# Verify we got the file
if [[ ! -f "$LOCAL_VRS" ]]; then
    fail "Pulled file not found: $LOCAL_VRS"
    exit 1
fi
SIZE_MB="$(du -sh "$LOCAL_VRS" 2>/dev/null | awk '{print $1}' || echo "?")"
pass "Session pulled (${SIZE_MB})"
echo ""

# ══════════════════════════════════════════════════════════════════════════
# 8. Validate session
# ══════════════════════════════════════════════════════════════════════════
VALIDATE_SCRIPT="${PROJECT_ROOT}/tools/validate_session.py"
if [[ -f "$VALIDATE_SCRIPT" ]]; then
    info "Running validate_session.py ..."
    if python3 "$VALIDATE_SCRIPT" "$LOCAL_VRS"; then
        pass "Session validation passed"
    else
        fail "Session validation failed (exit code $?)"
        TEST_FAILED=1
    fi
else
    warn "Validation script not found at $VALIDATE_SCRIPT — skipping validation."
fi
echo ""

# ══════════════════════════════════════════════════════════════════════════
# 9. Summary
# ══════════════════════════════════════════════════════════════════════════
TOTAL_ELAPSED="$(elapsed)"

echo "======================================================================"
if [[ $TEST_FAILED -eq 0 ]]; then
    pass "ALL CHECKS PASSED"
else
    fail "TEST FAILED — see errors above"
fi
echo "======================================================================"
echo ""
info "Profile:          $PROFILE"
info "Duration:         ${DURATION}s"
info "Device session:   $DEVICE_SESSION_PATH"
info "Local session:    $LOCAL_VRS"
info "Total test time:  ${TOTAL_ELAPSED}s"
echo ""

exit $TEST_FAILED

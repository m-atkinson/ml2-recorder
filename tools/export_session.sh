#!/usr/bin/env bash
set -euo pipefail

# --------------------------------------------------------------------------
# export_session.sh — Pull ML2 session .vrs file(s) from device and
# optionally convert to HDF5.
#
# Usage:
#   ./export_session.sh                  # Pull latest session
#   ./export_session.sh --all            # Pull all sessions
#   ./export_session.sh --list           # List sessions on device
#   ./export_session.sh --convert        # Pull + convert to HDF5
#   ./export_session.sh --session <name> # Pull a specific session (e.g. session_20260101_120000)
# --------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Colour helpers
if [ -t 1 ]; then
    GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[1;33m'
    BOLD='\033[1m'; RESET='\033[0m'
else
    GREEN='' RED='' YELLOW='' BOLD='' RESET=''
fi

pass()  { printf "${GREEN}${BOLD}[PASS]${RESET} %s\n" "$*"; }
fail()  { printf "${RED}${BOLD}[FAIL]${RESET} %s\n" "$*"; }
info()  { printf "${BOLD}[INFO]${RESET} %s\n" "$*"; }
warn()  { printf "${YELLOW}${BOLD}[WARN]${RESET} %s\n" "$*"; }

# VRS files live in the app-specific external files directory on the device
DEVICE_BASE="/storage/emulated/0/Android/data/com.ml2.recorder/files/recordings"
LOCAL_BASE="${PROJECT_ROOT}/pulled_sessions"
CONVERT=0
PULL_ALL=0
LIST_ONLY=0
SESSION_NAME=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --convert)   CONVERT=1; shift ;;
        --all)       PULL_ALL=1; shift ;;
        --list)      LIST_ONLY=1; shift ;;
        --session)   SESSION_NAME="$2"; shift 2 ;;
        --help|-h)
            echo "Usage: $0 [--list] [--all] [--convert] [--session <name>]"
            echo ""
            echo "  --list      List .vrs session files on device"
            echo "  --all       Pull all sessions"
            echo "  --session   Pull a specific session by base name (without .vrs)"
            echo "  --convert   Also convert to HDF5 after pulling"
            echo ""
            echo "Default: pull the latest session"
            exit 0 ;;
        *) fail "Unknown argument: $1"; exit 1 ;;
    esac
done

# Check adb
if ! command -v adb &>/dev/null; then
    fail "adb not found in PATH"
    exit 1
fi

# Check device
if ! adb devices 2>/dev/null | grep -qE '\bdevice$'; then
    fail "No device connected"
    exit 1
fi

# ══════════════════════════════════════════════════════════════════════════
# List .vrs files on device (sorted newest first by name)
# ══════════════════════════════════════════════════════════════════════════
# ls -t gives modification-time order; we grep for .vrs to ignore other files
VRS_FILES="$(adb shell ls -t "${DEVICE_BASE}/" 2>/dev/null | tr -d '\r' | grep '\.vrs$' || true)"

if [[ -z "$VRS_FILES" ]]; then
    fail "No .vrs session files found on device at ${DEVICE_BASE}/"
    exit 1
fi

if [[ $LIST_ONLY -eq 1 ]]; then
    info "Sessions on device (${DEVICE_BASE}/):"
    echo "$VRS_FILES" | while read -r f; do
        SIZE="$(adb shell du -sh "${DEVICE_BASE}/${f}" 2>/dev/null | awk '{print $1}' || echo "?")"
        echo "  ${f}  (${SIZE})"
    done
    exit 0
fi

# ══════════════════════════════════════════════════════════════════════════
# Determine which sessions to pull
# ══════════════════════════════════════════════════════════════════════════
FILES_TO_PULL=()

if [[ -n "$SESSION_NAME" ]]; then
    # Accept with or without .vrs extension
    TARGET="${SESSION_NAME%.vrs}.vrs"
    if echo "$VRS_FILES" | grep -qx "$TARGET"; then
        FILES_TO_PULL+=("$TARGET")
    else
        fail "Session not found: $TARGET"
        info "Available sessions:"
        echo "$VRS_FILES"
        exit 1
    fi
elif [[ $PULL_ALL -eq 1 ]]; then
    while IFS= read -r f; do
        [[ -n "$f" ]] && FILES_TO_PULL+=("$f")
    done <<< "$VRS_FILES"
else
    # Latest only
    LATEST="$(echo "$VRS_FILES" | head -1)"
    FILES_TO_PULL+=("$LATEST")
fi

info "Will pull ${#FILES_TO_PULL[@]} session(s)"
echo ""

# ══════════════════════════════════════════════════════════════════════════
# Pull sessions
# ══════════════════════════════════════════════════════════════════════════
mkdir -p "$LOCAL_BASE"

for VRS_FILE in "${FILES_TO_PULL[@]}"; do
    LOCAL_VRS="${LOCAL_BASE}/${VRS_FILE}"
    DEVICE_VRS="${DEVICE_BASE}/${VRS_FILE}"

    if [[ -f "$LOCAL_VRS" ]]; then
        warn "Already exists locally: $LOCAL_VRS — skipping pull"
    else
        info "Pulling ${VRS_FILE}..."
        if adb pull "$DEVICE_VRS" "${LOCAL_BASE}/" 2>&1; then
            SIZE_MB="$(du -sh "$LOCAL_VRS" 2>/dev/null | awk '{print $1}' || echo "?")"
            pass "Pulled ${VRS_FILE} (${SIZE_MB})"
        else
            fail "Failed to pull ${VRS_FILE}"
            continue
        fi
    fi

    # Validate
    info "Validating ${VRS_FILE}..."
    if python3 "${SCRIPT_DIR}/validate_session.py" "$LOCAL_VRS" 2>&1 | tail -8; then
        echo ""
    fi

    # Convert to HDF5 if requested
    if [[ $CONVERT -eq 1 ]]; then
        H5_PATH="${LOCAL_BASE}/${VRS_FILE%.vrs}.h5"
        if [[ -f "$H5_PATH" ]]; then
            warn "HDF5 already exists: $H5_PATH — skipping conversion"
        else
            info "Converting to HDF5..."
            if python3 "${SCRIPT_DIR}/convert_session_to_hdf5.py" "$LOCAL_VRS" -o "$H5_PATH"; then
                pass "HDF5 created: $H5_PATH"
            else
                fail "HDF5 conversion failed for ${VRS_FILE}"
            fi
        fi
    fi
    echo ""
done

info "Done. Sessions in: $LOCAL_BASE/"

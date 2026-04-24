#!/usr/bin/env bash
set -euo pipefail

# Downloads third-party single-header dependencies into the C++ source tree.
# Run from the project root, or the script will cd there automatically.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CPP_DIR="${PROJECT_ROOT}/app/src/main/cpp"

echo "Downloading stb_image_write.h ..."
curl -sL \
    "https://raw.githubusercontent.com/nothings/stb/master/stb_image_write.h" \
    -o "${CPP_DIR}/stb_image_write.h"
echo "  -> ${CPP_DIR}/stb_image_write.h"

echo "Done."

#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

if [ $# -lt 1 ]; then
    echo "Usage: $0 <path_to_pdf> [options]"
    exit 1
fi

python3 "$SCRIPT_DIR/fast_split_and_process.py" "$@"

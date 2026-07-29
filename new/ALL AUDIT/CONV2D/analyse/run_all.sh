#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export MPLBACKEND=Agg
python3 "$HERE/01_preflight_all_conv2d.py"
python3 "$HERE/02_compare_all_platforms.py"

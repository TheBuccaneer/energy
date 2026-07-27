#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export MPLBACKEND=Agg
python3 "$HERE/01_validate_reduction.py" "$@"
python3 "$HERE/02_analyze_reduction.py" "$@"
echo "REDUCTION platform analysis complete: $(cd "$HERE/../../results/REDUCTION" && pwd)"

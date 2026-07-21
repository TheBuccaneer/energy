#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export MPLBACKEND=Agg
python3 "$HERE/01_validate_stream.py" "$@"
python3 "$HERE/02_analyze_stream.py" "$@"
echo "STREAM platform analysis complete: $(cd "$HERE/../../results/STREAM" && pwd)"

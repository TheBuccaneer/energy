#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
python3 "$HERE/01_validate_strided_gemm.py" "$@"
python3 "$HERE/02_analyze_strided_gemm.py" "$@"
echo "Local STRIDED_GEMM audit complete."
echo "Results: $(cd "$HERE/../.." && pwd)/results/STRIDED_GEMM"

#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
python3 "$HERE/01_validate_gemm_3090.py" "$@"
python3 "$HERE/02_analyze_gemm_3090.py" "$@"
echo "RTX 3090 GEMM audit complete."
echo "Results: $(cd "$HERE/../.." && pwd)/results/GEMM"

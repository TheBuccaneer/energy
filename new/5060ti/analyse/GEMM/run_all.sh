#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
python3 "$HERE/01_validate_gemm_5060ti.py" "$@"
python3 "$HERE/02_analyze_gemm_5060ti.py" "$@"
echo "RTX 5060 Ti GEMM audit complete."
echo "Results: $(cd "$HERE/../.." && pwd)/results/GEMM"

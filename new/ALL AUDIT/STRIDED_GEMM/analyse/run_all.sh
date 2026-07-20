#!/usr/bin/env bash
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
export MPLBACKEND=Agg

python3 "$HERE/01_preflight_all_strided.py" "$@"
python3 "$HERE/02_build_unified_stats.py" "$@"
python3 "$HERE/03_compare_all_platforms.py" "$@"
python3 "$HERE/04_compare_dense_vs_strided.py" "$@"
python3 "$HERE/05_generate_reports.py" "$@"
python3 "$HERE/06_integrity_audit.py" "$@"

echo "All-platform STRIDED_GEMM audit complete."
echo "Results: $(cd "$HERE/.." && pwd)/results"

#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export MPLBACKEND=Agg
python3 "$HERE/01_preflight_all_stream.py"
python3 "$HERE/02_build_unified_stats.py"
python3 "$HERE/03_compare_all_platforms.py"
python3 "$HERE/04_generate_reports.py"
python3 "$HERE/05_integrity_audit.py"
echo "All-platform STREAM audit complete: $(cd "$HERE/../results" && pwd)"

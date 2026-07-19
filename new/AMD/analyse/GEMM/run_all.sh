#!/usr/bin/env bash
set -uo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../../.." && pwd)

intel_ok=0
amd_ok=0

echo "=== Validate Intel ==="
if python3 "$ROOT/INTEL/analyse/GEMM/01_validate_gemm.py"; then
    intel_ok=1
    python3 "$ROOT/INTEL/analyse/GEMM/02_analyze_gemm.py"
else
    echo "Intel validation failed; Intel analysis skipped." >&2
fi

echo "=== Validate AMD ==="
if python3 "$ROOT/AMD/analyse/GEMM/01_validate_gemm.py"; then
    amd_ok=1
    python3 "$ROOT/AMD/analyse/GEMM/02_analyze_gemm.py"
else
    echo "AMD validation failed; AMD analysis skipped." >&2
fi

if [[ "$intel_ok" -eq 1 && "$amd_ok" -eq 1 ]]; then
    python3 "$HERE/03_compare_gemm.py"
    echo "GEMM analysis complete for Intel and AMD."
    echo "Results: $ROOT/INTEL/results/GEMM and $ROOT/AMD/results/GEMM"
    exit 0
fi

echo "Cross-platform comparison skipped because at least one validation failed." >&2
exit 2

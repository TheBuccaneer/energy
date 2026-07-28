#!/usr/bin/env bash
set -euo pipefail

if (( EUID == 0 )); then
    echo "ERROR: do not start this quickcheck with sudo." >&2
    echo "Run 01_enable with sudo, then start this quickcheck as your normal user." >&2
    exit 2
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
RUNNER="$SCRIPT_DIR/02_run_CPU_Intel_CONV2D_only.sh"

if [[ ! -f "$RUNNER" ]]; then
    echo "ERROR: missing CONV2D runner: $RUNNER" >&2
    exit 2
fi

echo "Intel CONV2D quickcheck"
echo "Details are written to new/INTEL/runs/CONV2D/."
echo
exec env \
    QUICKCHECK_ONLY=1 \
    POWER_OFF_AT_END=0 \
    REPS=10 \
    SESSIONS=5 \
    QUICK_REPS=2 \
    bash "$RUNNER"

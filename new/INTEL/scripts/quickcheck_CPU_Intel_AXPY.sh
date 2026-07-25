#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
RUNNER="$SCRIPT_DIR/02_run_CPU_Intel_AXPY_only.sh"

if [[ ! -f "$RUNNER" ]]; then
    echo "ERROR: missing AXPY runner: $RUNNER" >&2
    exit 2
fi

echo "Intel AXPY source-only quickcheck"
echo
echo "Prerequisite:"
echo "  sudo bash \"$SCRIPT_DIR/01_enable_CPU_Intel.sh\""
echo
echo "This quickcheck:"
echo "  - compiles the real Intel AXPY source"
echo "  - runs N={1M,64M,256M}"
echo "  - runs threads={1,20}"
echo "  - uses 2 repetitions"
echo "  - validates 12 CSV rows"
echo "  - validates exact zero checksum errors"
echo "  - validates the anti-collapse B/2B scaling gate"
echo "  - never starts official sessions"
echo "  - never powers off"
echo

exec env \
    QUICKCHECK_ONLY=1 \
    POWER_OFF_AT_END=0 \
    REPS=10 \
    SESSIONS=5 \
    QUICK_REPS=2 \
    bash "$RUNNER"

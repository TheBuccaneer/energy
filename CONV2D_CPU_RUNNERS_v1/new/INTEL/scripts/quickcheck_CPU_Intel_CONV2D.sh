#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
RUNNER="$SCRIPT_DIR/02_run_CPU_Intel_CONV2D_only.sh"

if [[ ! -f "$RUNNER" ]]; then
    echo "ERROR: missing CONV2D runner: $RUNNER" >&2
    exit 2
fi

echo "Intel CONV2D source-only hardware quickcheck"
echo
echo "Prerequisite:"
echo "  sudo bash \"$SCRIPT_DIR/01_enable_CPU_Intel.sh\""
echo
echo "This quickcheck:"
echo "  - compiles the real Intel CONV2D source with oneDNN/OpenMP"
echo "  - rejects TBB or mixed OpenMP runtimes"
echo "  - runs all six frozen Conv2D shapes"
echo "  - runs threads={1,20}"
echo "  - uses 2 repetitions (24 validated CSV rows)"
echo "  - enables oneDNN verbose and records selected implementations/layouts"
echo "  - validates FLOPs, logical bytes, device-domain energy and 32-sample checksums"
echo "  - runs the exclusive shape-1 B/2B anti-collapse gate"
echo "  - never starts official sessions"
echo "  - restores CPU settings and never powers off"
echo

exec env \
    QUICKCHECK_ONLY=1 \
    POWER_OFF_AT_END=0 \
    REPS=10 \
    SESSIONS=5 \
    QUICK_REPS=2 \
    bash "$RUNNER"

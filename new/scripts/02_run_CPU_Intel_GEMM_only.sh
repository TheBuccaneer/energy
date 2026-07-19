#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
COMMON="$ROOT/scripts/common"
STAMP=$(date +%Y%m%d_%H%M%S)
REPS=${REPS:-10}
SESSIONS=${SESSIONS:-5}
SESSION_PAUSE_SECONDS=${SESSION_PAUSE_SECONDS:-300}
POWER_OFF_AT_END=${POWER_OFF_AT_END:-1}

SRC="$ROOT/scripts/GEMM/CPU/INTEL/main_gemm.cpp"
BIN="$ROOT/scripts/GEMM/CPU/INTEL/main_gemm"
OUTDIR="$ROOT/runs/GEMM/CPU/INTEL"
RESTORE="$ROOT/scripts/03_disable_CPU_Intel.sh"

[[ -f "$SRC" ]] || { echo "ERROR: missing $SRC" >&2; exit 1; }
[[ -f "$RESTORE" ]] || { echo "ERROR: missing $RESTORE" >&2; exit 1; }

# Obtain sudo once and keep authorization valid for restore/poweroff.
sudo -v
(
    while true; do
        sudo -n true 2>/dev/null || exit
        sleep 60
    done
) &
SUDO_KEEPALIVE_PID=$!

SUCCESS=0
cleanup() {
    status=$?
    trap - EXIT

    kill "$SUDO_KEEPALIVE_PID" 2>/dev/null || true
    wait "$SUDO_KEEPALIVE_PID" 2>/dev/null || true

    echo "Restoring Intel CPU settings and RAPL permissions..."
    sudo bash "$RESTORE" || true

    if [[ "$SUCCESS" -eq 1 && "$POWER_OFF_AT_END" -eq 1 ]]; then
        echo "All GEMM sessions completed successfully. Powering off."
        sudo systemctl poweroff
    elif [[ "$SUCCESS" -eq 1 ]]; then
        echo "All GEMM sessions completed successfully. POWER_OFF_AT_END=0; system remains running."
    else
        echo "GEMM run stopped or failed. Settings restored; system will not power off." >&2
    fi

    exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

mkdir -p "$OUTDIR"

# Important: GEMM is intentionally built WITHOUT -fopenmp.
g++ -O3 -march=native -std=c++17 \
    -I"$COMMON" \
    "$SRC" \
    -lopenblas -lpthread -lm \
    -o "$BIN"

for ((session=1; session<=SESSIONS; session++)); do
    session_id="${STAMP}_session${session}"
    seed=$((20261000 + session))
    output="$OUTDIR/gemm_intel_${session_id}.csv"

    echo "=== GEMM | $session_id | seed=$seed | reps=$REPS ==="

    env -u OMP_PROC_BIND \
        -u OMP_PLACES \
        -u GOMP_CPU_AFFINITY \
        OMP_DYNAMIC=FALSE \
        stdbuf -oL -eL \
        "$BIN" "$output" "$REPS" "$session_id" "$seed"

    [[ -s "$output" ]] || { echo "ERROR: missing or empty output: $output" >&2; exit 1; }

    if [[ "$session" -lt "$SESSIONS" ]]; then
        echo "GEMM session $session/$SESSIONS complete. Cooling for $SESSION_PAUSE_SECONDS seconds."
        sleep "$SESSION_PAUSE_SECONDS"
    fi
done

echo "Completed $SESSIONS GEMM sessions × $REPS repetitions = $((SESSIONS * REPS)) measurements per N×thread configuration."
SUCCESS=1

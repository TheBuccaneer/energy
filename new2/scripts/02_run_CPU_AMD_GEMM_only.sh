#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
COMMON="$ROOT/scripts/common"
STAMP=$(date +%Y%m%d_%H%M%S)
REPS=${REPS:-10}
SESSIONS=${SESSIONS:-5}
POWER_OFF_AT_END=${POWER_OFF_AT_END:-1}

SRC="$ROOT/scripts/GEMM/CPU/AMD/main_gemm.cpp"
BIN="$ROOT/scripts/GEMM/CPU/AMD/main_gemm"
OUTDIR="$ROOT/runs2/GEMM/CPU/AMD"
RESTORE="$ROOT/scripts/03_disable_CPU_AMD.sh"

[[ -f "$SRC" ]] || { echo "ERROR: missing $SRC" >&2; exit 1; }
[[ -f "$RESTORE" ]] || { echo "ERROR: missing $RESTORE" >&2; exit 1; }

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

    echo "Restoring AMD CPU settings and RAPL permissions..."
    sudo bash "$RESTORE" || true

    if [[ "$SUCCESS" -eq 1 && "$POWER_OFF_AT_END" -eq 1 ]]; then
        echo "All AMD GEMM sessions completed successfully. Powering off."
        sudo systemctl poweroff
    elif [[ "$SUCCESS" -eq 1 ]]; then
        echo "All AMD GEMM sessions completed successfully. POWER_OFF_AT_END=0; system remains running."
    else
        echo "AMD GEMM run stopped or failed. Settings restored; system will not power off." >&2
    fi

    exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

mkdir -p "$OUTDIR"

# GEMM intentionally built WITHOUT -fopenmp.
g++ -O3 -march=native -std=c++17 \
    -I"$COMMON" \
    "$SRC" \
    -lopenblas -lpthread -lm \
    -o "$BIN"

echo "Linked libraries:"
ldd "$BIN" | grep -E 'openblas|gomp|omp|pthread' || true

for ((session=1; session<=SESSIONS; session++)); do
    session_id="${STAMP}_session${session}"
    seed=$((20262000 + session))
    output="$OUTDIR/gemm_amd_${session_id}.csv"

    echo "=== AMD GEMM | session $session/$SESSIONS | $session_id | seed=$seed | reps=$REPS ==="

    env -u OMP_PROC_BIND \
        -u OMP_PLACES \
        -u GOMP_CPU_AFFINITY \
        OMP_DYNAMIC=FALSE \
        stdbuf -oL -eL \
        "$BIN" "$output" "$REPS" "$session_id" "$seed"

    [[ -s "$output" ]] || { echo "ERROR: missing or empty output: $output" >&2; exit 1; }

    rows=$(( $(wc -l < "$output") - 1 ))
    expected=$((9 * 9 * REPS))
    if [[ "$rows" -ne "$expected" ]]; then
        echo "ERROR: $output has $rows data rows; expected $expected." >&2
        exit 1
    fi

    echo "AMD GEMM session $session/$SESSIONS complete: $rows rows."
done

echo "Completed $SESSIONS AMD GEMM sessions × $REPS repetitions with no pause between sessions."
echo "Expected measurements per N×thread configuration: $((SESSIONS * REPS))."
SUCCESS=1

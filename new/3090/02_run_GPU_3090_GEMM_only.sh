#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")" && pwd)
STAMP=$(date +%Y%m%d_%H%M%S)
REPS=${REPS:-10}
SESSIONS=${SESSIONS:-5}
GPU_INDEX=${GPU_INDEX:-0}
EXPECTED_GPU=${EXPECTED_GPU:-RTX 3090}
POWER_OFF_AT_END=${POWER_OFF_AT_END:-1}

SRC="$ROOT/scripts/GEMM/main_gemm.cu"
BIN="$ROOT/scripts/GEMM/main_gemm"
OUTDIR="$ROOT/runs/GEMM"
RESTORE="$ROOT/03_disable_GPU_3090.sh"

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

    sudo env GPU_INDEX="$GPU_INDEX" bash "$RESTORE" || true

    if [[ "$SUCCESS" -eq 1 && "$POWER_OFF_AT_END" -eq 1 ]]; then
        echo "All RTX 3090 GEMM sessions completed successfully. Powering off."
        sudo systemctl poweroff
    elif [[ "$SUCCESS" -eq 1 ]]; then
        echo "All RTX 3090 GEMM sessions completed successfully. POWER_OFF_AT_END=0."
    else
        echo "RTX 3090 GEMM run stopped or failed. GPU restored; no shutdown." >&2
    fi
    exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

mkdir -p "$OUTDIR"

nvcc -O3 -std=c++17 -lineinfo \
    "$SRC" -lcublas -lnvidia-ml -o "$BIN"

ldd "$BIN" | grep -E 'cublas|nvidia-ml|cuda' || true

for ((session=1; session<=SESSIONS; session++)); do
    session_id="${STAMP}_session${session}"
    seed=$((20263000 + session))
    output="$OUTDIR/gemm_3090_${session_id}.csv"

    echo "=== RTX 3090 GEMM | session $session/$SESSIONS | $session_id | reps=$REPS ==="

    CUDA_VISIBLE_DEVICES="$GPU_INDEX" \
    NVIDIA_TF32_OVERRIDE=0 \
    BENCH_EXPECTED_GPU="$EXPECTED_GPU" \
    stdbuf -oL -eL \
        "$BIN" "$output" "$REPS" "$session_id" "$seed"

    [[ -s "$output" ]] || { echo "ERROR: missing or empty output: $output" >&2; exit 1; }
    rows=$(( $(wc -l < "$output") - 1 ))
    expected=$((9 * REPS))
    if [[ "$rows" -ne "$expected" ]]; then
        echo "ERROR: $output has $rows rows; expected $expected." >&2
        exit 1
    fi
    echo "RTX 3090 GEMM session $session/$SESSIONS complete: $rows rows."
done

echo "Completed $SESSIONS RTX 3090 GEMM sessions with no pause between sessions."
echo "Measurements per problem size: $((SESSIONS * REPS))."
SUCCESS=1

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
GPU_INDEX=${GPU_INDEX:-0}
EXPECTED_GPU=${EXPECTED_GPU:-RTX 5060 Ti}
WARMUP_DURATION=${WARMUP_DURATION:-60}
LOGDIR="$ROOT/runs/GEMM/qc"
WARMUP_SRC="$ROOT/scripts/gpu_warmup_fp32.cu"
WARMUP_BIN="$ROOT/scripts/gpu_warmup_fp32"

mkdir -p "$LOGDIR"

name=$(nvidia-smi -i "$GPU_INDEX" --query-gpu=name --format=csv,noheader | xargs)
[[ "$name" == *"$EXPECTED_GPU"* ]] || {
    echo "ERROR: GPU $GPU_INDEX is '$name', expected a name containing '$EXPECTED_GPU'." >&2
    exit 1
}

echo "Configuring GPU $GPU_INDEX: $name"
sudo nvidia-smi -i "$GPU_INDEX" -pm 1
sudo nvidia-smi -i "$GPU_INDEX" --reset-gpu-clocks 2>/dev/null || true
sudo nvidia-smi -i "$GPU_INDEX" --reset-memory-clocks 2>/dev/null || true

default_power=$(nvidia-smi -i "$GPU_INDEX" --query-gpu=power.default_limit --format=csv,noheader,nounits | xargs)
sudo nvidia-smi -i "$GPU_INDEX" -pl "$default_power"

nvidia-smi -i "$GPU_INDEX" -q -d CLOCK,POWER,TEMPERATURE \
    > "$LOGDIR/enable_before_$(date +%Y%m%d_%H%M%S).log"

nvcc -O3 -std=c++17 "$WARMUP_SRC" -lcublas -o "$WARMUP_BIN"
CUDA_VISIBLE_DEVICES="$GPU_INDEX" NVIDIA_TF32_OVERRIDE=0 \
    "$WARMUP_BIN" "$WARMUP_DURATION"

nvidia-smi -i "$GPU_INDEX" -q -d CLOCK,POWER,TEMPERATURE \
    > "$LOGDIR/enable_after_$(date +%Y%m%d_%H%M%S).log"

nvidia-smi -i "$GPU_INDEX" \
    --query-gpu=index,name,temperature.gpu,power.draw,power.limit,clocks.current.graphics,clocks.current.memory \
    --format=csv

echo "GPU 5060 Ti ready."

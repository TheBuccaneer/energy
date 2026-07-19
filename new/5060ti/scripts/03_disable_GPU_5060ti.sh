#!/usr/bin/env bash
set -euo pipefail

GPU_INDEX=${GPU_INDEX:-0}

echo "Restoring GPU $GPU_INDEX..."
sudo nvidia-smi -i "$GPU_INDEX" --reset-gpu-clocks 2>/dev/null || true
sudo nvidia-smi -i "$GPU_INDEX" --reset-memory-clocks 2>/dev/null || true

default_power=$(nvidia-smi -i "$GPU_INDEX" --query-gpu=power.default_limit --format=csv,noheader,nounits | xargs)
sudo nvidia-smi -i "$GPU_INDEX" -pl "$default_power" 2>/dev/null || true
sudo nvidia-smi -i "$GPU_INDEX" -pm 0 2>/dev/null || true
sleep 3
nvidia-smi -i "$GPU_INDEX" \
    --query-gpu=index,name,temperature.gpu,power.draw,power.limit,clocks.current.graphics,clocks.current.memory \
    --format=csv

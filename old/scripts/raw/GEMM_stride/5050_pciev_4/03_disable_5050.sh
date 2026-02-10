#!/bin/bash
# GPU Disable Script

echo "=== GPU Disable & Reset ==="
echo ""

# Power Limits auf Default zurücksetzen
echo "Resetting power limits to default..."
for gpu_id in $(nvidia-smi --query-gpu=index --format=csv,noheader); do
    gpu_name=$(nvidia-smi -i $gpu_id --query-gpu=name --format=csv,noheader)
    default_pl=$(nvidia-smi -i $gpu_id --query-gpu=power.default_limit --format=csv,noheader,nounits | xargs printf "%.0f")
    
    echo "GPU $gpu_id ($gpu_name): Resetting to ${default_pl}W"
    sudo nvidia-smi -i $gpu_id -pl $default_pl
done

# Persistence Mode deaktivieren
echo ""
echo "Disabling persistence mode..."
sudo nvidia-smi -pm 0

echo ""
echo "✓ GPU reset complete!"
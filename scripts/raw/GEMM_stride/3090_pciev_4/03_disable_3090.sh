#!/bin/bash
# GPU Disable Script - Macht alle Änderungen vom Warmup rückgängig

echo "=== GPU Disable & Reset ==="
echo ""

# Alle Clock Locks zurücksetzen
echo "Resetting clock locks..."
sudo nvidia-smi --reset-gpu-clocks 2>/dev/null || true
sudo nvidia-smi --reset-memory-clocks 2>/dev/null || true

# Power Limits auf Default zurücksetzen
echo "Resetting power limits to default..."
for gpu_id in $(nvidia-smi --query-gpu=index --format=csv,noheader); do
    gpu_name=$(nvidia-smi -i $gpu_id --query-gpu=name --format=csv,noheader)
    default_pl=$(nvidia-smi -i $gpu_id --query-gpu=power.default_limit --format=csv,noheader,nounits | xargs printf "%.0f")
    
    echo "GPU $gpu_id ($gpu_name): Resetting to ${default_pl}W"
    sudo nvidia-smi -i $gpu_id -pl $default_pl 2>/dev/null || echo "  Warning: Could not reset power limit"
done

# Persistence Mode deaktivieren
echo ""
echo "Disabling persistence mode..."
sudo nvidia-smi -pm 0

echo ""
echo "Waiting for GPU to idle..."
sleep 3

# Final status
echo ""
echo "Final GPU status (should be in idle state):"
nvidia-smi --query-gpu=index,name,temperature.gpu,power.draw,clocks.current.graphics,clocks.current.memory --format=csv

echo ""
echo "✓ GPU reset complete!"
echo ""

# Optional: Logs erstellen
echo "Saving idle state snapshot..."
nvidia-smi -q -d CLOCK,POWER,TEMPERATURE > gpu_idle_state.log 2>/dev/null || true
echo "  (Saved to gpu_idle_state.log)"
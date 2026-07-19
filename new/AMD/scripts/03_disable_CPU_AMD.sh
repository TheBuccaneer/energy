#!/bin/bash
set -euo pipefail

[ "$EUID" -eq 0 ] || { echo "Run with sudo: sudo scripts/03_disable_CPU_AMD.sh"; exit 1; }

STATE_DIR=/tmp/energy_amd_measurement_state
if [ ! -d "$STATE_DIR" ]; then
    echo "No saved AMD measurement state found; nothing to restore."
    exit 0
fi

if [ -r "$STATE_DIR/governors" ]; then
    while IFS='|' read -r path value; do
        if [ -w "$path" ]; then
            echo "$value" > "$path"
        fi
    done < "$STATE_DIR/governors"
fi

if [ -r "$STATE_DIR/epp" ]; then
    while IFS='|' read -r path value; do
        if [ -w "$path" ]; then
            echo "$value" > "$path" 2>/dev/null || true
        fi
    done < "$STATE_DIR/epp"
fi

if [ -r "$STATE_DIR/perf_event_paranoid" ]; then
    cat "$STATE_DIR/perf_event_paranoid" > /proc/sys/kernel/perf_event_paranoid
fi

rm -rf "$STATE_DIR"
echo "AMD CPU governors, EPP settings, and perf permissions restored."

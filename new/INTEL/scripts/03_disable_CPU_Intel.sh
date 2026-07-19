#!/bin/bash
set -euo pipefail

[ "$EUID" -eq 0 ] || { echo "Run with sudo: sudo scripts/03_disable_CPU_Intel.sh"; exit 1; }

GOV_STATE=/tmp/energy_intel_governors.state
RAPL_STATE=/tmp/energy_intel_rapl_modes.state

if [ -r "$GOV_STATE" ]; then
    while IFS='|' read -r path value; do
        [ -w "$path" ] && echo "$value" > "$path"
    done < "$GOV_STATE"
    rm -f "$GOV_STATE"
fi

if [ -r "$RAPL_STATE" ]; then
    while IFS='|' read -r path mode; do
        [ -e "$path" ] && chmod "$mode" "$path" 2>/dev/null || true
    done < "$RAPL_STATE"
    rm -f "$RAPL_STATE"
fi

echo "CPU governors and RAPL file modes restored."

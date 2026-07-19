#!/bin/bash
set -euo pipefail

[ "$EUID" -eq 0 ] || { echo "Run with sudo: sudo scripts/01_enable_CPU_Intel.sh"; exit 1; }

GOV_STATE=/tmp/energy_intel_governors.state
RAPL_STATE=/tmp/energy_intel_rapl_modes.state
if [ -e "$GOV_STATE" ] || [ -e "$RAPL_STATE" ]; then
    echo "CPU measurement state already exists. Run sudo scripts/03_disable_CPU_Intel.sh first."
    exit 1
fi
: > "$GOV_STATE"
: > "$RAPL_STATE"

modprobe intel_rapl_msr 2>/dev/null || true

for cpu in /sys/devices/system/cpu/cpu[0-9]*; do
    gov="$cpu/cpufreq/scaling_governor"
    [ -r "$gov" ] || continue
    printf '%s|%s\n' "$gov" "$(cat "$gov")" >> "$GOV_STATE"
    echo performance > "$gov"
done

while IFS= read -r energy; do
    [ -e "$energy" ] || continue
    printf '%s|%s\n' "$energy" "$(stat -c '%a' "$energy")" >> "$RAPL_STATE"
    chmod a+r "$energy" 2>/dev/null || true
done < <(find -L /sys/class/powercap -name energy_uj 2>/dev/null | sort)

user_name=${SUDO_USER:-root}
rapl=$(find -L /sys/class/powercap -path '*intel-rapl:*/energy_uj' 2>/dev/null | head -n1 || true)
if [ -z "$rapl" ]; then
    echo "WARNING: no RAPL package energy_uj file found"
elif sudo -u "$user_name" test -r "$rapl"; then
    echo "Governor: performance"
    echo "RAPL readable as $user_name: yes"
else
    echo "Governor: performance"
    echo "RAPL readable as $user_name: no; runner will use sudo for benchmarks"
fi

echo "Turbo and frequency limits were not changed."

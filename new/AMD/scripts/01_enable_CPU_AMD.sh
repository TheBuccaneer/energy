#!/bin/bash
set -euo pipefail

[ "$EUID" -eq 0 ] || { echo "Run with sudo: sudo scripts/01_enable_CPU_AMD.sh"; exit 1; }

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
STATE_DIR=/tmp/energy_amd_measurement_state
if [ -e "$STATE_DIR" ]; then
    echo "AMD measurement state already exists. Run sudo scripts/03_disable_CPU_AMD.sh first."
    exit 1
fi
mkdir -p "$STATE_DIR"
: > "$STATE_DIR/governors"
: > "$STATE_DIR/epp"

SUCCESS=0
rollback_on_error() {
    status=$?
    trap - EXIT
    if [ "$SUCCESS" -ne 1 ]; then
        echo "AMD preparation failed; restoring previous settings..." >&2
        bash "$SCRIPT_DIR/03_disable_CPU_AMD.sh" >/dev/null 2>&1 || true
    fi
    exit "$status"
}
trap rollback_on_error EXIT

cat /proc/sys/kernel/perf_event_paranoid > "$STATE_DIR/perf_event_paranoid"

for cpu in /sys/devices/system/cpu/cpu[0-9]*; do
    gov="$cpu/cpufreq/scaling_governor"
    if [ -r "$gov" ]; then
        printf '%s|%s\n' "$gov" "$(cat "$gov")" >> "$STATE_DIR/governors"
        echo performance > "$gov"
    fi

    epp="$cpu/cpufreq/energy_performance_preference"
    if [ -r "$epp" ] && [ -w "$epp" ]; then
        printf '%s|%s\n' "$epp" "$(cat "$epp")" >> "$STATE_DIR/epp"
        echo performance > "$epp" 2>/dev/null || true
    fi
done

# The benchmark opens a system-wide AMD package-energy perf event as the
# normal user. This permission is temporary and restored by script 03.
echo -1 > /proc/sys/kernel/perf_event_paranoid

if [ ! -r /sys/bus/event_source/devices/power/events/energy-pkg ]; then
    echo "ERROR: power/energy-pkg/ is not available on this kernel." >&2
    exit 1
fi

user_name=${SUDO_USER:-root}
if sudo -u "$user_name" perf stat -a -e power/energy-pkg/ -- sleep 0.15 >/dev/null 2>&1; then
    echo "Governor: performance"
    echo "AMD package energy via perf readable as $user_name: yes"
else
    echo "ERROR: AMD package energy is still not readable as $user_name." >&2
    exit 1
fi

echo "Boost and frequency limits were not changed."
SUCCESS=1
trap - EXIT

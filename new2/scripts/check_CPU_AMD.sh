#!/bin/bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
TMP_DIR=$(mktemp -d /tmp/energy-amd-check.XXXXXX)
TEMP_ENABLED=0
cleanup() {
    status=$?
    if [ "$TEMP_ENABLED" -eq 1 ]; then
        sudo bash "$ROOT/scripts/03_disable_CPU_AMD.sh" >/dev/null 2>&1 || true
    fi
    rm -rf "$TMP_DIR"
    exit "$status"
}
trap cleanup EXIT

fail() { echo "ERROR: $*" >&2; exit 1; }

echo "=== CPU ==="
VENDOR=$(awk -F: '/vendor_id/{gsub(/[[:space:]]/,"",$2); print $2; exit}' /proc/cpuinfo)
[ "$VENDOR" = "AuthenticAMD" ] || fail "This package is intended for an AMD CPU; detected $VENDOR."
lscpu | grep -E 'Model name|Socket|Core|Thread|CPU\(s\)|NUMA node\(s\)'
ONLINE_CPUS=$(nproc)
[ "$ONLINE_CPUS" -ge 64 ] || fail "64-thread extension requested, but only $ONLINE_CPUS logical CPUs are online."

echo
echo "=== Required commands ==="
for c in g++ perf stdbuf awk tee ldconfig ldd; do
    command -v "$c" >/dev/null 2>&1 || fail "Missing command: $c"
    echo "$c: $(command -v "$c")"
done

echo
echo "=== OpenBLAS ==="
printf '#include <cblas.h>\n' | g++ -x c++ -E - >/dev/null 2>&1 || fail "Missing cblas.h (install libopenblas-dev)."
ldconfig -p 2>/dev/null | grep 'libopenblas\.so' >/dev/null || fail "Missing libopenblas.so (install libopenblas-dev)."

echo
echo "=== oneDNN ==="
if printf '#include <dnnl.hpp>\n' | g++ -x c++ -E - >/dev/null 2>&1; then
    echo "Header: dnnl.hpp"
elif printf '#include <oneapi/dnnl/dnnl.hpp>\n' | g++ -x c++ -E - >/dev/null 2>&1; then
    echo "Header: oneapi/dnnl/dnnl.hpp"
else
    fail "Missing oneDNN headers (install libdnnl-dev)."
fi
ldconfig -p 2>/dev/null | grep 'libdnnl\.so' >/dev/null || fail "Missing libdnnl.so (install libdnnl-dev)."

echo
echo "=== Build all six workloads ==="
CXX=(g++ -O3 -march=native -std=c++17 -fopenmp -I"$ROOT/scripts/common")
"${CXX[@]}" "$ROOT/scripts/GEMM/CPU/AMD/main_gemm.cpp" -lopenblas -lpthread -lm -o "$TMP_DIR/main_gemm"
"${CXX[@]}" "$ROOT/scripts/STRIDED_GEMM/CPU/AMD/main_gemm_strided.cpp" -lopenblas -lpthread -lm -o "$TMP_DIR/main_gemm_strided"
"${CXX[@]}" "$ROOT/scripts/STREAM/CPU/AMD/main_stream.cpp" -lpthread -lm -o "$TMP_DIR/main_stream"
"${CXX[@]}" "$ROOT/scripts/AXPY/CPU/AMD/main_axpy.cpp" -lpthread -lm -o "$TMP_DIR/main_axpy"
"${CXX[@]}" "$ROOT/scripts/REDUCTION/CPU/AMD/main_reduction.cpp" -lpthread -lm -o "$TMP_DIR/main_reduction"
"${CXX[@]}" "$ROOT/scripts/CONV2D/CPU/AMD/main_conv2d.cpp" -ldnnl -lpthread -lm -o "$TMP_DIR/main_conv2d"
echo "All six workloads compiled and linked successfully."

echo
echo "=== Thread grid ==="
grep -F 'THREAD_COUNTS{1, 2, 4, 8, 10, 16, 20, 32, 64}' \
    "$ROOT/scripts/common/benchmark_common.hpp" >/dev/null || fail "Expected 1..64 thread grid not found."
echo "1, 2, 4, 8, 10, 16, 20, 32, 64"

echo
echo "=== Exact C++ AMD energy backend ==="
if [ ! -d /tmp/energy_amd_measurement_state ]; then
    echo "Temporarily enabling AMD measurement permissions for the exact backend test..."
    sudo bash "$ROOT/scripts/01_enable_CPU_AMD.sh"
    TEMP_ENABLED=1
fi
"${CXX[@]}" "$ROOT/scripts/check_energy_backend.cpp" -lpthread -lm -o "$TMP_DIR/check_energy_backend"
"$TMP_DIR/check_energy_backend"

echo
echo "=== Output isolation ==="
if grep -R --line-number --exclude='check_CPU_AMD.sh' 'CPU/INTEL' "$ROOT/scripts"; then
    fail "An Intel path remains in the AMD scripts."
fi
if grep -nF '"$ROOT/runs/' "$ROOT/scripts/02_run_CPU_AMD_5min_autoshutdown.sh"; then
    fail "Runner still targets runs/ instead of runs2/."
fi
grep -F '"$ROOT/runs2/' "$ROOT/scripts/02_run_CPU_AMD_5min_autoshutdown.sh" >/dev/null \
    || fail "Runner does not target runs2."
echo "Only CPU/AMD and runs2 paths are used."

echo
echo "Environment and package check passed."

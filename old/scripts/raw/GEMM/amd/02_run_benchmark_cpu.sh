#!/bin/bash
# 02_run_benchmark_cpu.sh - Compile and run CPU GEMM benchmark with optional output path
# Usage: ./02_run_benchmark_cpu.sh [output_path]

set -e  # Exit on error

# Parse optional output path
OUTPUT_PATH="$1"

echo "========================================"
echo "CPU GEMM Benchmark - Compile & Run"
echo "========================================"
echo ""

# Check if OpenBLAS is installed
echo "Checking OpenBLAS installation..."
if ! pkg-config --exists openblas 2>/dev/null && ! ldconfig -p | grep -q libopenblas; then
    echo "WARNING: OpenBLAS not found!"
    echo "Install with: sudo apt-get install libopenblas-dev"
    echo "Attempting compilation anyway..."
fi
echo ""

# Compile
echo "Compiling cpu_bench..."
g++ -O3 -march=native -std=c++17 -Wall -Wextra -o cpu_bench main.cpp -lopenblas

if [ $? -ne 0 ]; then
    echo "ERROR: Compilation failed!"
    exit 1
fi

echo "✓ Compilation successful"
echo ""

# Check RAPL access
echo "Checking RAPL energy access..."
if [ -r /sys/class/powercap/intel-rapl:0/energy_uj ] || [ -r /sys/class/powercap/amd-rapl:0/energy_uj ]; then
    echo "✓ RAPL accessible"
else
    echo "WARNING: RAPL not accessible (energy will be -1)"
    echo "Fix with: sudo chmod -R a+r /sys/class/powercap/*rapl*"
fi
echo ""

# Check CPU stabilization
echo "Checking CPU frequency stabilization..."
GOVERNOR=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo "unknown")
if [ "$GOVERNOR" == "performance" ]; then
    echo "✓ CPU governor: performance"
else
    echo "WARNING: CPU governor is '$GOVERNOR' (not 'performance')"
    echo "Run: sudo bash 01_enable_CPU_Intel.sh (works for AMD too)"
fi
echo ""

# Run benchmark
echo "Running benchmark..."
echo "Expected duration: ~3-4 hours (3150 measurements)"
echo ""

if [ -z "$OUTPUT_PATH" ]; then
    # No output path provided - use default
    echo "Using default output path: data/raw/energy_benchmark_cpu.csv"
    ./cpu_bench
else
    # Output path provided
    echo "Output path: $OUTPUT_PATH"
    ./cpu_bench --output "$OUTPUT_PATH"
fi

echo ""
echo "========================================"
echo "Benchmark complete!"
echo "========================================"
echo ""

# Automatic shutdown after benchmark
echo "⚠️  SYSTEM WILL SHUTDOWN IN 30 SECONDS  ⚠️"
echo ""
echo "Press Ctrl+C NOW to cancel shutdown!"
echo ""

for i in {30..1}; do
    echo -ne "Shutting down in $i seconds...\r"
    sleep 1
done

echo ""
echo "Shutting down NOW..."

# Force shutdown (doesn't wait for user processes)
sudo shutdown -h now

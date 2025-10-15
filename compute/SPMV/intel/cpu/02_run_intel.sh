#!/bin/bash

# run_spmv_cpu.sh - Build and run SpMV CPU benchmark with thread scaling

set -e  # Exit on error

# Configuration
SOURCE="main.cpp"
BINARY="main"
COMPILER="g++"
FLAGS="-O3 -DNDEBUG -fopenmp -std=c++17 -march=native"

# Thread counts to benchmark
THREAD_COUNTS=(1 2 4 8 10 16 20)

# System info
PHYS_CORES=$(lscpu -p | grep -v '^#' | sort -u -t, -k2,4 | wc -l)
TOTAL_CORES=$(nproc)

echo "========================================"
echo "SpMV CPU Benchmark - Thread Scaling"
echo "========================================"

# Build
echo "Building ${BINARY}..."
${COMPILER} ${FLAGS} -o ${BINARY} ${SOURCE}
echo "✓ Build successful"
echo ""

# System info
echo "System info:"
echo "  Physical cores: ${PHYS_CORES}"
echo "  Total cores:    ${TOTAL_CORES}"
echo "  Thread sweep:   ${THREAD_COUNTS[@]}"
echo ""

# Check RAPL permissions
if [ ! -r /sys/class/powercap/intel-rapl:0/energy_uj ]; then
    echo "Warning: No RAPL read access. Run as root or:"
    echo "   sudo chmod -R a+r /sys/class/powercap/intel-rapl*"
    echo ""
fi

# Run for each thread count
for THREADS in "${THREAD_COUNTS[@]}"; do
    echo "========================================"
    echo "Running with ${THREADS} thread(s)"
    echo "========================================"
    echo ""
    
    ./${BINARY} --threads ${THREADS}
    
    # Move data folder to thread-specific directory
    if [ -d "data" ]; then
        mv data data_${THREADS}
        echo "✓ Data saved to: data_${THREADS}/raw/spmv_cpu.csv"
    fi
    
    echo ""
    echo "✓ Completed ${THREADS} thread(s)"
    echo ""
done

echo "========================================"
echo "✓ All runs complete!"
echo "Results saved in:"
for THREADS in "${THREAD_COUNTS[@]}"; do
    if [ -d "data_${THREADS}" ]; then
        echo "  data_${THREADS}/raw/spmv_cpu.csv"
    fi
done
echo "========================================"
#!/bin/bash

# run_stream_cpu.sh - Build and run STREAM CPU benchmark with thread scaling

set -e  # Exit on error

# Configuration
SOURCE="main.cpp"
BINARY="main"
COMPILER="g++"
FLAGS="-O3 -march=native -fopenmp -std=c++17"

# Optional build flags (uncomment as needed)
# FLAGS="$FLAGS -DNT_STORE"     # Non-temporal stores
# FLAGS="$FLAGS -DUSE_DOUBLE"   # FP64

# Fixed thread counts
THREAD_COUNTS=(1 2 4 8 10 16 20)

# System info
PHYS_CORES=$(lscpu -p | grep -v '^#' | sort -u -t, -k2,4 | wc -l)
TOTAL_CORES=$(nproc)

echo "========================================"
echo "STREAM CPU Benchmark - Thread Scaling"
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
    echo "⚠️  Warning: No RAPL read access. Run as root or:"
    echo "   sudo chmod -R a+r /sys/class/powercap/intel-rapl*"
    echo ""
fi

# OpenMP settings (fixed)
export OMP_PLACES=cores
export OMP_PROC_BIND=spread

# Run for each thread count
for THREADS in "${THREAD_COUNTS[@]}"; do
    echo "========================================"
    echo "Running with ${THREADS} thread(s)"
    echo "========================================"
    
    export OMP_NUM_THREADS=${THREADS}
    
    echo "OpenMP settings:"
    echo "  OMP_NUM_THREADS: ${OMP_NUM_THREADS}"
    echo "  OMP_PLACES:      ${OMP_PLACES}"
    echo "  OMP_PROC_BIND:   ${OMP_PROC_BIND}"
    echo ""
    
    ./${BINARY}
    
    # Rename data folder with thread count suffix
    if [ -d "data" ]; then
        mv data data_${THREADS}
        echo "✓ Data saved to: data_${THREADS}/"
    fi
    
    echo ""
    echo "✓ Completed ${THREADS} thread(s)"
    echo ""
done

echo "========================================"
echo "✓ All runs complete!"
echo "Results saved in:"
for THREADS in "${THREAD_COUNTS[@]}"; do
    echo "  data_${THREADS}/raw/stream_triad_cpu.csv"
done
echo "========================================"
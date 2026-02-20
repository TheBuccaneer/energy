#!/bin/bash
# run_benchmark.sh - Compile and run GEMM benchmark with optional output path
# Usage: ./run_benchmark.sh [output_path]

export NVIDIA_TF32_OVERRIDE=0   # disable TF32 globally

set -e  # Exit on error

# Parse optional output path
OUTPUT_PATH="$1"

echo "========================================"
echo "CONV Benchmark - Compile & Run"
echo "========================================"
echo ""

# Compile
echo "Compiling gemm_bench..."
nvcc -O3 -std=c++17 -o conv2d_3090 main.cu -lcudnn -lnvidia-ml

if [ $? -ne 0 ]; then
    echo "ERROR: Compilation failed!"
    exit 1
fi

echo "✓ Compilation successful"
echo ""

# Run benchmark
echo "Running benchmark..."
if [ -z "$OUTPUT_PATH" ]; then
    # No output path provided - use default
    echo "Using default output path"
    ./conv2d_3090
else
    # Output path provided
    echo "Output path: $OUTPUT_PATH"
    ./conv2d_3090 --output "$OUTPUT_PATH"
fi

echo ""
echo "========================================"
echo "Benchmark complete!"
echo "========================================"

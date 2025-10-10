#!/bin/bash

# Script to compile and run cpu_bench 10 times with cooling periods
# Saves data folders sequentially: data_1/, data_2/, ..., data_10/

set -e  # Exit on error

echo "========================================"
echo "CPU GEMM Benchmark - Automated Runner"
echo "========================================"
echo ""

# ============================================================================
# Step 1: Compile
# ============================================================================

echo "Step 1: Compiling cpu_bench..."
echo "Command: g++ -O3 -march=native -std=c++17 -o cpu_bench main.cpp -lopenblas"
echo ""

g++ -O3 -march=native -std=c++17 -o cpu_bench main.cpp -lopenblas

if [ $? -ne 0 ]; then
    echo "ERROR: Compilation failed!"
    exit 1
fi

if [ ! -f "cpu_bench" ]; then
    echo "ERROR: cpu_bench binary not found after compilation!"
    exit 1
fi

echo "✓ Compilation successful"
echo ""

# ============================================================================
# Step 2: Cleanup all existing data folders
# ============================================================================

echo "Step 2: Cleaning up existing data folders..."

# Remove main data folder if exists
if [ -d "data" ]; then
    echo "  delete or change data first/"
    exit 1
fi

echo "✓ Cleanup complete"
echo ""

# ============================================================================
# Step 3: Run benchmark 10 times
# ============================================================================

echo "Step 3: Running benchmark 10 times..."
echo "========================================"
echo ""

TOTAL_RUNS=10
COOLDOWN_SECONDS=20

for i in $(seq 1 $TOTAL_RUNS); do
    echo "┌────────────────────────────────────┐"
    echo "│ Run $i/$TOTAL_RUNS starting at $(date +%H:%M:%S) │"
    echo "└────────────────────────────────────┘"
    echo ""

    # Run the benchmark
    ./cpu_bench

    # Check if data folder was created
    if [ -d "data" ]; then
        # Move data folder to numbered version
        mv data data_$i
        echo ""
        echo "✓ Run $i completed successfully"
        echo "✓ Data saved to: data_$i/"
        
        # Show quick stats
        if [ -f "data_$i/raw/energy_benchmark_cpu.csv" ]; then
            LINES=$(wc -l < "data_$i/raw/energy_benchmark_cpu.csv")
            echo "✓ CSV contains $LINES lines (1 header + $((LINES-1)) measurements)"
        fi
    else
        echo ""
        echo "⚠ WARNING: Run $i failed - no data folder created!"
        echo "⚠ Continuing with next run..."
    fi

    # Cool-down pause between runs (except after last run)
    if [ $i -lt $TOTAL_RUNS ]; then
        echo ""
        echo "Cooling down for $COOLDOWN_SECONDS seconds..."
        echo "(CPU thermal stabilization)"
        
        # Progress bar for cooldown
        for s in $(seq 1 $COOLDOWN_SECONDS); do
            printf "\r[%-60s] %d/%ds" "$(printf '#%.0s' $(seq 1 $((s*60/COOLDOWN_SECONDS))))" "$s" "$COOLDOWN_SECONDS"
            sleep 1
        done
        echo ""
    fi

    echo ""
done

# ============================================================================
# Step 4: Summary
# ============================================================================

echo ""
echo "========================================"
echo "All $TOTAL_RUNS benchmark runs completed!"
echo "========================================"
echo ""

# List all data folders
echo "Data folders created:"
for i in $(seq 1 $TOTAL_RUNS); do
    if [ -d "data_$i" ]; then
        SIZE=$(du -sh "data_$i" | cut -f1)
        if [ -f "data_$i/raw/energy_benchmark_cpu.csv" ]; then
            LINES=$(wc -l < "data_$i/raw/energy_benchmark_cpu.csv")
            echo "  ✓ data_$i/ ($SIZE, $LINES lines)"
        else
            echo "  ✗ data_$i/ ($SIZE, CSV missing!)"
        fi
    else
        echo "  ✗ data_$i/ (folder missing!)"
    fi
done

echo ""
echo "Quick validation:"
ls -lh data_*/raw/*.csv 2>/dev/null || echo "  No CSV files found!"
cat data_*/raw/energy_benchmark_cpu.csv | awk 'NR==1 || !/^timestamp/' > merged_all_runs.csv
echo "Done!"
#!/bin/bash

# Script to compile main.cpp once and run the reduction benchmark
# with different thread counts (1, 2, 4, 8, 10, 16, 20)
# Each thread count: 10 runs with cooling periods
# Results saved in: 1/data_1..10/, 2/data_1..10/, etc.

set -e  # Exit on error

echo "========================================"
echo "CPU Reduction Benchmark - Automated Runner"
echo "========================================"
echo ""

# ============================================================================
# Step 1: Compile once
# ============================================================================

echo "Step 1: Compiling main.cpp..."
echo "Command: g++ -O3 -march=native -std=c++17 -o main main.cpp -lopenblas"
echo ""

g++ -O3 -march=native -std=c++17 -o main main.cpp -lopenblas

if [ $? -ne 0 ]; then
    echo "ERROR: Compilation failed!"
    exit 1
fi

if [ ! -f "main" ]; then
    echo "ERROR: main binary not found after compilation!"
    exit 1
fi

echo "✓ Compilation successful"
echo ""

# ============================================================================
# Step 2: Check for existing thread folders
# ============================================================================

echo "Step 2: Checking for existing result folders..."

THREAD_COUNTS=(1 2 4 8 10 16 20)
EXISTING_FOLDERS=()

for threads in "${THREAD_COUNTS[@]}"; do
    if [ -d "$threads" ]; then
        EXISTING_FOLDERS+=("$threads")
    fi
done

if [ ${#EXISTING_FOLDERS[@]} -gt 0 ]; then
    echo "ERROR: The following folders already exist:"
    for folder in "${EXISTING_FOLDERS[@]}"; do
        echo "  - $folder/"
    done
    echo ""
    echo "Please delete or rename these folders first."
    exit 1
fi

# Also check for stray data folder
if [ -d "data" ]; then
    echo "ERROR: 'data' folder already exists. Please remove it first."
    exit 1
fi

echo "✓ No conflicting folders found"
echo ""

# ============================================================================
# Step 3: Run benchmarks for all thread counts
# ============================================================================

TOTAL_RUNS=10
COOLDOWN_SECONDS=20

echo "Step 3: Running benchmarks..."
echo "========================================"
echo "Thread counts: ${THREAD_COUNTS[@]}"
echo "Runs per thread count: $TOTAL_RUNS"
echo "Cooldown between runs: ${COOLDOWN_SECONDS}s"
echo "========================================"
echo ""

for threads in "${THREAD_COUNTS[@]}"; do
    echo ""
    echo "╔════════════════════════════════════════╗"
    echo "║ Starting Thread Count: $threads               ║"
    echo "╚════════════════════════════════════════╝"
    echo ""
    
    # Create folder for this thread count
    mkdir -p "$threads"
    echo "✓ Created folder: $threads/"
    echo ""
    
    # Run 10 times for this thread count
    for i in $(seq 1 $TOTAL_RUNS); do
        echo "┌──────────────────────────────────────┐"
        echo "│ Threads=$threads | Run $i/$TOTAL_RUNS | $(date +%H:%M:%S) │"
        echo "└──────────────────────────────────────┘"
        echo ""
        
        # Run the benchmark with current thread count
        echo "Command: ./main --threads $threads"
        ./main --threads $threads
        
        # Check if data folder was created
        if [ -d "data" ]; then
            # Rename data folder
            mv data "data_$i"
            
            # Move into thread folder
            mv "data_$i" "$threads/"
            
            echo ""
            echo "✓ Run $i completed successfully"
            echo "✓ Data saved to: $threads/data_$i/"
            
            # Show quick stats
            if [ -f "$threads/data_$i/raw/reduction_cpu_openblas.csv" ]; then
                LINES=$(wc -l < "$threads/data_$i/raw/reduction_cpu_openblas.csv")
                echo "✓ CSV contains $LINES lines (1 header + $((LINES-1)) measurements)"
            fi
        else
            echo ""
            echo "⚠ WARNING: Run $i failed - no data folder created!"
            echo "⚠ Continuing with next run..."
        fi
        
        # Cool-down pause between runs (except after last run of last thread count)
        if [ $i -lt $TOTAL_RUNS ] || [ $threads -ne ${THREAD_COUNTS[-1]} ]; then
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
    
    echo "✓ Completed all runs for thread count $threads"
    echo ""
done

# ============================================================================
# Step 4: Summary
# ============================================================================

echo ""
echo "========================================"
echo "All benchmarks completed!"
echo "========================================"
echo ""

# Summary for each thread count
for threads in "${THREAD_COUNTS[@]}"; do
    echo "Thread count $threads:"
    
    if [ -d "$threads" ]; then
        FOLDER_SIZE=$(du -sh "$threads" | cut -f1)
        echo "  Folder size: $FOLDER_SIZE"
        
        # Count successful runs
        SUCCESS_COUNT=0
        for i in $(seq 1 $TOTAL_RUNS); do
            if [ -f "$threads/data_$i/raw/reduction_cpu_openblas.csv" ]; then
                SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
            fi
        done
        
        echo "  Successful runs: $SUCCESS_COUNT/$TOTAL_RUNS"
        
        # List data folders
        for i in $(seq 1 $TOTAL_RUNS); do
            if [ -d "$threads/data_$i" ]; then
                if [ -f "$threads/data_$i/raw/reduction_cpu_openblas.csv" ]; then
                    LINES=$(wc -l < "$threads/data_$i/raw/reduction_cpu_openblas.csv")
                    echo "    ✓ data_$i/ ($LINES lines)"
                else
                    echo "    ✗ data_$i/ (CSV missing!)"
                fi
            else
                echo "    ✗ data_$i/ (folder missing!)"
            fi
        done
    else
        echo "  ✗ Folder missing!"
    fi
    echo ""
done

echo "Final folder structure:"
tree -L 2 -d 2>/dev/null || ls -la

echo ""
echo "Done!"
#!/bin/bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
COMMON="$ROOT/scripts/common"
STAMP=$(date +%Y%m%d_%H%M%S)
REPS=${REPS:-10}
SESSIONS=${SESSIONS:-5}
WORKLOAD_PAUSE_SECONDS=${WORKLOAD_PAUSE_SECONDS:-300}
SESSION_PAUSE_SECONDS=${SESSION_PAUSE_SECONDS:-300}
POWER_OFF_AT_END=${POWER_OFF_AT_END:-1}

# Obtain sudo once and keep the credential alive for the unattended run.
sudo -v
(
    while true; do
        sudo -n true 2>/dev/null || exit
        sleep 60
    done
) &
SUDO_KEEPALIVE_PID=$!

SUCCESS=0
cleanup() {
    status=$?
    trap - EXIT

    kill "$SUDO_KEEPALIVE_PID" 2>/dev/null || true
    wait "$SUDO_KEEPALIVE_PID" 2>/dev/null || true

    echo "Restoring CPU governors and RAPL permissions..."
    sudo bash "$ROOT/scripts/03_disable_CPU_Intel.sh" || true

    if [ "$SUCCESS" -eq 1 ] && [ "$POWER_OFF_AT_END" -eq 1 ]; then
        echo "All campaigns completed successfully. Powering off."
        sudo systemctl poweroff
    elif [ "$SUCCESS" -eq 1 ]; then
        echo "All campaigns completed successfully. POWER_OFF_AT_END=0; system remains running."
    else
        echo "Run stopped or failed. Settings were restored; system will not power off." >&2
    fi

    exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

build() {
    g++ -O3 -march=native -std=c++17 -fopenmp -I"$COMMON" \
        "$ROOT/scripts/GEMM/CPU/INTEL/main_gemm.cpp" \
        -lopenblas -lpthread -lm -o "$ROOT/scripts/GEMM/CPU/INTEL/main_gemm"

    g++ -O3 -march=native -std=c++17 -fopenmp -I"$COMMON" \
        "$ROOT/scripts/STRIDED_GEMM/CPU/INTEL/main_gemm_strided.cpp" \
        -lopenblas -lpthread -lm -o "$ROOT/scripts/STRIDED_GEMM/CPU/INTEL/main_gemm_strided"

    g++ -O3 -march=native -std=c++17 -fopenmp -I"$COMMON" \
        "$ROOT/scripts/STREAM/CPU/INTEL/main_stream.cpp" \
        -lpthread -lm -o "$ROOT/scripts/STREAM/CPU/INTEL/main_stream"

    g++ -O3 -march=native -std=c++17 -fopenmp -I"$COMMON" \
        "$ROOT/scripts/AXPY/CPU/INTEL/main_axpy.cpp" \
        -lpthread -lm -o "$ROOT/scripts/AXPY/CPU/INTEL/main_axpy"

    g++ -O3 -march=native -std=c++17 -fopenmp -I"$COMMON" \
        "$ROOT/scripts/REDUCTION/CPU/INTEL/main_reduction.cpp" \
        -lpthread -lm -o "$ROOT/scripts/REDUCTION/CPU/INTEL/main_reduction"

    g++ -O3 -march=native -std=c++17 -fopenmp -I"$COMMON" \
        "$ROOT/scripts/CONV2D/CPU/INTEL/main_conv2d.cpp" \
        -ldnnl -lpthread -lm -o "$ROOT/scripts/CONV2D/CPU/INTEL/main_conv2d" || {
            echo "Conv2D build failed. Install oneDNN development headers/library (libdnnl-dev)."
            exit 1
        }
}

for workload in GEMM STRIDED_GEMM STREAM AXPY REDUCTION CONV2D; do
    mkdir -p "$ROOT/runs/$workload/CPU/INTEL"
done

build

RAPL=$(find -L /sys/class/powercap -path '*intel-rapl:*/energy_uj' 2>/dev/null | head -n1 || true)
if [ -n "$RAPL" ] && [ -r "$RAPL" ]; then
    RUN=(env OMP_PROC_BIND=close OMP_PLACES=cores OMP_DYNAMIC=FALSE DNNL_VERBOSE=0)
    USING_SUDO=0
else
    RUN=(sudo env OMP_PROC_BIND=close OMP_PLACES=cores OMP_DYNAMIC=FALSE DNNL_VERBOSE=0)
    USING_SUDO=1
fi

workloads=(GEMM STRIDED_GEMM STREAM AXPY REDUCTION CONV2D)

run_one() {
    local workload=$1 session_id=$2 seed=$3
    local binary output stem
    case "$workload" in
        GEMM)
            binary="$ROOT/scripts/GEMM/CPU/INTEL/main_gemm"
            stem=gemm ;;
        STRIDED_GEMM)
            binary="$ROOT/scripts/STRIDED_GEMM/CPU/INTEL/main_gemm_strided"
            stem=gemm_strided ;;
        STREAM)
            binary="$ROOT/scripts/STREAM/CPU/INTEL/main_stream"
            stem=stream ;;
        AXPY)
            binary="$ROOT/scripts/AXPY/CPU/INTEL/main_axpy"
            stem=axpy ;;
        REDUCTION)
            binary="$ROOT/scripts/REDUCTION/CPU/INTEL/main_reduction"
            stem=reduction ;;
        CONV2D)
            binary="$ROOT/scripts/CONV2D/CPU/INTEL/main_conv2d"
            stem=conv2d ;;
        *) echo "Unknown workload: $workload"; exit 1 ;;
    esac

    output="$ROOT/runs/$workload/CPU/INTEL/${stem}_intel_${session_id}.csv"
    echo "=== $workload | $session_id | seed=$seed | reps=$REPS ==="
    set +e
    "${RUN[@]}" "$binary" "$output" "$REPS" "$session_id" "$seed"
    status=$?
    set -e
    if [ "$USING_SUDO" -eq 1 ] && [ -e "$output" ]; then
        sudo chown "$(id -u):$(id -g)" "$output"
    fi
    [ "$status" -eq 0 ] || exit "$status"
}

for ((session=1; session<=SESSIONS; session++)); do
    session_id="${STAMP}_session${session}"
    start=$(( (session - 1) % ${#workloads[@]} ))

    for ((offset=0; offset<${#workloads[@]}; offset++)); do
        index=$(( (start + offset) % ${#workloads[@]} ))
        workload=${workloads[$index]}
        seed=$((20260000 + session * 100 + index))
        run_one "$workload" "$session_id" "$seed"
        if [ "$offset" -lt $(( ${#workloads[@]} - 1 )) ]; then
            sleep "$WORKLOAD_PAUSE_SECONDS"
        fi
    done

    if [ "$session" -lt "$SESSIONS" ]; then
        sleep "$SESSION_PAUSE_SECONDS"
    fi
done

echo "Completed $SESSIONS sessions × $REPS repetitions = $((SESSIONS * REPS)) runs per configuration."
SUCCESS=1

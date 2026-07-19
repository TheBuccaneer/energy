#!/usr/bin/env bash
set -euo pipefail

PLATFORM="AMD"
PLATFORM_LC="amd"
THREAD_FILTER="1,4,10,32,64"
EXPECTED_THREADS=5
REPS=${REPS:-3}
SEED=${SEED:-4096}

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)
SRC="$SCRIPT_DIR/STRIDED_GEMM/main_gemm_strided_${PLATFORM_LC}.cpp"
COMMON="$SCRIPT_DIR/common"
BUILD_DIR="$SCRIPT_DIR/STRIDED_GEMM/.build"
BIN="$BUILD_DIR/main_gemm_strided_${PLATFORM_LC}_quickcheck"
RUN_DIR="$ROOT_DIR/runs/STRIDED_GEMM"
ENABLE="$SCRIPT_DIR/01_enable_CPU_${PLATFORM}.sh"
RESTORE="$SCRIPT_DIR/03_disable_CPU_${PLATFORM}.sh"
SETTINGS_TOUCHED=0
KEEPALIVE_PID=""

cleanup() {
    local status=$?
    trap - EXIT INT TERM
    if [[ -n "$KEEPALIVE_PID" ]]; then
        kill "$KEEPALIVE_PID" 2>/dev/null || true
        wait "$KEEPALIVE_PID" 2>/dev/null || true
    fi
    if (( SETTINGS_TOUCHED == 1 )) && [[ -f "$RESTORE" ]]; then
        echo "[cleanup] Restoring ${PLATFORM} CPU settings..."
        sudo bash "$RESTORE" || status=1
    fi
    exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

for required in "$SRC" "$ENABLE" "$RESTORE" "$COMMON/benchmark_common.hpp"; do
    if [[ ! -f "$required" ]]; then
        echo "Missing required file: $required" >&2
        exit 2
    fi
done

mkdir -p "$BUILD_DIR" "$RUN_DIR"

echo "[build] Compiling quickcheck without OpenMP..."
g++ -O3 -march=native -std=c++17 \
    -I"$COMMON" \
    "$SRC" \
    -lopenblas -lpthread -lm \
    -o "$BIN"

sudo -v
(
    while true; do
        sudo -n true || exit
        sleep 60
        kill -0 "$$" 2>/dev/null || exit
    done
) 2>/dev/null &
KEEPALIVE_PID=$!

SETTINGS_TOUCHED=1
sudo bash "$ENABLE"

stamp=$(date +%Y%m%d_%H%M%S)
session_id="quickcheck_strided_gemm_${PLATFORM_LC}_${stamp}"
output="$RUN_DIR/${session_id}.csv"
log="$RUN_DIR/${session_id}.log"

BENCH_SIZE_FILTER=4096 \
BENCH_THREAD_FILTER="$THREAD_FILTER" \
env -u OMP_PROC_BIND \
    -u OMP_PLACES \
    -u GOMP_CPU_AFFINITY \
    OMP_DYNAMIC=FALSE \
    stdbuf -oL -eL \
    "$BIN" "$output" "$REPS" "$session_id" "$SEED" \
    2>&1 | tee "$log"

if grep -q 'OpenBLAS backend=sequential' "$log"; then
    echo "Quickcheck failed: sequential OpenBLAS backend detected" >&2
    exit 2
fi

IFS=',' read -r -a requested_threads <<< "$THREAD_FILTER"
for threads in "${requested_threads[@]}"; do
    if ! grep -Fq "[OpenBLAS] requested=${threads} active=${threads}" "$log"; then
        echo "Quickcheck failed: missing requested=active confirmation for ${threads} threads" >&2
        exit 2
    fi
done

expected=$((EXPECTED_THREADS * REPS))
python3 - "$output" "$expected" "$REPS" "$THREAD_FILTER" <<'PY'
import csv
import math
import statistics
import sys
from collections import Counter, defaultdict

path, expected_s, reps_s, thread_csv = sys.argv[1:]
expected = int(expected_s)
reps = int(reps_s)
expected_threads = {int(x) for x in thread_csv.split(',')}

with open(path, newline='', encoding='utf-8') as handle:
    reader = csv.DictReader(handle)
    rows = list(reader)
    fields = set(reader.fieldnames or [])

if len(rows) != expected:
    raise SystemExit(f"row-count mismatch: got {len(rows)}, expected {expected}")

required = {
    'workload', 'num_threads', 'problem_size', 'batches',
    'e2e_time_s', 'kernel_time_s', 'wall_time_s',
    'total_energy_j', 'gflops_per_s', 'checksum_ok'
}
missing = sorted(required - fields)
if missing:
    raise SystemExit(f"missing CSV columns: {missing}")

def finite_positive(row, key):
    try:
        value = float(row[key])
    except (TypeError, ValueError):
        raise SystemExit(f"invalid {key}: {row.get(key)!r}")
    if not math.isfinite(value) or value <= 0.0:
        raise SystemExit(f"non-positive/non-finite {key}: {value}")
    return value

def truthy(value):
    return str(value).strip().lower() in {'1', 't', 'true', 'yes'}

counts = Counter()
times = defaultdict(list)
for index, row in enumerate(rows, start=1):
    if row['workload'] != 'STRIDED_GEMM':
        raise SystemExit(f"row {index}: wrong workload {row['workload']!r}")
    if 'implementation' in fields and row['implementation'] != 'openblas_sgemm_ld2n':
        raise SystemExit(f"row {index}: wrong implementation {row['implementation']!r}")
    if 'execution_mode' in fields and row['execution_mode'] != 'cpu_native':
        raise SystemExit(f"row {index}: wrong execution_mode {row['execution_mode']!r}")

    threads = int(row['num_threads'])
    size = int(row['problem_size'])
    if size != 4096:
        raise SystemExit(f"row {index}: expected N=4096, got {size}")
    if threads not in expected_threads:
        raise SystemExit(f"row {index}: unexpected thread count {threads}")
    counts[threads] += 1

    batches = int(row['batches'])
    if batches <= 0:
        raise SystemExit(f"row {index}: batches must be positive")
    wall = finite_positive(row, 'wall_time_s')
    for key in ('e2e_time_s', 'kernel_time_s', 'total_energy_j', 'gflops_per_s'):
        finite_positive(row, key)
    if not truthy(row['checksum_ok']):
        raise SystemExit(f"row {index}: checksum failed")

    if 'time_per_op_ms_e2e' in fields:
        per_op = finite_positive(row, 'time_per_op_ms_e2e')
    else:
        per_op = 1000.0 * wall / batches
    times[threads].append(per_op)

    if 'problem_spec' in fields and row['problem_spec'] != 'N=4096;ld=8192':
        raise SystemExit(f"row {index}: wrong problem_spec {row['problem_spec']!r}")

for threads in expected_threads:
    if counts[threads] != reps:
        raise SystemExit(
            f"threads={threads}: got {counts[threads]} rows, expected {reps}")

medians = {threads: statistics.median(values) for threads, values in times.items()}
base = medians[1]
multithread = {threads: value for threads, value in medians.items() if threads != 1}
best_threads, best_time = min(multithread.items(), key=lambda item: item[1])
best_speedup = base / best_time

print('Median time per operation at N=4096:')
for threads in sorted(medians):
    print(f"  {threads:>2} threads: {medians[threads]:.6f} ms")
print(f"Best speedup: {best_speedup:.3f}x at {best_threads} threads")

if best_speedup < 1.25:
    raise SystemExit(
        f"scaling check failed: best multithread speedup is only {best_speedup:.3f}x (<1.25x)")

catastrophic = [
    (threads, value / base)
    for threads, value in multithread.items()
    if value >= 5.0 * base
]
if catastrophic:
    details = ', '.join(f"{t}T={ratio:.2f}x slower" for t, ratio in catastrophic)
    raise SystemExit(f"catastrophic threading regression: {details}")

warnings = [
    (threads, value / base)
    for threads, value in multithread.items()
    if value > 1.25 * base
]
for threads, ratio in warnings:
    print(
        f"WARNING: {threads} threads are {ratio:.2f}x slower than one thread",
        file=sys.stderr)

print(f"Quickcheck passed: {len(rows)} rows, checksums valid, threading plausible")
PY

echo "Quickcheck output: $output"

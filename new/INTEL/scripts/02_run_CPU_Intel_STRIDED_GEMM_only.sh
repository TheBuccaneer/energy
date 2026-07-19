#!/usr/bin/env bash
set -euo pipefail

PLATFORM="Intel"
PLATFORM_LC="intel"
THREAD_COUNT=7
POWER_OFF_AT_END=${POWER_OFF_AT_END:-1}
REPS=${REPS:-10}
SESSIONS=${SESSIONS:-5}
SEED_BASE=${SEED_BASE:-42000}

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)
SRC="$SCRIPT_DIR/STRIDED_GEMM/main_gemm_strided_${PLATFORM_LC}.cpp"
COMMON="$SCRIPT_DIR/common"
BUILD_DIR="$SCRIPT_DIR/STRIDED_GEMM/.build"
BIN="$BUILD_DIR/main_gemm_strided_${PLATFORM_LC}"
RUN_DIR="$ROOT_DIR/runs/STRIDED_GEMM"
ENABLE="$SCRIPT_DIR/01_enable_CPU_${PLATFORM}.sh"
RESTORE="$SCRIPT_DIR/03_disable_CPU_${PLATFORM}.sh"

SUCCESS=0
SETTINGS_TOUCHED=0
KEEPALIVE_PID=""

cleanup() {
    local status=$?
    trap - EXIT INT TERM

    if [[ -n "$KEEPALIVE_PID" ]]; then
        kill "$KEEPALIVE_PID" 2>/dev/null || true
        wait "$KEEPALIVE_PID" 2>/dev/null || true
    fi

    if (( SETTINGS_TOUCHED == 1 )); then
        if [[ -f "$RESTORE" ]]; then
            echo "[cleanup] Restoring ${PLATFORM} CPU settings..."
            if ! sudo bash "$RESTORE"; then
                echo "[cleanup] WARNING: restore script failed." >&2
                status=1
            fi
        else
            echo "[cleanup] WARNING: restore script not found: $RESTORE" >&2
            status=1
        fi
    fi

    if (( SUCCESS == 1 && status == 0 && POWER_OFF_AT_END == 1 )); then
        echo "[done] Five validated sessions completed. Powering off."
        sudo /sbin/poweroff
    elif (( SUCCESS == 1 && status == 0 )); then
        echo "[done] Five validated sessions completed. POWER_OFF_AT_END=0, staying online."
    else
        echo "[abort] Campaign incomplete or failed; no automatic power-off." >&2
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

command -v g++ >/dev/null || { echo "g++ not found" >&2; exit 2; }
command -v python3 >/dev/null || { echo "python3 not found" >&2; exit 2; }
command -v stdbuf >/dev/null || { echo "stdbuf not found" >&2; exit 2; }

mkdir -p "$BUILD_DIR" "$RUN_DIR"

echo "[build] Compiling without OpenMP..."
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

echo "[setup] Applying ${PLATFORM} CPU measurement settings..."
SETTINGS_TOUCHED=1
sudo bash "$ENABLE"

validate_session() {
    local csv=$1
    local expected=$2

    python3 - "$csv" "$expected" "$REPS" "1,2,4,8,10,16,20" <<'PY'
import csv
import math
import sys
from collections import Counter

path, expected_s, reps_s, thread_csv = sys.argv[1:]
expected = int(expected_s)
reps = int(reps_s)
expected_threads = {int(x) for x in thread_csv.split(',')}
expected_sizes = {64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384}

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

def positive(row, key):
    try:
        value = float(row[key])
    except (TypeError, ValueError):
        raise SystemExit(f"invalid {key}: {row.get(key)!r}")
    if not math.isfinite(value) or value <= 0.0:
        raise SystemExit(f"non-positive/non-finite {key}: {value}")

def truthy(value):
    return str(value).strip().lower() in {'1', 't', 'true', 'yes'}

counts = Counter()
for index, row in enumerate(rows, start=1):
    if row['workload'] != 'STRIDED_GEMM':
        raise SystemExit(f"row {index}: wrong workload {row['workload']!r}")
    if 'implementation' in fields and row['implementation'] != 'openblas_sgemm_ld2n':
        raise SystemExit(f"row {index}: wrong implementation {row['implementation']!r}")
    if 'execution_mode' in fields and row['execution_mode'] != 'cpu_native':
        raise SystemExit(f"row {index}: wrong execution_mode {row['execution_mode']!r}")

    threads = int(row['num_threads'])
    size = int(row['problem_size'])
    if threads not in expected_threads:
        raise SystemExit(f"row {index}: unexpected thread count {threads}")
    if size not in expected_sizes:
        raise SystemExit(f"row {index}: unexpected problem size {size}")
    counts[(size, threads)] += 1

    if int(row['batches']) <= 0:
        raise SystemExit(f"row {index}: batches must be positive")
    for key in ('e2e_time_s', 'kernel_time_s', 'wall_time_s', 'total_energy_j', 'gflops_per_s'):
        positive(row, key)
    if not truthy(row['checksum_ok']):
        raise SystemExit(f"row {index}: checksum failed")

    if 'runtime_status' in fields and row['runtime_status'] not in {'below', 'in_range', 'above'}:
        raise SystemExit(f"row {index}: invalid runtime_status {row['runtime_status']!r}")
    if 'problem_spec' in fields:
        expected_spec = f"N={size};ld={2 * size}"
        if row['problem_spec'] != expected_spec:
            raise SystemExit(
                f"row {index}: wrong problem_spec {row['problem_spec']!r}; expected {expected_spec!r}")

for size in expected_sizes:
    for threads in expected_threads:
        actual = counts[(size, threads)]
        if actual != reps:
            raise SystemExit(
                f"configuration N={size}, threads={threads}: got {actual} rows, expected {reps}")

print(f"validated {len(rows)} rows across {len(counts)} configurations")
PY
}

stamp=$(date +%Y%m%d_%H%M%S)
expected=$((9 * THREAD_COUNT * REPS))

for session in $(seq 1 "$SESSIONS"); do
    seed=$((SEED_BASE + session - 1))
    session_id="strided_gemm_${PLATFORM_LC}_${stamp}_session${session}"
    output="$RUN_DIR/${session_id}.csv"
    log="$RUN_DIR/${session_id}.log"

    echo "[run] Session ${session}/${SESSIONS}: seed=$seed"
    env -u OMP_PROC_BIND \
        -u OMP_PLACES \
        -u GOMP_CPU_AFFINITY \
        OMP_DYNAMIC=FALSE \
        stdbuf -oL -eL \
        "$BIN" "$output" "$REPS" "$session_id" "$seed" \
        2>&1 | tee "$log"

    validate_session "$output" "$expected"
    echo "[run] Session ${session}/${SESSIONS} validated: $output"
done

if (( SESSIONS != 5 )); then
    echo "Refusing success status: official campaign requires exactly five sessions, got $SESSIONS" >&2
    exit 2
fi

SUCCESS=1

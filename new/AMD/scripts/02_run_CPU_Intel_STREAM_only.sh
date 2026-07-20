#!/usr/bin/env bash
set -euo pipefail

PLATFORM="Intel"
PLATFORM_LC="intel"
ALL_THREADS="1,2,4,8,10,16,20"
THREAD_COUNT=7
QUICK_THREADS="1,10,20"

REPS=${REPS:-10}
SESSIONS=${SESSIONS:-5}
QUICK_REPS=${QUICK_REPS:-3}
SEED_BASE=${SEED_BASE:-20263100}
POWER_OFF_AT_END=${POWER_OFF_AT_END:-1}

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)

SRC="$SCRIPT_DIR/STREAM/CPU/INTEL/main_stream_intel.cpp"
COMMON="$SCRIPT_DIR/common"
BUILD_DIR="$SCRIPT_DIR/STREAM/CPU/INTEL/.build"
BIN="$BUILD_DIR/main_stream_intel"
RUN_DIR="$ROOT_DIR/runs/STREAM/CPU/INTEL"
RESTORE="$SCRIPT_DIR/03_disable_CPU_Intel.sh"

SIZES="1000000,2000000,4000000,8000000,16000000,32000000,64000000,128000000,256000000"
QUICK_SIZE="64000000"

SUCCESS=0
KEEPALIVE_PID=""

cleanup() {
    local status=$?
    trap - EXIT INT TERM

    if [[ -n "$KEEPALIVE_PID" ]]; then
        kill "$KEEPALIVE_PID" 2>/dev/null || true
        wait "$KEEPALIVE_PID" 2>/dev/null || true
    fi

    echo "[cleanup] Restoring ${PLATFORM} CPU settings and RAPL permissions..."
    if ! sudo bash "$RESTORE"; then
        echo "[cleanup] WARNING: restore script failed." >&2
        status=1
    fi

    if (( SUCCESS == 1 && status == 0 && POWER_OFF_AT_END == 1 )); then
        echo "[done] STREAM quickcheck and five validated sessions completed. Powering off."
        sudo systemctl poweroff
    elif (( SUCCESS == 1 && status == 0 )); then
        echo "[done] STREAM quickcheck and five validated sessions completed. POWER_OFF_AT_END=0; staying online."
    else
        echo "[abort] STREAM campaign incomplete or failed; no automatic power-off." >&2
    fi

    exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

for required in "$SRC" "$COMMON/benchmark_common.hpp" "$RESTORE"; do
    [[ -f "$required" ]] || {
        echo "ERROR: missing required file: $required" >&2
        exit 2
    }
done

for command in g++ python3 stdbuf tee ldd; do
    command -v "$command" >/dev/null || {
        echo "ERROR: required command not found: $command" >&2
        exit 2
    }
done

if (( SESSIONS != 5 )); then
    echo "ERROR: official STREAM campaign requires exactly 5 sessions; got $SESSIONS." >&2
    exit 2
fi
if (( REPS != 10 )); then
    echo "ERROR: official STREAM campaign requires exactly 10 repetitions; got $REPS." >&2
    exit 2
fi

mkdir -p "$BUILD_DIR" "$RUN_DIR"

echo "[build] Compiling ${PLATFORM} STREAM with OpenMP..."
g++ -O3 -march=native -std=c++17 -fopenmp \
    -I"$COMMON" \
    "$SRC" \
    -lpthread -lm \
    -o "$BIN"

if ! ldd "$BIN" | grep -q 'libgomp'; then
    echo "ERROR: OpenMP runtime libgomp is not linked." >&2
    exit 2
fi
if ldd "$BIN" | grep -qi 'openblas'; then
    echo "ERROR: STREAM must not link against OpenBLAS." >&2
    exit 2
fi

echo "[preflight] Build OK; OpenMP linked; OpenBLAS absent."

sudo -v
(
    while true; do
        sudo -n true 2>/dev/null || exit
        sleep 60
        kill -0 "$$" 2>/dev/null || exit
    done
) &
KEEPALIVE_PID=$!

validate_csv() {
    local csv_path=$1
    local expected_sizes=$2
    local expected_threads=$3
    local expected_reps=$4

    python3 - "$csv_path" "$expected_sizes" "$expected_threads" "$expected_reps" <<'PY'
import csv
import math
import sys
from collections import Counter, defaultdict

path, sizes_csv, threads_csv, reps_s = sys.argv[1:]
expected_sizes = {int(value) for value in sizes_csv.split(",") if value}
expected_threads = {int(value) for value in threads_csv.split(",") if value}
expected_reps = int(reps_s)
expected_rows = len(expected_sizes) * len(expected_threads) * expected_reps

with open(path, newline="", encoding="utf-8") as handle:
    reader = csv.DictReader(handle)
    rows = list(reader)
    fieldnames = reader.fieldnames or []
    fields = set(fieldnames)

if len(rows) != expected_rows:
    raise SystemExit(
        f"row-count mismatch: got {len(rows)}, expected {expected_rows}"
    )

required = {
    "schema_version", "workload", "implementation", "execution_mode",
    "device_name", "num_threads", "problem_size", "problem_spec",
    "batches", "e2e_time_s", "kernel_time_s", "wall_time_s",
    "total_energy_j", "energy_per_op_j", "time_per_op_ms_kernel",
    "time_per_op_ms_e2e", "flops_total", "gflops_per_s",
    "logical_bytes_per_op", "avg_power_w", "runtime_status",
    "repetition", "checksum_ok",
}
missing = sorted(required - fields)
if missing:
    raise SystemExit(f"missing CSV columns: {missing}")

def finite_float(row, key):
    try:
        value = float(row[key])
    except (TypeError, ValueError):
        raise SystemExit(f"invalid {key}: {row.get(key)!r}")
    if not math.isfinite(value):
        raise SystemExit(f"non-finite {key}: {value}")
    return value

def positive(row, key):
    value = finite_float(row, key)
    if value <= 0.0:
        raise SystemExit(f"non-positive {key}: {value}")
    return value

def truthy(value):
    return str(value).strip().lower() in {"1", "t", "true", "yes"}

def close(actual, expected, *, rel=2e-5, abs_=1e-9):
    return math.isclose(actual, expected, rel_tol=rel, abs_tol=abs_)

counts = Counter()
repetitions = defaultdict(set)
schemas = set()

for index, row in enumerate(rows, start=1):
    schemas.add(row["schema_version"].strip())
    if row["workload"] != "STREAM":
        raise SystemExit(f"row {index}: wrong workload {row['workload']!r}")
    if row["implementation"] != "openmp_triad":
        raise SystemExit(
            f"row {index}: wrong implementation {row['implementation']!r}"
        )
    if row["execution_mode"] != "cpu_native":
        raise SystemExit(
            f"row {index}: wrong execution_mode {row['execution_mode']!r}"
        )
    if not row["device_name"].strip():
        raise SystemExit(f"row {index}: empty device_name")

    threads = int(row["num_threads"])
    size = int(row["problem_size"])
    repetition = int(row["repetition"])
    batches = int(row["batches"])

    if threads not in expected_threads:
        raise SystemExit(f"row {index}: unexpected thread count {threads}")
    if size not in expected_sizes:
        raise SystemExit(f"row {index}: unexpected problem size {size}")
    if repetition < 1 or repetition > expected_reps:
        raise SystemExit(f"row {index}: invalid repetition {repetition}")
    if batches <= 0:
        raise SystemExit(f"row {index}: batches must be positive")

    expected_spec = f"elements={size}"
    if row["problem_spec"] != expected_spec:
        raise SystemExit(
            f"row {index}: wrong problem_spec {row['problem_spec']!r}; "
            f"expected {expected_spec!r}"
        )

    e2e = positive(row, "e2e_time_s")
    kernel = positive(row, "kernel_time_s")
    wall = positive(row, "wall_time_s")
    energy = positive(row, "total_energy_j")
    energy_per_op = positive(row, "energy_per_op_j")
    time_kernel = positive(row, "time_per_op_ms_kernel")
    time_e2e = positive(row, "time_per_op_ms_e2e")
    flops_total = positive(row, "flops_total")
    gflops = positive(row, "gflops_per_s")
    logical_bytes = positive(row, "logical_bytes_per_op")
    avg_power = positive(row, "avg_power_w")

    if kernel > e2e * (1.0 + 2e-5):
        raise SystemExit(
            f"row {index}: kernel_time_s exceeds e2e_time_s "
            f"({kernel} > {e2e})"
        )

    expected_flops = 2.0 * size * batches
    expected_bytes = 12.0 * size

    if not close(flops_total, expected_flops):
        raise SystemExit(
            f"row {index}: flops_total={flops_total}, expected {expected_flops}"
        )
    if not close(logical_bytes, expected_bytes):
        raise SystemExit(
            f"row {index}: logical_bytes_per_op={logical_bytes}, "
            f"expected {expected_bytes}"
        )
    if not close(energy_per_op, energy / batches):
        raise SystemExit(f"row {index}: energy_per_op_j identity failed")
    if not close(time_kernel, kernel * 1000.0 / batches):
        raise SystemExit(f"row {index}: kernel time/op identity failed")
    if not close(time_e2e, e2e * 1000.0 / batches):
        raise SystemExit(f"row {index}: e2e time/op identity failed")
    if not close(gflops, flops_total / kernel / 1.0e9):
        raise SystemExit(f"row {index}: GFLOP/s identity failed")
    if not close(avg_power, energy / wall):
        raise SystemExit(f"row {index}: average-power identity failed")

    if row["runtime_status"] not in {"below", "in_range", "above"}:
        raise SystemExit(
            f"row {index}: invalid runtime_status "
            f"{row['runtime_status']!r}"
        )
    if not truthy(row["checksum_ok"]):
        raise SystemExit(f"row {index}: checksum failed")

    if "dram_energy_j" in fields:
        dram = finite_float(row, "dram_energy_j")
        if dram < 0.0 and dram != -1.0:
            raise SystemExit(
                f"row {index}: invalid dram_energy_j sentinel {dram}"
            )

    counts[(size, threads)] += 1
    repetitions[(size, threads)].add(repetition)

if len(schemas) != 1 or not next(iter(schemas), "").startswith("cpu-gpu-v"):
    raise SystemExit(f"unexpected schema_version values: {sorted(schemas)}")

for size in expected_sizes:
    for threads in expected_threads:
        key = (size, threads)
        if counts[key] != expected_reps:
            raise SystemExit(
                f"configuration size={size}, threads={threads}: "
                f"got {counts[key]} rows, expected {expected_reps}"
            )
        if repetitions[key] != set(range(1, expected_reps + 1)):
            raise SystemExit(
                f"configuration size={size}, threads={threads}: "
                f"bad repetition set {sorted(repetitions[key])}"
            )

status_counts = Counter(row["runtime_status"] for row in rows)
print(
    f"validated {len(rows)} rows, {len(counts)} configurations, "
    f"schema={next(iter(schemas))}, runtime_status={dict(status_counts)}"
)
PY
}

run_stream() {
    local output=$1
    local log=$2
    local reps=$3
    local session_id=$4
    local seed=$5
    local size_filter=$6
    local thread_filter=$7

    if [[ -n "$size_filter" && -n "$thread_filter" ]]; then
        env -u GOMP_CPU_AFFINITY \
            OMP_DYNAMIC=FALSE \
            OMP_PROC_BIND=spread \
            OMP_PLACES=cores \
            BENCH_SIZE_FILTER="$size_filter" \
            BENCH_THREAD_FILTER="$thread_filter" \
            stdbuf -oL -eL \
            "$BIN" "$output" "$reps" "$session_id" "$seed" \
            2>&1 | tee "$log"
    else
        env -u GOMP_CPU_AFFINITY \
            -u BENCH_SIZE_FILTER \
            -u BENCH_THREAD_FILTER \
            OMP_DYNAMIC=FALSE \
            OMP_PROC_BIND=spread \
            OMP_PLACES=cores \
            stdbuf -oL -eL \
            "$BIN" "$output" "$reps" "$session_id" "$seed" \
            2>&1 | tee "$log"
    fi
}

stamp=$(date +%Y%m%d_%H%M%S)

quick_id="stream_${PLATFORM_LC}_${stamp}_quickcheck"
quick_csv="$RUN_DIR/${quick_id}.csv"
quick_log="$RUN_DIR/${quick_id}.log"
quick_seed=$((SEED_BASE + 900))

echo "[quickcheck] N=${QUICK_SIZE}; threads=${QUICK_THREADS}; reps=${QUICK_REPS}"
run_stream \
    "$quick_csv" "$quick_log" "$QUICK_REPS" "$quick_id" "$quick_seed" \
    "$QUICK_SIZE" "$QUICK_THREADS"
validate_csv "$quick_csv" "$QUICK_SIZE" "$QUICK_THREADS" "$QUICK_REPS"
echo "[quickcheck] PASS: $quick_csv"

for ((session=1; session<=SESSIONS; session++)); do
    seed=$((SEED_BASE + session - 1))
    session_id="stream_${PLATFORM_LC}_${stamp}_session${session}"
    output="$RUN_DIR/${session_id}.csv"
    log="$RUN_DIR/${session_id}.log"

    echo "[run] ${PLATFORM} STREAM session ${session}/${SESSIONS}; seed=${seed}; reps=${REPS}"
    run_stream "$output" "$log" "$REPS" "$session_id" "$seed" "" ""
    validate_csv "$output" "$SIZES" "$ALL_THREADS" "$REPS"
    echo "[run] Session ${session}/${SESSIONS} PASS: $output"
done

echo "[done] ${PLATFORM} STREAM: integrated quickcheck plus ${SESSIONS} validated sessions completed."
echo "[done] Official measurements per size×thread configuration: $((SESSIONS * REPS))."
SUCCESS=1

#!/usr/bin/env bash
set -euo pipefail

PLATFORM="Intel"
PLATFORM_LC="intel"
THREADS="1,10,20"
SIZE=${SIZE:-64000000}
REPS=${REPS:-3}
SEED=${SEED:-20263991}

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)

SRC="$SCRIPT_DIR/STREAM/main_stream_intel.cpp"
COMMON="$SCRIPT_DIR/common"
BUILD_DIR="$SCRIPT_DIR/STREAM/.build"
BIN="$BUILD_DIR/main_stream_intel"
RUN_DIR="$ROOT_DIR/runs/STREAM"
OUT="$RUN_DIR/stream_${PLATFORM_LC}_quickcheck.csv"
LOG="$RUN_DIR/stream_${PLATFORM_LC}_quickcheck.log"

for required in "$SRC" "$COMMON/benchmark_common.hpp"; do
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

mkdir -p "$BUILD_DIR" "$RUN_DIR"
rm -f "$OUT" "$LOG"

echo "[build] Compiling ${PLATFORM} STREAM quickcheck with OpenMP..."
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

echo "[run] ${PLATFORM} STREAM quickcheck | N=${SIZE} | threads=${THREADS} | reps=${REPS}"

env -u GOMP_CPU_AFFINITY \
    OMP_DYNAMIC=FALSE \
    OMP_PROC_BIND=spread \
    OMP_PLACES=cores \
    BENCH_SIZE_FILTER="$SIZE" \
    BENCH_THREAD_FILTER="$THREADS" \
    stdbuf -oL -eL \
    "$BIN" "$OUT" "$REPS" "quickcheck_stream_${PLATFORM_LC}" "$SEED" \
    2>&1 | tee "$LOG"

python3 - "$OUT" "$SIZE" "$THREADS" "$REPS" <<'PY'
import csv
import math
import statistics
import sys
from collections import Counter, defaultdict

path, size_s, threads_s, reps_s = sys.argv[1:]
expected_size = int(size_s)
expected_threads = [int(value) for value in threads_s.split(",") if value]
expected_reps = int(reps_s)
expected_rows = len(expected_threads) * expected_reps

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
repetition_sets = defaultdict(set)
kernel_by_thread = defaultdict(list)
energy_by_thread = defaultdict(list)

for index, row in enumerate(rows, start=1):
    if not row["schema_version"].startswith("cpu-gpu-v"):
        raise SystemExit(
            f"row {index}: unexpected schema_version "
            f"{row['schema_version']!r}"
        )
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
    if size != expected_size:
        raise SystemExit(f"row {index}: unexpected problem size {size}")
    if repetition < 1 or repetition > expected_reps:
        raise SystemExit(f"row {index}: invalid repetition {repetition}")
    if batches <= 0:
        raise SystemExit(f"row {index}: batches must be positive")
    if row["problem_spec"] != f"elements={size}":
        raise SystemExit(
            f"row {index}: wrong problem_spec {row['problem_spec']!r}"
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
            f"row {index}: kernel_time_s exceeds e2e_time_s"
        )

    expected_flops = 2.0 * size * batches
    expected_bytes = 12.0 * size

    strict_checks = [
        ("flops_total", flops_total, expected_flops),
        ("logical_bytes_per_op", logical_bytes, expected_bytes),
        ("energy_per_op_j", energy_per_op, energy / batches),
        ("time_per_op_ms_kernel", time_kernel, kernel * 1000.0 / batches),
        ("time_per_op_ms_e2e", time_e2e, e2e * 1000.0 / batches),
    ]
    for name, actual, expected in strict_checks:
        if not close(actual, expected):
            raise SystemExit(
                f"row {index}: {name} identity failed: "
                f"{actual} vs {expected}"
            )

    # These two CSV fields are intentionally serialized with lower precision.
    if not math.isclose(
        gflops,
        flops_total / kernel / 1.0e9,
        rel_tol=2.0e-3,
        abs_tol=1.1e-2,
    ):
        raise SystemExit(
            f"row {index}: gflops_per_s identity failed after "
            f"rounding allowance: {gflops} vs "
            f"{flops_total / kernel / 1.0e9}"
        )

    if not math.isclose(
        avg_power,
        energy / wall,
        rel_tol=2.0e-3,
        abs_tol=1.1e-1,
    ):
        raise SystemExit(
            f"row {index}: avg_power_w identity failed after "
            f"rounding allowance: {avg_power} vs {energy / wall}"
        )

    if row["runtime_status"] not in {"below", "in_range", "above"}:
        raise SystemExit(
            f"row {index}: invalid runtime_status "
            f"{row['runtime_status']!r}"
        )
    if not truthy(row["checksum_ok"]):
        raise SystemExit(f"row {index}: checksum failed")

    counts[threads] += 1
    repetition_sets[threads].add(repetition)
    kernel_by_thread[threads].append(kernel / batches)
    energy_by_thread[threads].append(energy / batches)

for threads in expected_threads:
    if counts[threads] != expected_reps:
        raise SystemExit(
            f"threads={threads}: got {counts[threads]} rows, "
            f"expected {expected_reps}"
        )
    if repetition_sets[threads] != set(range(1, expected_reps + 1)):
        raise SystemExit(
            f"threads={threads}: invalid repetition set "
            f"{sorted(repetition_sets[threads])}"
        )

if 1 in expected_threads and len(expected_threads) > 1:
    one_thread = statistics.median(kernel_by_thread[1])
    best_mt_thread = min(
        (thread for thread in expected_threads if thread != 1),
        key=lambda thread: statistics.median(kernel_by_thread[thread]),
    )
    best_mt = statistics.median(kernel_by_thread[best_mt_thread])
    speedup = one_thread / best_mt

    if speedup < 1.25:
        raise SystemExit(
            f"implausible STREAM scaling: best multithread speedup "
            f"is only {speedup:.2f}x (thread={best_mt_thread})"
        )
else:
    speedup = float("nan")
    best_mt_thread = None

print(f"validated {len(rows)} rows")
for threads in expected_threads:
    median_time = statistics.median(kernel_by_thread[threads])
    median_energy = statistics.median(energy_by_thread[threads])
    logical_bw = (12.0 * expected_size) / median_time / 1.0e9
    print(
        f"threads={threads:>2}: "
        f"median_time/op={median_time*1000.0:.3f} ms, "
        f"logical_BW={logical_bw:.1f} GB/s, "
        f"median_energy/op={median_energy:.6f} J"
    )
if best_mt_thread is not None:
    print(
        f"best multithread speedup vs 1T: "
        f"{speedup:.2f}x at {best_mt_thread} threads"
    )
PY

echo "Quickcheck PASS: $OUT"

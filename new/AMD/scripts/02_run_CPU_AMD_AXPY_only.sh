#!/usr/bin/env bash
set -euo pipefail

PLATFORM="AMD"
PLATFORM_LC="amd"

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)
SRC="$SCRIPT_DIR/AXPY/main_axpy_amd.cpp"
COMMON="$SCRIPT_DIR/common"
HEADER="$COMMON/benchmark_common.hpp"
BUILD_DIR="$SCRIPT_DIR/AXPY/.build"
BIN="$BUILD_DIR/main_axpy_amd"
RUN_DIR="$ROOT_DIR/runs/AXPY"
RESTORE="$SCRIPT_DIR/03_disable_CPU_AMD.sh"

SUCCESS=0
KEEPALIVE_PID=""

die() {
    echo "ERROR: $*" >&2
    exit 2
}

cleanup() {
    local status=$?
    trap - EXIT INT TERM

    if [[ -n "$KEEPALIVE_PID" ]]; then
        kill "$KEEPALIVE_PID" 2>/dev/null || true
        wait "$KEEPALIVE_PID" 2>/dev/null || true
    fi

    echo "[cleanup] Restoring AMD CPU settings and RAPL permissions..."
    if ! sudo bash "$RESTORE"; then
        echo "[cleanup] WARNING: restore failed." >&2
        status=1
    fi

    finish_action "$status"
    exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

for required in "$SRC" "$HEADER" "$RESTORE"; do
    [[ -f "$required" ]] || die "missing required file: $required"
done
for command in g++ python3 stdbuf tee ldd sha256sum; do
    command -v "$command" >/dev/null || die "required command not found: $command"
done

mkdir -p "$BUILD_DIR" "$RUN_DIR"

echo "[build] Compiling AMD AXPY with OpenMP..."
g++ -O3 -march=native -std=c++17 -fopenmp \
    -I"$COMMON" "$SRC" -lpthread -lm -o "$BIN"

ldd "$BIN" | grep -q 'libgomp' || die "libgomp is not linked"
if ldd "$BIN" | grep -qi openblas; then
    die "AXPY must not link against OpenBLAS"
fi

echo "[preflight] Build OK; OpenMP linked; OpenBLAS absent."
echo "[preflight] source_sha256=$(sha256sum "$SRC" | awk '{print $1}')"
echo "[preflight] header_sha256=$(sha256sum "$HEADER" | awk '{print $1}')"
echo "[preflight] runner_sha256=$(sha256sum "${BASH_SOURCE[0]}" | awk '{print $1}')"
echo "[preflight] binary_sha256=$(sha256sum "$BIN" | awk '{print $1}')"

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
    local sizes=$2
    local threads=$3
    local reps=$4
    local session_id=$5

    python3 - "$csv_path" "$sizes" "$threads" "$reps" "$session_id" <<'PYVALID'

import csv, math, sys
from collections import Counter, defaultdict

path, sizes_s, threads_s, reps_s, expected_session = sys.argv[1:]
sizes = {int(x) for x in sizes_s.split(",") if x}
threads = {int(x) for x in threads_s.split(",") if x}
reps = int(reps_s)
expected_rows = len(sizes) * len(threads) * reps

header = [
    "schema_version","timestamp","session_id","sequence_index","run_id_global",
    "repetition","workload","implementation","execution_mode","device_name",
    "num_threads","problem_size","problem_spec","batches","e2e_time_s",
    "kernel_time_s","wall_time_s","device_energy_j","total_energy_j",
    "dram_energy_j","energy_per_op_j","energy_per_second_j",
    "energy_per_flop_j","time_per_op_ms_kernel","time_per_op_ms_e2e",
    "flops_total","gflops_per_s","logical_bytes_per_op","avg_power_w",
    "runtime_status","pcie_gen","pcie_width","sm_clock_mhz",
    "clock_before_mhz","clock_after_mhz","mem_clock_mhz","temp_c",
    "temp_before_c","temp_after_c","throttle_reasons","cpu_cycles",
    "cpu_instructions","cpu_ipc","cpu_cache_misses","checksum_ok",
]

with open(path, newline="", encoding="utf-8") as f:
    rd = csv.DictReader(f)
    rows = list(rd)
    if rd.fieldnames != header:
        raise SystemExit(f"header mismatch: {rd.fieldnames}")

if len(rows) != expected_rows:
    raise SystemExit(f"row-count mismatch: got {len(rows)}, expected {expected_rows}")

def num(r, k):
    try:
        v = float(r[k])
    except Exception:
        raise SystemExit(f"invalid {k}: {r.get(k)!r}")
    if not math.isfinite(v):
        raise SystemExit(f"non-finite {k}: {v}")
    return v

def pos(r, k):
    v = num(r, k)
    if v <= 0:
        raise SystemExit(f"non-positive {k}: {v}")
    return v

def integer(r, k):
    try:
        return int(r[k])
    except Exception:
        raise SystemExit(f"invalid integer {k}: {r.get(k)!r}")

def close(a, b):
    return math.isclose(a, b, rel_tol=2e-12, abs_tol=1e-15)

def truth(v):
    return str(v).strip().lower() in {"1","t","true","yes"}

counts = Counter()
seen = defaultdict(set)
seqs, runids = [], []

for i, r in enumerate(rows, 1):
    if r["schema_version"] != "cpu-gpu-v2":
        raise SystemExit(f"row {i}: wrong schema")
    if r["session_id"] != expected_session:
        raise SystemExit(f"row {i}: wrong session")
    if r["workload"] != "AXPY":
        raise SystemExit(f"row {i}: wrong workload")
    if r["implementation"] != "openmp_axpy_inplace_fp32":
        raise SystemExit(f"row {i}: wrong implementation")
    if r["execution_mode"] != "cpu_native":
        raise SystemExit(f"row {i}: wrong mode")

    n = integer(r, "problem_size")
    t = integer(r, "num_threads")
    rep = integer(r, "repetition")
    b = integer(r, "batches")
    seq = integer(r, "sequence_index")
    rid = integer(r, "run_id_global")

    if n not in sizes or t not in threads or not (1 <= rep <= reps):
        raise SystemExit(f"row {i}: unexpected configuration")
    if not (1 <= b <= 250000):
        raise SystemExit(f"row {i}: invalid batches={b}")

    expected_spec = (
        f"elements={n};alpha=3.0;x=period29*2^-16;"
        "y0=period31*2^-8;reset=outside_window;max_batches=250000"
    )
    if r["problem_spec"] != expected_spec:
        raise SystemExit(f"row {i}: bad problem_spec")

    e2e = pos(r, "e2e_time_s")
    ker = pos(r, "kernel_time_s")
    wall = pos(r, "wall_time_s")
    dev = pos(r, "device_energy_j")
    total = pos(r, "total_energy_j")
    dram = num(r, "dram_energy_j")

    if not close(ker, e2e) or not close(wall, e2e):
        raise SystemExit(f"row {i}: CPU time identity failed")
    if dram != -1.0:
        raise SystemExit(f"row {i}: AMD dram_energy_j must be -1")
    if not close(total, dev):
        raise SystemExit(f"row {i}: AMD total_energy_j must equal device_energy_j")

    flops = integer(r, "flops_total")
    bytes_ = integer(r, "logical_bytes_per_op")
    if flops != 2 * n * b:
        raise SystemExit(f"row {i}: bad flops_total")
    if bytes_ != 12 * n:
        raise SystemExit(f"row {i}: bad logical_bytes_per_op")

    formulas = [
        ("energy_per_op_j", num(r,"energy_per_op_j"), dev / b),
        ("energy_per_second_j", num(r,"energy_per_second_j"), dev / wall),
        ("energy_per_flop_j", num(r,"energy_per_flop_j"), dev / flops),
        ("time_per_op_ms_kernel", num(r,"time_per_op_ms_kernel"), 1000 * ker / b),
        ("time_per_op_ms_e2e", num(r,"time_per_op_ms_e2e"), 1000 * e2e / b),
        ("gflops_per_s", num(r,"gflops_per_s"), flops / ker / 1e9),
        ("avg_power_w", num(r,"avg_power_w"), dev / wall),
    ]
    for name, actual, expected in formulas:
        if not close(actual, expected):
            raise SystemExit(f"row {i}: {name} identity failed")

    status = "below" if e2e < .75 else ("in_range" if e2e <= 1.25 else "above")
    if r["runtime_status"] != status:
        raise SystemExit(f"row {i}: bad runtime_status")
    if status == "below":
        raise SystemExit(f"row {i}: forbidden below row")

    for key in ("pcie_gen","pcie_width","sm_clock_mhz","mem_clock_mhz",
                "cpu_cycles","cpu_instructions","cpu_cache_misses"):
        if num(r, key) != -1.0:
            raise SystemExit(f"row {i}: {key} must be -1")
    if num(r, "cpu_ipc") != -1.0:
        raise SystemExit(f"row {i}: cpu_ipc must be -1")
    if not truth(r["checksum_ok"]):
        raise SystemExit(f"row {i}: checksum failed")

    counts[(n,t)] += 1
    seen[(n,t)].add(rep)
    seqs.append(seq)
    runids.append(rid)

if seqs != list(range(1, expected_rows + 1)):
    raise SystemExit("sequence_index not contiguous")
if runids != seqs:
    raise SystemExit("run_id_global mismatch")

for n in sizes:
    for t in threads:
        if counts[(n,t)] != reps:
            raise SystemExit(f"coverage failed N={n}, threads={t}")
        if seen[(n,t)] != set(range(1, reps + 1)):
            raise SystemExit(f"repetition coverage failed N={n}, threads={t}")

print(f"validated {len(rows)} AXPY rows across {len(counts)} configurations")

PYVALID
}

run_axpy() {
    local output=$1
    local log=$2
    local reps=$3
    local session_id=$4
    local seed=$5
    local size_filter=$6
    local thread_filter=$7
    local anti_collapse=$8

    if (( anti_collapse == 1 )); then
        env \
            -u GOMP_CPU_AFFINITY \
            OMP_DYNAMIC=FALSE \
            OMP_PROC_BIND=spread \
            OMP_PLACES=cores \
            BENCH_SIZE_FILTER="$size_filter" \
            BENCH_THREAD_FILTER="$thread_filter" \
            AXPY_ANTI_COLLAPSE_PROBE=1 \
            stdbuf -oL -eL \
            "$BIN" "$output" "$reps" "$session_id" "$seed" \
            2>&1 | tee "$log"
    elif [[ -n "$size_filter" ]]; then
        env \
            -u GOMP_CPU_AFFINITY \
            -u AXPY_ANTI_COLLAPSE_PROBE \
            OMP_DYNAMIC=FALSE \
            OMP_PROC_BIND=spread \
            OMP_PLACES=cores \
            BENCH_SIZE_FILTER="$size_filter" \
            BENCH_THREAD_FILTER="$thread_filter" \
            stdbuf -oL -eL \
            "$BIN" "$output" "$reps" "$session_id" "$seed" \
            2>&1 | tee "$log"
    else
        env \
            -u GOMP_CPU_AFFINITY \
            -u BENCH_SIZE_FILTER \
            -u BENCH_THREAD_FILTER \
            -u AXPY_ANTI_COLLAPSE_PROBE \
            OMP_DYNAMIC=FALSE \
            OMP_PROC_BIND=spread \
            OMP_PLACES=cores \
            stdbuf -oL -eL \
            "$BIN" "$output" "$reps" "$session_id" "$seed" \
            2>&1 | tee "$log"
    fi
}

ALL_THREADS="1,2,4,8,10,16,20,32,64"
SIZES="1000000,2000000,4000000,8000000,16000000,32000000,64000000,128000000,256000000"

REPS=${REPS:-10}
SESSIONS=${SESSIONS:-5}
POWER_OFF_AT_END=${POWER_OFF_AT_END:-1}
SESSION_PAUSE_S=${SESSION_PAUSE_S:-60}
SEED_BASE=$((0x4158505900001000))

finish_action() {
    local status=$1
    if (( SUCCESS == 1 && status == 0 && POWER_OFF_AT_END == 1 )); then
        echo "[done] Five validated AMD AXPY sessions completed. Powering off."
        sudo systemctl poweroff
    elif (( SUCCESS == 1 && status == 0 )); then
        echo "[done] Five validated AMD AXPY sessions completed. Staying online."
    else
        echo "[abort] AMD AXPY campaign incomplete or failed; no power-off." >&2
    fi
}

[[ "$SESSIONS" -eq 5 ]] || die "official campaign requires SESSIONS=5"
[[ "$REPS" -eq 10 ]] || die "official campaign requires REPS=10"
[[ "$POWER_OFF_AT_END" =~ ^[01]$ ]] || die "POWER_OFF_AT_END must be 0 or 1"

stamp=$(date +%Y%m%d_%H%M%S)

echo "[run] Starting FULL AMD AXPY campaign directly."
echo "[run] No quickcheck/prerun is executed by this 02 runner."

for ((session=1; session<=SESSIONS; session++)); do
    seed=$((SEED_BASE + session))
    session_id="axpy_amd_${stamp}_session${session}"
    csv="$RUN_DIR/${session_id}.csv"
    log="$RUN_DIR/${session_id}.log"

    echo "[run] AMD AXPY session ${session}/${SESSIONS}; seed=${seed}; reps=${REPS}"
    run_axpy "$csv" "$log" "$REPS" "$session_id" "$seed" "" "" 0
    validate_csv "$csv" "$SIZES" "$ALL_THREADS" "$REPS" "$session_id"
    echo "[run] Session ${session}/${SESSIONS} PASS: $csv"

    if (( session < SESSIONS )); then
        echo "[pause] ${SESSION_PAUSE_S}s"
        sleep "$SESSION_PAUSE_S"
    fi
done

echo "[done] 5 sessions × 810 rows = 4050 official AMD AXPY rows."
SUCCESS=1

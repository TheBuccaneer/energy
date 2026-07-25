#!/usr/bin/env bash
set -euo pipefail

PLATFORM="Intel"
PLATFORM_LC="intel"

ALL_THREADS="1,2,4,8,10,16,20"
QUICK_THREADS="1,20"

SIZES="1000000,2000000,4000000,8000000,16000000,32000000,64000000,128000000,256000000"
QUICK_SIZES="1000000,64000000,256000000"

REPS=${REPS:-10}
SESSIONS=${SESSIONS:-5}
QUICK_REPS=${QUICK_REPS:-2}
POWER_OFF_AT_END=${POWER_OFF_AT_END:-1}
QUICKCHECK_ONLY=${QUICKCHECK_ONLY:-0}

# Frozen AXPY seed rule: 0x41585059 == ASCII "AXPY".
SEED_BASE=$((0x4158505900000000))
QUICK_SEED=$((0x4158505900000900))
SESSION_PAUSE_S=60

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)

SRC="$SCRIPT_DIR/AXPY/main_axpy_${PLATFORM_LC}.cpp"
COMMON="$SCRIPT_DIR/common"
HEADER="$COMMON/benchmark_common.hpp"
BUILD_DIR="$SCRIPT_DIR/AXPY/.build"
BIN="$BUILD_DIR/main_axpy_${PLATFORM_LC}"
RUN_DIR="$ROOT_DIR/runs/AXPY"
RESTORE="$SCRIPT_DIR/03_disable_CPU_${PLATFORM}.sh"

GOV_STATE=/tmp/energy_intel_governors.state
RAPL_STATE=/tmp/energy_intel_rapl_modes.state

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

    echo "[cleanup] Restoring ${PLATFORM} CPU settings and RAPL permissions..."
    if ! sudo bash "$RESTORE"; then
        echo "[cleanup] WARNING: restore script failed." >&2
        status=1
    fi

    if (( SUCCESS == 1 && status == 0 && QUICKCHECK_ONLY == 0 && POWER_OFF_AT_END == 1 )); then
        echo "[done] AXPY quickcheck and five validated sessions completed. Powering off."
        sudo systemctl poweroff
    elif (( SUCCESS == 1 && status == 0 )); then
        if (( QUICKCHECK_ONLY == 1 )); then
            echo "[done] AXPY quickcheck completed. No official session was started."
        else
            echo "[done] AXPY campaign completed. POWER_OFF_AT_END=0; staying online."
        fi
    else
        echo "[abort] AXPY run incomplete or failed; no automatic power-off." >&2
    fi

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

[[ "$SESSIONS" -eq 5 ]] || die "official AXPY campaign requires SESSIONS=5; got $SESSIONS"
[[ "$REPS" -eq 10 ]] || die "official AXPY campaign requires REPS=10; got $REPS"
[[ "$QUICK_REPS" -eq 2 ]] || die "AXPY quickcheck requires QUICK_REPS=2; got $QUICK_REPS"
[[ "$QUICKCHECK_ONLY" =~ ^[01]$ ]] || die "QUICKCHECK_ONLY must be 0 or 1"
[[ "$POWER_OFF_AT_END" =~ ^[01]$ ]] || die "POWER_OFF_AT_END must be 0 or 1"

# The project uses an explicit enable step. Do not silently change it.
if [[ ! -e "$GOV_STATE" || ! -e "$RAPL_STATE" ]]; then
    die "Intel measurement state is not active. Run: sudo bash \"$SCRIPT_DIR/01_enable_CPU_Intel.sh\""
fi

mkdir -p "$BUILD_DIR" "$RUN_DIR"

COMPILE_CMD=(
    g++
    -O3
    -march=native
    -std=c++17
    -fopenmp
    -I"$COMMON"
    "$SRC"
    -lpthread
    -lm
    -o "$BIN"
)

echo "[build] Compiling ${PLATFORM} AXPY with OpenMP..."
printf '[build] command:'
printf ' %q' "${COMPILE_CMD[@]}"
printf '\n'
"${COMPILE_CMD[@]}"

ldd "$BIN" | grep -q 'libgomp' || die "OpenMP runtime libgomp is not linked"
if ldd "$BIN" | grep -qi 'openblas'; then
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
    local expected_sizes=$2
    local expected_threads=$3
    local expected_reps=$4
    local expected_session_id=$5

    python3 - "$csv_path" "$expected_sizes" "$expected_threads" "$expected_reps" "$expected_session_id" <<'PYVALID'
import csv
import math
import sys
from collections import Counter, defaultdict

path, sizes_s, threads_s, reps_s, expected_session_id = sys.argv[1:]
sizes = {int(value) for value in sizes_s.split(",") if value}
threads = {int(value) for value in threads_s.split(",") if value}
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

with open(path, newline="", encoding="utf-8") as handle:
    reader = csv.DictReader(handle)
    rows = list(reader)
    if reader.fieldnames != header:
        raise SystemExit(f"header mismatch:\nactual={reader.fieldnames}\nexpected={header}")

if len(rows) != expected_rows:
    raise SystemExit(f"row-count mismatch: got {len(rows)}, expected {expected_rows}")

def number(row, key):
    try:
        value = float(row[key])
    except (TypeError, ValueError):
        raise SystemExit(f"invalid {key}: {row.get(key)!r}")
    if not math.isfinite(value):
        raise SystemExit(f"non-finite {key}: {value}")
    return value

def positive(row, key):
    value = number(row, key)
    if value <= 0.0:
        raise SystemExit(f"non-positive {key}: {value}")
    return value

def integer(row, key):
    try:
        return int(row[key])
    except (TypeError, ValueError):
        raise SystemExit(f"invalid integer {key}: {row.get(key)!r}")

def close(actual, expected, *, rel=2e-12, abs_=1e-15):
    return math.isclose(actual, expected, rel_tol=rel, abs_tol=abs_)

def truthy(value):
    return str(value).strip().lower() in {"1", "t", "true", "yes"}

counts = Counter()
repetition_sets = defaultdict(set)
sequence_values = []
run_ids = []
status_counts = Counter()

for index, row in enumerate(rows, start=1):
    if row["schema_version"] != "cpu-gpu-v2":
        raise SystemExit(f"row {index}: wrong schema_version {row['schema_version']!r}")
    if row["session_id"] != expected_session_id:
        raise SystemExit(f"row {index}: wrong session_id {row['session_id']!r}")
    if row["workload"] != "AXPY":
        raise SystemExit(f"row {index}: wrong workload {row['workload']!r}")
    if row["implementation"] != "openmp_axpy_inplace_fp32":
        raise SystemExit(f"row {index}: wrong implementation {row['implementation']!r}")
    if row["execution_mode"] != "cpu_native":
        raise SystemExit(f"row {index}: wrong execution_mode {row['execution_mode']!r}")
    if not row["device_name"].strip():
        raise SystemExit(f"row {index}: empty device_name")

    sequence = integer(row, "sequence_index")
    run_id = integer(row, "run_id_global")
    repetition = integer(row, "repetition")
    thread = integer(row, "num_threads")
    n = integer(row, "problem_size")
    batches = integer(row, "batches")

    sequence_values.append(sequence)
    run_ids.append(run_id)

    if n not in sizes:
        raise SystemExit(f"row {index}: unexpected N={n}")
    if thread not in threads:
        raise SystemExit(f"row {index}: unexpected threads={thread}")
    if repetition < 1 or repetition > reps:
        raise SystemExit(f"row {index}: invalid repetition={repetition}")
    if batches <= 0 or batches > 250000:
        raise SystemExit(f"row {index}: invalid batches={batches}")

    expected_spec = (
        f"elements={n};alpha=3.0;x=period29*2^-16;"
        "y0=period31*2^-8;reset=outside_window;max_batches=250000"
    )
    if row["problem_spec"] != expected_spec:
        raise SystemExit(
            f"row {index}: problem_spec mismatch:\n"
            f"actual={row['problem_spec']!r}\nexpected={expected_spec!r}"
        )

    e2e = positive(row, "e2e_time_s")
    kernel = positive(row, "kernel_time_s")
    wall = positive(row, "wall_time_s")
    device = positive(row, "device_energy_j")
    total = positive(row, "total_energy_j")
    dram = number(row, "dram_energy_j")

    if not close(kernel, e2e) or not close(wall, e2e):
        raise SystemExit(
            f"row {index}: CPU time identity failed: "
            f"kernel={kernel}, e2e={e2e}, wall={wall}"
        )

    if dram == -1.0:
        expected_total = device
    elif dram >= 0.0:
        expected_total = device + dram
    else:
        raise SystemExit(f"row {index}: invalid dram_energy_j={dram}")

    if not close(total, expected_total):
        raise SystemExit(
            f"row {index}: total energy identity failed: "
            f"{total} vs {expected_total}"
        )

    flops_total = integer(row, "flops_total")
    logical_bytes = integer(row, "logical_bytes_per_op")
    expected_flops = 2 * n * batches
    expected_bytes = 12 * n
    if flops_total != expected_flops:
        raise SystemExit(
            f"row {index}: flops_total={flops_total}, expected={expected_flops}"
        )
    if logical_bytes != expected_bytes:
        raise SystemExit(
            f"row {index}: logical_bytes_per_op={logical_bytes}, "
            f"expected={expected_bytes}"
        )

    formula_checks = [
        ("energy_per_op_j", number(row, "energy_per_op_j"), device / batches),
        ("energy_per_second_j", number(row, "energy_per_second_j"), device / wall),
        ("energy_per_flop_j", number(row, "energy_per_flop_j"), device / flops_total),
        ("time_per_op_ms_kernel", number(row, "time_per_op_ms_kernel"), 1000.0 * kernel / batches),
        ("time_per_op_ms_e2e", number(row, "time_per_op_ms_e2e"), 1000.0 * e2e / batches),
        ("gflops_per_s", number(row, "gflops_per_s"), flops_total / kernel / 1.0e9),
        ("avg_power_w", number(row, "avg_power_w"), device / wall),
    ]
    for name, actual, expected in formula_checks:
        if not close(actual, expected):
            raise SystemExit(
                f"row {index}: {name} identity failed: {actual} vs {expected}"
            )

    expected_status = "below" if e2e < 0.75 else ("in_range" if e2e <= 1.25 else "above")
    if row["runtime_status"] != expected_status:
        raise SystemExit(
            f"row {index}: runtime_status={row['runtime_status']!r}, "
            f"expected={expected_status!r}"
        )
    if expected_status == "below":
        raise SystemExit(f"row {index}: forbidden below row was written")
    status_counts[expected_status] += 1

    for key in ("pcie_gen", "pcie_width", "sm_clock_mhz", "mem_clock_mhz",
                "cpu_cycles", "cpu_instructions", "cpu_cache_misses"):
        if number(row, key) != -1.0:
            raise SystemExit(f"row {index}: {key} must be -1, got {row[key]!r}")
    if number(row, "cpu_ipc") != -1.0:
        raise SystemExit(f"row {index}: cpu_ipc must be -1")
    if not truthy(row["checksum_ok"]):
        raise SystemExit(f"row {index}: checksum failed")

    counts[(n, thread)] += 1
    repetition_sets[(n, thread)].add(repetition)

if sequence_values != list(range(1, expected_rows + 1)):
    raise SystemExit(
        f"sequence_index must be 1..{expected_rows}, got "
        f"{sequence_values[:5]}...{sequence_values[-5:]}"
    )
if run_ids != sequence_values:
    raise SystemExit("run_id_global must equal sequence_index within each session")

for n in sizes:
    for thread in threads:
        key = (n, thread)
        if counts[key] != reps:
            raise SystemExit(
                f"coverage N={n}, threads={thread}: "
                f"got {counts[key]} rows, expected {reps}"
            )
        if repetition_sets[key] != set(range(1, reps + 1)):
            raise SystemExit(
                f"repetition set N={n}, threads={thread}: "
                f"{sorted(repetition_sets[key])}"
            )

print(
    f"validated {len(rows)} AXPY rows across {len(counts)} configurations; "
    f"runtime_status={dict(status_counts)}"
)
PYVALID
}

validate_quickcheck_log() {
    local log_path=$1
    local expected_rows=$2

    python3 - "$log_path" "$expected_rows" <<'PYLOG'
import math
import re
import sys

path, expected_rows_s = sys.argv[1:]
expected_rows = int(expected_rows_s)
text = open(path, encoding="utf-8", errors="replace").read()

result_lines = [line for line in text.splitlines() if line.startswith("[AXPY] ")]
error_lines = [line for line in text.splitlines() if "max_abs_error=" in line]

if len(result_lines) != expected_rows:
    raise SystemExit(
        f"quickcheck log: got {len(result_lines)} [AXPY] rows, expected {expected_rows}"
    )
if len(error_lines) != expected_rows:
    raise SystemExit(
        f"quickcheck log: got {len(error_lines)} error lines, expected {expected_rows}"
    )

error_pattern = re.compile(
    r"max_abs_error=(?P<abs>\S+)\s+max_rel_error=(?P<rel>\S+)"
)
for index, line in enumerate(error_lines, start=1):
    match = error_pattern.search(line)
    if not match:
        raise SystemExit(f"error line {index}: malformed: {line!r}")
    abs_error = float(match.group("abs"))
    rel_error = float(match.group("rel"))
    if not math.isfinite(abs_error) or not math.isfinite(rel_error):
        raise SystemExit(f"error line {index}: non-finite checksum diagnostic")
    if abs_error != 0.0 or rel_error != 0.0:
        raise SystemExit(
            f"error line {index}: expected exact zero errors, "
            f"got abs={abs_error}, rel={rel_error}"
        )

probe_lines = [line for line in text.splitlines() if line.startswith("[ANTI_COLLAPSE] ")]
if len(probe_lines) != 1:
    raise SystemExit(
        f"expected exactly one anti-collapse line, got {len(probe_lines)}"
    )

fields = {}
for token in probe_lines[0].split()[1:]:
    if "=" in token:
        key, value = token.split("=", 1)
        fields[key] = value

required = {
    "N","threads","B_cal","B_probe","two_B_probe","t1","t2","ratio",
    "time_basis","checksum1","checksum2","gate",
}
missing = sorted(required - fields.keys())
if missing:
    raise SystemExit(f"anti-collapse line missing fields: {missing}")

n = int(fields["N"])
threads = int(fields["threads"])
b_cal = int(fields["B_cal"])
b_probe = int(fields["B_probe"])
two_b_probe = int(fields["two_B_probe"])
t1 = float(fields["t1"])
t2 = float(fields["t2"])
ratio = float(fields["ratio"])

if n != 1_000_000 or threads != 20:
    raise SystemExit(f"anti-collapse wrong configuration: N={n}, threads={threads}")
if not (1 <= b_cal <= 250000):
    raise SystemExit(f"anti-collapse invalid B_cal={b_cal}")
if b_probe < 100 or two_b_probe != 2 * b_probe or two_b_probe > 250000:
    raise SystemExit(
        f"anti-collapse invalid probe batches: B={b_probe}, 2B={two_b_probe}"
    )
if not all(math.isfinite(value) for value in (t1, t2, ratio)):
    raise SystemExit("anti-collapse non-finite timing or ratio")
if t1 < 0.020 or t2 < 0.020:
    raise SystemExit(f"anti-collapse duration too short: t1={t1}, t2={t2}")
if not (1.7 <= ratio <= 2.3):
    raise SystemExit(f"anti-collapse ratio outside [1.7,2.3]: {ratio}")
if fields["time_basis"] != "wall_time_s":
    raise SystemExit(f"anti-collapse wrong time basis: {fields['time_basis']!r}")
if fields["checksum1"] != "OK" or fields["checksum2"] != "OK":
    raise SystemExit("anti-collapse checksum failure")
if fields["gate"] != "PASS":
    raise SystemExit(f"anti-collapse gate={fields['gate']!r}")

print(
    "validated exact checksum diagnostics and anti-collapse gate: "
    f"B={b_probe}, 2B={two_b_probe}, ratio={ratio:.6f}"
)
PYLOG
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

    local -a env_args=(
        env
        -u GOMP_CPU_AFFINITY
        OMP_DYNAMIC=FALSE
        OMP_PROC_BIND=spread
        OMP_PLACES=cores
    )

    if [[ -n "$size_filter" ]]; then
        env_args+=(BENCH_SIZE_FILTER="$size_filter")
    else
        env_args+=(-u BENCH_SIZE_FILTER)
    fi
    if [[ -n "$thread_filter" ]]; then
        env_args+=(BENCH_THREAD_FILTER="$thread_filter")
    else
        env_args+=(-u BENCH_THREAD_FILTER)
    fi
    if (( anti_collapse == 1 )); then
        env_args+=(AXPY_ANTI_COLLAPSE_PROBE=1)
    else
        env_args+=(-u AXPY_ANTI_COLLAPSE_PROBE)
    fi

    "${env_args[@]}" \
        stdbuf -oL -eL \
        "$BIN" "$output" "$reps" "$session_id" "$seed" \
        2>&1 | tee "$log"
}

stamp=$(date +%Y%m%d_%H%M%S)

quick_id="axpy_${PLATFORM_LC}_${stamp}_quickcheck"
quick_csv="$RUN_DIR/${quick_id}.csv"
quick_log="$RUN_DIR/${quick_id}.log"
quick_expected=$((3 * 2 * QUICK_REPS))

echo "[quickcheck] sizes=${QUICK_SIZES}; threads=${QUICK_THREADS}; reps=${QUICK_REPS}"
run_axpy \
    "$quick_csv" "$quick_log" "$QUICK_REPS" "$quick_id" "$QUICK_SEED" \
    "$QUICK_SIZES" "$QUICK_THREADS" 1
validate_csv "$quick_csv" "$QUICK_SIZES" "$QUICK_THREADS" "$QUICK_REPS" "$quick_id"
validate_quickcheck_log "$quick_log" "$quick_expected"
echo "[quickcheck] PASS: $quick_csv"

if (( QUICKCHECK_ONLY == 1 )); then
    POWER_OFF_AT_END=0
    SUCCESS=1
    exit 0
fi

for ((session=1; session<=SESSIONS; session++)); do
    seed=$((SEED_BASE + session))
    session_id="axpy_${PLATFORM_LC}_${stamp}_session${session}"
    output="$RUN_DIR/${session_id}.csv"
    log="$RUN_DIR/${session_id}.log"

    echo "[run] ${PLATFORM} AXPY session ${session}/${SESSIONS}; seed=${seed}; reps=${REPS}"
    run_axpy "$output" "$log" "$REPS" "$session_id" "$seed" "" "" 0
    validate_csv "$output" "$SIZES" "$ALL_THREADS" "$REPS" "$session_id"
    echo "[run] Session ${session}/${SESSIONS} PASS: $output"

    if (( session < SESSIONS )); then
        echo "[pause] Sleeping ${SESSION_PAUSE_S}s before the next independent session..."
        sleep "$SESSION_PAUSE_S"
    fi
done

echo "[done] ${PLATFORM} AXPY: quickcheck plus ${SESSIONS} validated sessions completed."
echo "[done] Official measurements per N×thread configuration: $((SESSIONS * REPS))."
SUCCESS=1

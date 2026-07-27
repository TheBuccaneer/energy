#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR="$SCRIPT_DIR"
SRC="$ROOT_DIR/scripts/AXPY/main_axpy.cu"
SIBLING_SRC="$ROOT_DIR/../5060ti/scripts/AXPY/main_axpy.cu"
BUILD_DIR="$ROOT_DIR/scripts/AXPY/.build"
BIN="$BUILD_DIR/main_axpy"
RUN_DIR="$ROOT_DIR/runs/AXPY"
RESTORE="$ROOT_DIR/03_disable_GPU_3090.sh"
GPU_LABEL="RTX 3090"
EXPECTED_GPU="${EXPECTED_GPU:-RTX 3090}"
GPU_INDEX="${GPU_INDEX:-0}"
REPS="${REPS:-10}"
SESSIONS="${SESSIONS:-5}"
SESSION_PAUSE_S="${SESSION_PAUSE_S:-60}"
POWER_OFF_AT_END="${POWER_OFF_AT_END:-1}"
SEED_BASE=$((0x4158505900000000))

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

    echo "[cleanup] Restoring ${GPU_LABEL} settings..."
    if ! sudo env GPU_INDEX="$GPU_INDEX" bash "$RESTORE"; then
        echo "[cleanup] WARNING: restore script failed." >&2
        status=1
    fi

    if (( SUCCESS == 1 && status == 0 && POWER_OFF_AT_END == 1 )); then
        echo "[done] All ${GPU_LABEL} AXPY sessions completed. Powering off."
        sudo systemctl poweroff
    elif (( SUCCESS == 1 && status == 0 )); then
        echo "[done] All ${GPU_LABEL} AXPY sessions completed. POWER_OFF_AT_END=0."
    else
        echo "[abort] ${GPU_LABEL} AXPY campaign incomplete; no automatic power-off." >&2
    fi
    exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

for value_name in REPS SESSIONS SESSION_PAUSE_S POWER_OFF_AT_END; do
    value=${!value_name}
    [[ "$value" =~ ^[0-9]+$ ]] || die "$value_name must be numeric"
done
[[ "$REPS" -eq 10 ]] || die "official AXPY campaign requires REPS=10"
[[ "$SESSIONS" -eq 5 ]] || die "official AXPY campaign requires SESSIONS=5"
[[ "$SESSION_PAUSE_S" -eq 60 ]] || die "official AXPY campaign requires SESSION_PAUSE_S=60"
[[ "$POWER_OFF_AT_END" =~ ^[01]$ ]] || die "POWER_OFF_AT_END must be 0 or 1"
[[ -f "$RESTORE" ]] || die "missing restore script: $RESTORE"

EXPECTED_SOURCE_SHA256="c4e3099929f736b3fe101a10b870f7e677f0b6c28e13be4577aeb1a376c18f93"

die() {
    echo "ERROR: $*" >&2
    exit 2
}

for command in nvcc python3 stdbuf tee sha256sum ldd nvidia-smi; do
    command -v "$command" >/dev/null || die "required command not found: $command"
done

[[ -f "$SRC" ]] || die "missing source: $SRC"
actual_source_sha=$(sha256sum "$SRC" | awk '{print $1}')
[[ "$actual_source_sha" == "$EXPECTED_SOURCE_SHA256" ]] || \
    die "source hash mismatch: got $actual_source_sha; expected $EXPECTED_SOURCE_SHA256"

if [[ -f "$SIBLING_SRC" ]]; then
    cmp -s "$SRC" "$SIBLING_SRC" || die "3090/5060 Ti AXPY sources are not byte-identical"
    echo "[preflight] GPU source byte-identity: PASS"
else
    echo "[preflight] WARNING: sibling GPU source not found; byte-identity not checked: $SIBLING_SRC" >&2
fi

mkdir -p "$BUILD_DIR" "$RUN_DIR"
COMPILE_CMD=(nvcc -O3 -std=c++17 -lineinfo "$SRC" -lnvidia-ml -o "$BIN")
echo "[build] Compiling ${GPU_LABEL} AXPY..."
printf '[build] command:'
printf ' %q' "${COMPILE_CMD[@]}"
printf '\n'
"${COMPILE_CMD[@]}"
ldd "$BIN" | grep -q 'libnvidia-ml' || die "binary is not linked against libnvidia-ml"

echo "[preflight] source_sha256=$actual_source_sha"
echo "[preflight] runner_sha256=$(sha256sum "${BASH_SOURCE[0]}" | awk '{print $1}')"
echo "[preflight] binary_sha256=$(sha256sum "$BIN" | awk '{print $1}')"

validate_csv() {
    local csv_path=$1
    local expected_reps=$2
    local expected_session_id=$3

    python3 - "$csv_path" "$expected_reps" "$expected_session_id" "$EXPECTED_GPU" <<'PYVALID'
import csv
import math
import sys
from collections import Counter, defaultdict

path, reps_s, expected_session_id, expected_gpu = sys.argv[1:]
reps = int(reps_s)
expected_sizes = {
    1_000_000, 2_000_000, 4_000_000, 8_000_000, 16_000_000,
    32_000_000, 64_000_000, 128_000_000, 256_000_000,
}
if "quickcheck" in expected_session_id:
    expected_sizes = {1_000_000, 64_000_000, 256_000_000}
expected_rows = len(expected_sizes) * reps

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
        raise SystemExit(
            f"header mismatch: got {len(reader.fieldnames or [])} columns; expected 45"
        )

if len(rows) != expected_rows:
    raise SystemExit(f"row-count mismatch: got {len(rows)}, expected {expected_rows}")

def num(row, key):
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError):
        raise SystemExit(f"invalid {key}: {row.get(key)!r}")
    if not math.isfinite(value):
        raise SystemExit(f"non-finite {key}: {value}")
    return value

def integer(row, key):
    try:
        return int(row[key])
    except (KeyError, TypeError, ValueError):
        raise SystemExit(f"invalid integer {key}: {row.get(key)!r}")

def positive(row, key):
    value = num(row, key)
    if value <= 0.0:
        raise SystemExit(f"non-positive {key}: {value}")
    return value

def close(actual, expected, rel=3e-12, abs_=1e-15):
    return math.isclose(actual, expected, rel_tol=rel, abs_tol=abs_)

def truthy(value):
    return str(value).strip().lower() in {"1", "t", "true", "yes", "ok"}

counts = Counter()
repetition_sets = defaultdict(set)
sequence_values = []
run_ids = []
status_counts = Counter()
timing_warnings = 0
throttle_nonzero = 0

for index, row in enumerate(rows, start=1):
    if len(row) != 45:
        raise SystemExit(f"row {index}: expected 45 columns, got {len(row)}")
    if row["schema_version"] != "cpu-gpu-v2":
        raise SystemExit(f"row {index}: wrong schema_version {row['schema_version']!r}")
    if row["session_id"] != expected_session_id:
        raise SystemExit(f"row {index}: wrong session_id {row['session_id']!r}")
    if row["workload"] != "AXPY":
        raise SystemExit(f"row {index}: wrong workload {row['workload']!r}")
    if row["implementation"] != "cuda_axpy_inplace_fp32":
        raise SystemExit(f"row {index}: wrong implementation {row['implementation']!r}")
    if row["execution_mode"] != "gpu_resident":
        raise SystemExit(f"row {index}: wrong execution_mode {row['execution_mode']!r}")
    if expected_gpu.lower() not in row["device_name"].lower():
        raise SystemExit(
            f"row {index}: GPU mismatch: expected {expected_gpu!r}, got {row['device_name']!r}"
        )
    if integer(row, "num_threads") != -1:
        raise SystemExit(f"row {index}: num_threads must be -1")

    sequence = integer(row, "sequence_index")
    run_id = integer(row, "run_id_global")
    repetition = integer(row, "repetition")
    n = integer(row, "problem_size")
    batches = integer(row, "batches")
    sequence_values.append(sequence)
    run_ids.append(run_id)

    if n not in expected_sizes:
        raise SystemExit(f"row {index}: unexpected N={n}")
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
    energy = positive(row, "device_energy_j")
    total = positive(row, "total_energy_j")
    dram = num(row, "dram_energy_j")

    if not close(e2e, wall):
        raise SystemExit(f"row {index}: e2e_time_s != wall_time_s")
    if not close(total, energy):
        raise SystemExit(f"row {index}: total_energy_j != device_energy_j")
    if dram != -1.0:
        raise SystemExit(f"row {index}: dram_energy_j must be -1")

    if kernel > e2e:
        excess = kernel - e2e
        allowed = max(0.0005, 0.005 * e2e)
        if excess > allowed:
            raise SystemExit(
                f"row {index}: material GPU timing contradiction: "
                f"kernel={kernel}, e2e={e2e}, excess={excess}, allowed={allowed}"
            )
        timing_warnings += 1

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
            f"row {index}: logical_bytes_per_op={logical_bytes}, expected={expected_bytes}"
        )

    checks = [
        ("energy_per_op_j", num(row, "energy_per_op_j"), energy / batches),
        ("energy_per_second_j", num(row, "energy_per_second_j"), energy / wall),
        ("energy_per_flop_j", num(row, "energy_per_flop_j"), energy / flops_total),
        ("time_per_op_ms_kernel", num(row, "time_per_op_ms_kernel"), 1000.0 * kernel / batches),
        ("time_per_op_ms_e2e", num(row, "time_per_op_ms_e2e"), 1000.0 * e2e / batches),
        ("gflops_per_s", num(row, "gflops_per_s"), flops_total / kernel / 1.0e9),
        ("avg_power_w", num(row, "avg_power_w"), energy / wall),
    ]
    for name, actual, expected in checks:
        if not close(actual, expected):
            raise SystemExit(
                f"row {index}: {name} identity failed: {actual} vs {expected}"
            )

    expected_status = "below" if e2e < 0.75 else ("in_range" if e2e <= 1.25 else "above")
    if row["runtime_status"] != expected_status:
        raise SystemExit(
            f"row {index}: runtime_status={row['runtime_status']!r}, expected={expected_status!r}"
        )
    if expected_status == "below":
        raise SystemExit(f"row {index}: forbidden below row was written")
    status_counts[expected_status] += 1

    for key in ("pcie_gen", "pcie_width", "sm_clock_mhz", "clock_before_mhz",
                "clock_after_mhz", "mem_clock_mhz", "temp_c", "temp_before_c",
                "temp_after_c"):
        if integer(row, key) <= 0:
            raise SystemExit(f"row {index}: {key} must be positive")
    for key in ("cpu_cycles", "cpu_instructions", "cpu_cache_misses"):
        if integer(row, key) != -1:
            raise SystemExit(f"row {index}: {key} must be -1")
    if num(row, "cpu_ipc") != -1.0:
        raise SystemExit(f"row {index}: cpu_ipc must be numeric -1")

    throttle = row["throttle_reasons"].strip()
    if not throttle.lower().startswith("0x"):
        raise SystemExit(f"row {index}: malformed throttle_reasons={throttle!r}")
    try:
        throttle_value = int(throttle, 16)
    except ValueError:
        raise SystemExit(f"row {index}: malformed throttle_reasons={throttle!r}")
    if throttle_value != 0:
        throttle_nonzero += 1

    if not truthy(row["checksum_ok"]):
        raise SystemExit(f"row {index}: checksum failed")

    counts[n] += 1
    repetition_sets[n].add(repetition)

if sequence_values != list(range(1, expected_rows + 1)):
    raise SystemExit(f"sequence_index must be 1..{expected_rows}")
if run_ids != sequence_values:
    raise SystemExit("run_id_global must equal sequence_index within each session")
for n in expected_sizes:
    if counts[n] != reps:
        raise SystemExit(f"coverage N={n}: got {counts[n]} rows, expected {reps}")
    if repetition_sets[n] != set(range(1, reps + 1)):
        raise SystemExit(f"repetition set N={n}: {sorted(repetition_sets[n])}")

print(
    f"validated {len(rows)} AXPY rows across {len(counts)} sizes; "
    f"runtime_status={dict(status_counts)}; "
    f"minor_timing_warnings={timing_warnings}; throttle_nonzero={throttle_nonzero}"
)
PYVALID
}


sudo -v
(
    while true; do
        sudo -n true 2>/dev/null || exit
        sleep 60
        kill -0 "$$" 2>/dev/null || exit
    done
) &
KEEPALIVE_PID=$!

stamp=$(date +%Y%m%d_%H%M%S)
manifest="$RUN_DIR/axpy_3090_${stamp}_manifest.txt"
[[ ! -e "$manifest" ]] || die "manifest already exists: $manifest"

repo_root=$(git -C "$ROOT_DIR" rev-parse --show-toplevel 2>/dev/null || true)
git_commit=$(git -C "$ROOT_DIR" rev-parse HEAD 2>/dev/null || echo unavailable)
git_status=$(git -C "$ROOT_DIR" status --porcelain 2>/dev/null || true)
nvcc_line=$(nvcc --version | tail -n 1)
driver_line=$(nvidia-smi --query-gpu=name,driver_version,pci.bus_id --format=csv,noheader -i "$GPU_INDEX" 2>/dev/null || echo unavailable)
compile_text=$(printf '%q ' "${COMPILE_CMD[@]}")

{
    echo "workload=AXPY"
    echo "platform=3090"
    echo "gpu_label=${GPU_LABEL}"
    echo "expected_gpu=${EXPECTED_GPU}"
    echo "gpu_index=${GPU_INDEX}"
    echo "campaign_stamp=${stamp}"
    echo "sessions=${SESSIONS}"
    echo "repetitions=${REPS}"
    echo "session_pause_s=${SESSION_PAUSE_S}"
    echo "source_path=${SRC}"
    echo "source_sha256=${actual_source_sha}"
    echo "runner_path=${BASH_SOURCE[0]}"
    echo "runner_sha256=$(sha256sum "${BASH_SOURCE[0]}" | awk '{print $1}')"
    echo "binary_path=${BIN}"
    echo "binary_sha256=$(sha256sum "$BIN" | awk '{print $1}')"
    echo "compile_command=${compile_text}"
    echo "repo_root=${repo_root:-unavailable}"
    echo "git_commit=${git_commit}"
    echo "git_dirty=$([[ -n "$git_status" ]] && echo yes || echo no)"
    echo "hostname=$(hostname)"
    echo "kernel=$(uname -srmo)"
    echo "nvcc=${nvcc_line}"
    echo "gpu_driver=${driver_line}"
    echo "seed_rule_requested=0x4158505900000000+session"
    echo "note_source_rng=source stores the supplied seed as uint32_t; effective low 32 bits are recorded per session"
} > "$manifest"

echo "[run] No quickcheck is executed by this 02 runner."
echo "[run] Manifest: $manifest"

for ((session=1; session<=SESSIONS; session++)); do
    seed=$((SEED_BASE + session))
    effective_seed=$((seed & 0xFFFFFFFF))
    session_id="axpy_3090_${stamp}_session${session}"
    output="$RUN_DIR/${session_id}.csv"
    log="$RUN_DIR/${session_id}.log"
    [[ ! -e "$output" && ! -e "$log" ]] || die "session output already exists: $session_id"

    echo "[run] ${GPU_LABEL} AXPY session ${session}/${SESSIONS}; reps=${REPS}; seed=${seed}"
    env \
        -u BENCH_SIZE_FILTER \
        -u AXPY_ANTI_COLLAPSE_PROBE \
        CUDA_VISIBLE_DEVICES="$GPU_INDEX" \
        NVIDIA_TF32_OVERRIDE=0 \
        BENCH_EXPECTED_GPU="$EXPECTED_GPU" \
        stdbuf -oL -eL \
        "$BIN" "$output" "$REPS" "$session_id" "$seed" \
        2>&1 | tee "$log"

    validate_csv "$output" "$REPS" "$session_id"

    order=$(python3 - "$output" <<'PYORDER'
import csv, sys
seen=[]
with open(sys.argv[1], newline='', encoding='utf-8') as f:
    for row in csv.DictReader(f):
        n=int(row['problem_size'])
        if n not in seen:
            seen.append(n)
print(','.join(map(str, seen)))
PYORDER
)
    {
        echo "session_${session}_id=${session_id}"
        echo "session_${session}_seed_requested=${seed}"
        echo "session_${session}_seed_effective_uint32=${effective_seed}"
        echo "session_${session}_configuration_order=${order}"
        echo "session_${session}_csv_sha256=$(sha256sum "$output" | awk '{print $1}')"
        echo "session_${session}_log_sha256=$(sha256sum "$log" | awk '{print $1}')"
    } >> "$manifest"

    echo "[run] Session ${session}/${SESSIONS} PASS: $output"
    if (( session < SESSIONS )); then
        echo "[pause] Sleeping 60 s before the next independent session..."
        sleep 60
    fi
done

echo "[done] ${GPU_LABEL} AXPY: ${SESSIONS} validated sessions completed."
echo "[done] Official measurements per problem size: $((SESSIONS * REPS))."
SUCCESS=1

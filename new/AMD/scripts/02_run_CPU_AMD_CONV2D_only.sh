#!/usr/bin/env bash
set -euo pipefail

if (( EUID == 0 )); then
    echo "ERROR: do not run this runner with sudo." >&2
    echo "Run 01_enable with sudo, then run this 02 script as your normal user." >&2
    exit 2
fi

PLATFORM="AMD"
PLATFORM_UC="AMD"
PLATFORM_LC="amd"
ENERGY_MODE="amd"

ALL_SHAPES="1,2,3,4,5,6"
QUICK_SHAPES="1,2,3,4,5,6"
ALL_THREADS="1,2,4,8,10,16,20,32,64"
QUICK_THREADS="1,64"
MAX_THREAD="64"

REPS=${REPS:-10}
SESSIONS=${SESSIONS:-5}
QUICK_REPS=${QUICK_REPS:-2}
SESSION_PAUSE_S=${SESSION_PAUSE_S:-60}
POWER_OFF_AT_END=${POWER_OFF_AT_END:-1}
QUICKCHECK_ONLY=${QUICKCHECK_ONLY:-0}

# Frozen CONV2D seed rule: 0x434F4E56 == ASCII "CONV".
SEED_BASE=$((0x434F4E5600001000))
QUICK_SEED=$((0x434F4E5600001900))
ANTI_SEED=$((0x434F4E56000019A0))
VERBOSE_PROBE_SEED=$((0x434F4E56000019B0))
VERBOSE_PROBE_SHAPE="6"
VERBOSE_PROBE_THREAD="1"

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)
PROJECT_DIR=$(cd -- "$ROOT_DIR/../.." && pwd)
SRC="$SCRIPT_DIR/CONV2D/main_conv2d_${PLATFORM_LC}.cpp"
COMMON="$SCRIPT_DIR/common"
HEADER="$COMMON/benchmark_common.hpp"
BUILD_DIR="$SCRIPT_DIR/CONV2D/.build"
BIN="$BUILD_DIR/main_conv2d_${PLATFORM_LC}"
RUN_DIR="$ROOT_DIR/runs/CONV2D"
RESTORE="$SCRIPT_DIR/03_disable_CPU_${PLATFORM}.sh"

STATE_DIR=/tmp/energy_amd_measurement_state

SUCCESS=0
STATE_ACTIVE=0
KEEPALIVE_PID=""
KEEPALIVE_FD=""
CLEANUP_RUNNING=0

log_error() {
    echo "ERROR: $*" >&2
}

die() {
    log_error "$*"
    exit 2
}

ensure_user_owned_directory() {
    local directory=$1
    local uid gid
    uid=$(id -u)
    gid=$(id -g)

    # A previous sudo-run quickcheck may have created these paths as root.
    # Repair only this benchmark's build/run directories before writing.
    sudo -n mkdir -p -- "$directory"
    sudo -n chown -R -- "$uid:$gid" "$directory"

    [[ -d "$directory" ]] || die "failed to create directory: $directory"
    [[ -w "$directory" ]] || die "directory is not writable after ownership repair: $directory"
}

stop_sudo_keepalive() {
    if [[ -n "$KEEPALIVE_FD" ]]; then
        printf 'stop\n' >&"$KEEPALIVE_FD" 2>/dev/null || true
    fi
    if [[ -n "$KEEPALIVE_PID" ]]; then
        wait "$KEEPALIVE_PID" 2>/dev/null || true
        KEEPALIVE_PID=""
    fi
    if [[ -n "$KEEPALIVE_FD" ]]; then
        exec {KEEPALIVE_FD}>&- 2>/dev/null || true
        KEEPALIVE_FD=""
    fi
}

start_sudo_keepalive() {
    local fifo_dir
    fifo_dir=$(mktemp -d)
    mkfifo "$fifo_dir/stop"
    exec {KEEPALIVE_FD}<>"$fifo_dir/stop"
    rm -rf -- "$fifo_dir"

    (
        while true; do
            sudo -n true 2>/dev/null || exit
            # Bash builtin timed read: no child `sleep` process remains to
            # delay cleanup.  stop_sudo_keepalive wakes this immediately.
            if IFS= read -r -t 45 -u "$KEEPALIVE_FD" _; then
                exit 0
            fi
        done
    ) &
    KEEPALIVE_PID=$!
}

measurement_state_is_active() {
    [[ -d "$STATE_DIR" && -r "$STATE_DIR/governors" && -r "$STATE_DIR/perf_event_paranoid" ]]
}

cleanup() {
    local status=$?
    local restored=1
    local poweroff_requested=0

    if (( CLEANUP_RUNNING == 1 )); then
        exit "$status"
    fi
    CLEANUP_RUNNING=1
    trap - EXIT INT TERM
    set +e

    # Keep the sudo timestamp alive through restore and poweroff.  This is
    # deliberately different from older AMD runners that killed the
    # keepalive before their final privileged operations.
    if (( STATE_ACTIVE == 1 )); then
        echo "[cleanup] Restoring ${PLATFORM} CPU settings and measurement permissions..."
        if sudo -n bash "$RESTORE"; then
            restored=1
        else
            echo "[cleanup] ERROR: restore script failed; automatic power-off is suppressed." >&2
            restored=0
            status=1
        fi
    fi

    if (( SUCCESS == 1 && status == 0 && restored == 1 )); then
        if (( QUICKCHECK_ONLY == 1 )); then
            echo "[done] CONV2D quickcheck completed. No official session was started; staying online."
        elif (( POWER_OFF_AT_END == 1 )); then
            echo "[done] CONV2D quickcheck and ${SESSIONS} validated ${PLATFORM} sessions completed."
            echo "[done] Settings restored. Synchronizing filesystems and powering off."
            sync
            poweroff_requested=1
            if ! sudo -n systemctl poweroff; then
                echo "[cleanup] ERROR: systemctl poweroff failed." >&2
                status=1
                poweroff_requested=0
            fi
        else
            echo "[done] CONV2D campaign completed. POWER_OFF_AT_END=0; staying online."
        fi
    else
        echo "[abort] CONV2D run incomplete or failed; no automatic power-off." >&2
    fi

    stop_sudo_keepalive

    if (( poweroff_requested == 1 )); then
        # systemctl normally returns after the shutdown transaction has been
        # queued.  Exiting here is safe; the machine will continue powering off.
        exit 0
    fi
    exit "$status"
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if measurement_state_is_active; then
    STATE_ACTIVE=1
else
    die "${PLATFORM} measurement state is not active. Run: sudo bash \"$SCRIPT_DIR/01_enable_CPU_${PLATFORM}.sh\""
fi

for required in "$SRC" "$HEADER" "$RESTORE"; do
    [[ -f "$required" ]] || die "missing required file: $required"
done

for command in g++ python3 stdbuf tee ldd sha256sum awk grep sync sudo systemctl mktemp mkfifo id chown mkdir; do
    command -v "$command" >/dev/null || die "required command not found: $command"
done

[[ "$SESSIONS" =~ ^[0-9]+$ ]] || die "SESSIONS must be an integer"
[[ "$REPS" =~ ^[0-9]+$ ]] || die "REPS must be an integer"
[[ "$QUICK_REPS" =~ ^[0-9]+$ ]] || die "QUICK_REPS must be an integer"
[[ "$SESSION_PAUSE_S" =~ ^[0-9]+$ ]] || die "SESSION_PAUSE_S must be an integer"
[[ "$SESSIONS" -eq 5 ]] || die "official CONV2D campaign requires SESSIONS=5; got $SESSIONS"
[[ "$REPS" -eq 10 ]] || die "official CONV2D campaign requires REPS=10; got $REPS"
[[ "$QUICK_REPS" -eq 2 ]] || die "CONV2D quickcheck requires QUICK_REPS=2; got $QUICK_REPS"
[[ "$QUICKCHECK_ONLY" =~ ^[01]$ ]] || die "QUICKCHECK_ONLY must be 0 or 1"
[[ "$POWER_OFF_AT_END" =~ ^[01]$ ]] || die "POWER_OFF_AT_END must be 0 or 1"

sudo -v
start_sudo_keepalive

ensure_user_owned_directory "$BUILD_DIR"
ensure_user_owned_directory "$RUN_DIR"

CXX=${CXX:-g++}
command -v "$CXX" >/dev/null || die "C++ compiler not found: $CXX"

DNNL_CFLAGS=()
DNNL_LIBS=(-ldnnl)
if command -v pkg-config >/dev/null 2>&1 && pkg-config --exists dnnl; then
    read -r -a DNNL_CFLAGS <<< "$(pkg-config --cflags dnnl)"
    read -r -a DNNL_LIBS <<< "$(pkg-config --libs dnnl)"
fi

COMPILE_CMD=(
    "$CXX"
    -O3
    -march=native
    -std=c++17
    -fopenmp
    -I"$COMMON"
    "${DNNL_CFLAGS[@]}"
    "$SRC"
    "${DNNL_LIBS[@]}"
    -lpthread
    -lm
    -o "$BIN"
)

echo "[build] ${PLATFORM} CONV2D"
"${COMPILE_CMD[@]}"

LDD_OUTPUT=$(ldd "$BIN")
echo "$LDD_OUTPUT" | grep -q 'libdnnl' || die "oneDNN library is not dynamically linked"
echo "$LDD_OUTPUT" | grep -q 'libgomp\.so' || die "GNU OpenMP runtime libgomp is not linked"
if echo "$LDD_OUTPUT" | grep -qE 'libiomp5\.so|(^|[[:space:]/])libomp\.so'; then
    die "multiple/conflicting OpenMP runtimes detected (libgomp plus libiomp/libomp)"
fi
if echo "$LDD_OUTPUT" | grep -qi 'libtbb'; then
    die "TBB is linked; official CONV2D requires oneDNN OpenMP runtime"
fi

echo "[preflight] build/link/runtime PASS"

validate_csv() {
    local csv_path=$1
    local expected_shapes=$2
    local expected_threads=$3
    local expected_reps=$4
    local expected_session_id=$5

    python3 - "$csv_path" "$expected_shapes" "$expected_threads" "$expected_reps" "$expected_session_id" "$ENERGY_MODE" <<'PYVALID'
import csv
import math
import sys
from collections import Counter, defaultdict

path, shapes_s, threads_s, reps_s, expected_session_id, energy_mode = sys.argv[1:]
shapes = {int(value) for value in shapes_s.split(",") if value}
threads = {int(value) for value in threads_s.split(",") if value}
reps = int(reps_s)
expected_rows = len(shapes) * len(threads) * reps

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

shape_data = {
    1: (32, 64, 56, 56, 64, 3, 3, 1, 1),
    2: (32, 64, 56, 56, 128, 3, 3, 2, 1),
    3: (32, 128, 28, 28, 256, 3, 3, 2, 1),
    4: (32, 256, 14, 14, 512, 3, 3, 2, 1),
    5: (32, 3, 224, 224, 64, 7, 7, 2, 3),
    6: (32, 256, 56, 56, 256, 1, 1, 1, 0),
}

def geometry(shape_id):
    n, c, h, w, k, r, s, stride, pad = shape_data[shape_id]
    hout = (h + 2 * pad - r) // stride + 1
    wout = (w + 2 * pad - s) // stride + 1
    return n, c, h, w, k, r, s, stride, pad, hout, wout

def expected_spec(shape_id):
    n, c, h, w, k, r, s, stride, pad, hout, wout = geometry(shape_id)
    return (
        f"N={n};C={c};H={h};W={w};K={k};R={r};S={s};"
        f"stride={stride};pad={pad};Hout={hout};Wout={wout};"
        "dtype=f32;input_layout=NCHW;weight_layout=OIHW;output_layout=NCHW;"
        "bias=none;activation=none;groups=1;dilation=1;"
        "algorithm_policy=convolution_auto;output=overwrite;"
        "reuse_regime=warm_resident;scratchpad=user;onednn_cpu_runtime=OpenMP"
    )

def expected_flops_per_op(shape_id):
    n, c, _h, _w, k, r, s, _stride, _pad, hout, wout = geometry(shape_id)
    return 2 * n * k * c * r * s * hout * wout

def expected_bytes_per_op(shape_id):
    n, c, h, w, k, r, s, _stride, _pad, hout, wout = geometry(shape_id)
    return 4 * (n * c * h * w + k * c * r * s + n * k * hout * wout)

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

def close(actual, expected, *, rel=2e-11, abs_=1e-14):
    return math.isclose(actual, expected, rel_tol=rel, abs_tol=abs_)

def truthy(value):
    return str(value).strip().lower() in {"1", "t", "true", "yes"}

with open(path, newline="", encoding="utf-8") as handle:
    reader = csv.DictReader(handle)
    rows = list(reader)
    if reader.fieldnames != header:
        raise SystemExit(f"header mismatch:\nactual={reader.fieldnames}\nexpected={header}")

if len(rows) != expected_rows:
    raise SystemExit(f"row-count mismatch: got {len(rows)}, expected {expected_rows}")

counts = Counter()
repetition_sets = defaultdict(set)
sequence_values = []
run_ids = []
status_counts = Counter()

for index, row in enumerate(rows, start=1):
    if row["schema_version"] != "cpu-gpu-v2":
        raise SystemExit(f"row {index}: wrong schema_version={row['schema_version']!r}")
    if row["session_id"] != expected_session_id:
        raise SystemExit(f"row {index}: wrong session_id={row['session_id']!r}")
    if row["workload"] != "CONV2D":
        raise SystemExit(f"row {index}: wrong workload={row['workload']!r}")
    implementation = row["implementation"]
    if not implementation.startswith("onednn_convolution_auto:"):
        raise SystemExit(f"row {index}: bad implementation prefix={implementation!r}")
    if ";scratchpad=user;execute_api=c" not in implementation:
        raise SystemExit(f"row {index}: implementation lacks scratchpad/C-API markers")
    if row["execution_mode"] != "cpu_native":
        raise SystemExit(f"row {index}: wrong execution_mode={row['execution_mode']!r}")
    if not row["device_name"].strip():
        raise SystemExit(f"row {index}: empty device_name")

    sequence = integer(row, "sequence_index")
    run_id = integer(row, "run_id_global")
    repetition = integer(row, "repetition")
    thread = integer(row, "num_threads")
    shape_id = integer(row, "problem_size")
    batches = integer(row, "batches")

    if shape_id not in shapes:
        raise SystemExit(f"row {index}: unexpected shape_id={shape_id}")
    if thread not in threads:
        raise SystemExit(f"row {index}: unexpected threads={thread}")
    if repetition < 1 or repetition > reps:
        raise SystemExit(f"row {index}: invalid repetition={repetition}")
    if batches <= 0 or batches > 100000:
        raise SystemExit(f"row {index}: invalid batches={batches}")
    if row["problem_spec"] != expected_spec(shape_id):
        raise SystemExit(
            f"row {index}: problem_spec mismatch:\n"
            f"actual={row['problem_spec']!r}\nexpected={expected_spec(shape_id)!r}"
        )

    e2e = positive(row, "e2e_time_s")
    kernel = positive(row, "kernel_time_s")
    wall = positive(row, "wall_time_s")
    device = positive(row, "device_energy_j")
    total = positive(row, "total_energy_j")
    dram = number(row, "dram_energy_j")

    if not close(kernel, e2e) or not close(wall, e2e):
        raise SystemExit(
            f"row {index}: CPU time identity failed: kernel={kernel}, e2e={e2e}, wall={wall}"
        )

    if energy_mode == "intel":
        if dram == -1.0:
            expected_total = device
        elif dram >= 0.0:
            expected_total = device + dram
        else:
            raise SystemExit(f"row {index}: invalid Intel dram_energy_j={dram}")
    elif energy_mode == "amd":
        if dram != -1.0:
            raise SystemExit(f"row {index}: AMD dram_energy_j must be -1, got {dram}")
        expected_total = device
    else:
        raise SystemExit(f"unknown energy mode: {energy_mode}")

    if not close(total, expected_total):
        raise SystemExit(f"row {index}: total energy identity failed: {total} vs {expected_total}")

    flops_total = integer(row, "flops_total")
    logical_bytes = integer(row, "logical_bytes_per_op")
    expected_flops = expected_flops_per_op(shape_id) * batches
    expected_bytes = expected_bytes_per_op(shape_id)
    if flops_total != expected_flops:
        raise SystemExit(f"row {index}: flops_total={flops_total}, expected={expected_flops}")
    if logical_bytes != expected_bytes:
        raise SystemExit(
            f"row {index}: logical_bytes_per_op={logical_bytes}, expected={expected_bytes}"
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
            raise SystemExit(f"row {index}: {name} identity failed: {actual} vs {expected}")

    expected_status = "below" if e2e < 0.75 else ("in_range" if e2e <= 1.25 else "above")
    if row["runtime_status"] != expected_status:
        raise SystemExit(
            f"row {index}: runtime_status={row['runtime_status']!r}, expected={expected_status!r}"
        )
    if expected_status == "below":
        raise SystemExit(f"row {index}: forbidden below row was written")
    status_counts[expected_status] += 1

    for key in (
        "pcie_gen", "pcie_width", "sm_clock_mhz", "mem_clock_mhz",
        "cpu_cycles", "cpu_instructions", "cpu_cache_misses",
    ):
        if number(row, key) != -1.0:
            raise SystemExit(f"row {index}: {key} must be -1, got {row[key]!r}")
    if number(row, "cpu_ipc") != -1.0:
        raise SystemExit(f"row {index}: cpu_ipc must be -1")
    if row["throttle_reasons"] != "":
        raise SystemExit(f"row {index}: CPU throttle_reasons must be empty")

    temp = number(row, "temp_c")
    temp_before = number(row, "temp_before_c")
    temp_after = number(row, "temp_after_c")
    if temp != max(temp_before, temp_after):
        raise SystemExit(
            f"row {index}: temp_c must equal max(before,after): {temp} vs {temp_before}/{temp_after}"
        )
    if not truthy(row["checksum_ok"]):
        raise SystemExit(f"row {index}: checksum failed")

    sequence_values.append(sequence)
    run_ids.append(run_id)
    counts[(shape_id, thread)] += 1
    repetition_sets[(shape_id, thread)].add(repetition)

if sequence_values != list(range(1, expected_rows + 1)):
    raise SystemExit(
        f"sequence_index must be 1..{expected_rows}, got "
        f"{sequence_values[:5]}...{sequence_values[-5:]}"
    )
if run_ids != sequence_values:
    raise SystemExit("run_id_global must equal sequence_index within each session")

for shape_id in shapes:
    for thread in threads:
        key = (shape_id, thread)
        if counts[key] != reps:
            raise SystemExit(
                f"coverage shape={shape_id}, threads={thread}: got {counts[key]}, expected {reps}"
            )
        if repetition_sets[key] != set(range(1, reps + 1)):
            raise SystemExit(
                f"repetition set shape={shape_id}, threads={thread}: "
                f"{sorted(repetition_sets[key])}"
            )

print(
    f"validated {len(rows)} CONV2D rows across {len(counts)} configurations; "
    f"runtime_status={dict(status_counts)}"
)
PYVALID
}

validate_measurement_log() {
    local log_path=$1
    local expected_shapes=$2
    local expected_threads=$3
    local expected_reps=$4
    local require_verbose=$5

    python3 - "$log_path" "$expected_shapes" "$expected_threads" "$expected_reps" "$require_verbose" <<'PYLOG'
import math
import re
import sys
from collections import Counter

path, shapes_s, threads_s, reps_s, require_verbose_s = sys.argv[1:]
shapes = {int(value) for value in shapes_s.split(",") if value}
threads = {int(value) for value in threads_s.split(",") if value}
reps = int(reps_s)
require_verbose = bool(int(require_verbose_s))
expected_configs = len(shapes) * len(threads)
expected_rows = expected_configs * reps
lines = open(path, encoding="utf-8", errors="replace").read().splitlines()

banner_lines = [line for line in lines if line.startswith("CONV2D | ")]
calibration_lines = [line for line in lines if line.startswith("[CALIBRATION] ")]
result_lines = [line for line in lines if line.startswith("[CONV2D] ")]
checksum_lines = [line for line in lines if line.startswith("[CHECKSUM] ")]
anti_lines = [line for line in lines if line.startswith("[ANTI_COLLAPSE] ")]
diagnostic_lines = [
    line for line in lines
    if line.startswith(("[CONFIG] ", "[ONEDNN] ", "[ENV] ", "[BENCHMARK] "))
]
verbose_lines = [
    line for line in lines
    if re.match(r"^(onednn_verbose|dnnl_verbose),", line.strip(), re.IGNORECASE)
]

if len(banner_lines) != 1:
    raise SystemExit(f"expected one compact CONV2D banner, got {len(banner_lines)}")
if len(calibration_lines) != expected_configs:
    raise SystemExit(
        f"CALIBRATION count mismatch: got {len(calibration_lines)}, expected {expected_configs}"
    )
if len(result_lines) != expected_rows:
    raise SystemExit(f"CONV2D count mismatch: got {len(result_lines)}, expected {expected_rows}")
if anti_lines:
    raise SystemExit("regular measurement log unexpectedly contains anti-collapse output")
if diagnostic_lines:
    raise SystemExit("normal measurement log unexpectedly contains audit/provenance diagnostics")
if checksum_lines:
    raise SystemExit("successful normal measurement unexpectedly emitted detailed checksum diagnostics")
if require_verbose:
    if not verbose_lines:
        raise SystemExit("required oneDNN verbose output is missing")
else:
    if verbose_lines:
        raise SystemExit("normal measurement unexpectedly contains oneDNN verbose output")

calibration_coverage = set()
for line in calibration_lines:
    fields = dict(token.split("=", 1) for token in line.split()[1:] if "=" in token)
    shape = int(fields["shape"])
    thread = int(fields["threads"])
    batches = int(fields["batches"])
    if shape not in shapes or thread not in threads or batches < 1:
        raise SystemExit(f"invalid calibration line: {line}")
    calibration_coverage.add((shape, thread))
expected_coverage = {(shape, thread) for shape in shapes for thread in threads}
if calibration_coverage != expected_coverage:
    raise SystemExit(f"calibration coverage mismatch: {sorted(calibration_coverage)}")

counts = Counter()
repetition_sets = {}
for line in result_lines:
    fields = dict(token.split("=", 1) for token in line.split()[1:] if "=" in token)
    shape = int(fields["shape"])
    thread = int(fields["threads"])
    rep = int(fields["rep"])
    batches = int(fields["batches"])
    runtime = float(fields["e2e_time_s"])
    energy = float(fields["device_energy_j"])
    status = fields["runtime_status"]
    checksum = fields["checksum"]
    key = (shape, thread)
    if shape not in shapes or thread not in threads:
        raise SystemExit(f"unexpected result configuration: {line}")
    if batches < 1 or not math.isfinite(runtime) or runtime <= 0.0:
        raise SystemExit(f"invalid result timing/batches: {line}")
    if not math.isfinite(energy) or energy <= 0.0:
        raise SystemExit(f"invalid result energy: {line}")
    if status == "below" or checksum != "OK":
        raise SystemExit(f"failed result gate: {line}")
    counts[key] += 1
    repetition_sets.setdefault(key, set()).add(rep)

for key in expected_coverage:
    if counts[key] != reps or repetition_sets.get(key) != set(range(1, reps + 1)):
        raise SystemExit(
            f"result coverage mismatch for {key}: count={counts[key]}, reps={sorted(repetition_sets.get(key, set()))}"
        )

print(f"validated compact CONV2D log: configurations={expected_configs}, rows={expected_rows}")
PYLOG
}

validate_anti_collapse_log() {
    local log_path=$1
    local expected_thread=$2

    python3 - "$log_path" "$expected_thread" <<'PYANTI'
import math
import re
import sys

path, expected_thread_s = sys.argv[1:]
expected_thread = int(expected_thread_s)
text = open(path, encoding="utf-8", errors="replace").read()
lines = text.splitlines()

anti_lines = [line for line in lines if line.startswith("[ANTI_COLLAPSE] ")]
if len(anti_lines) != 1:
    raise SystemExit(f"expected exactly one anti-collapse line, got {len(anti_lines)}")
if any(line.startswith("[CONV2D] ") for line in lines):
    raise SystemExit("anti-collapse mode emitted regular CONV2D result lines")
if any(line.startswith("[CALIBRATION] ") for line in lines):
    raise SystemExit("anti-collapse mode emitted regular calibration lines")
if any(line.startswith(("[CONFIG] ", "[ONEDNN] ", "[ENV] ", "[BENCHMARK] ", "CONV2D | ")) for line in lines):
    raise SystemExit("anti-collapse quiet mode emitted audit/provenance diagnostics")

fields = dict(token.split("=", 1) for token in anti_lines[0].split()[1:] if "=" in token)
required = {
    "shape", "threads", "B", "two_B", "t1", "t2", "ratio",
    "checksum_B", "checksum_2B", "gate",
}
missing = sorted(required - fields.keys())
if missing:
    raise SystemExit(f"anti-collapse line missing fields: {missing}")

shape = int(fields["shape"])
threads = int(fields["threads"])
b = int(fields["B"])
two_b = int(fields["two_B"])
t1 = float(fields["t1"])
t2 = float(fields["t2"])
ratio = float(fields["ratio"])
if shape != 1 or threads != expected_thread:
    raise SystemExit(f"anti-collapse wrong configuration: shape={shape}, threads={threads}")
if b < 1 or two_b != 2 * b or two_b > 100000:
    raise SystemExit(f"anti-collapse invalid batches: B={b}, two_B={two_b}")
if not all(math.isfinite(value) for value in (t1, t2, ratio)):
    raise SystemExit("anti-collapse non-finite timing or ratio")
if t1 < 0.020 or t2 < 0.020:
    raise SystemExit(f"anti-collapse duration too short: t1={t1}, t2={t2}")
if not (1.7 <= ratio <= 2.3):
    raise SystemExit(f"anti-collapse ratio outside [1.7,2.3]: {ratio}")
if fields["checksum_B"] != "PASS" or fields["checksum_2B"] != "PASS":
    raise SystemExit("anti-collapse checksum failure")
if fields["gate"] != "PASS":
    raise SystemExit(f"anti-collapse gate={fields['gate']!r}")


verbose_lines = [
    line for line in lines
    if re.match(r"^(onednn_verbose|dnnl_verbose),", line.strip(), re.IGNORECASE)
]
if verbose_lines:
    raise SystemExit("anti-collapse unexpectedly contains oneDNN verbose output")

print(f"validated anti-collapse gate: B={b}, two_B={two_b}, ratio={ratio:.6f}; verbose=no")
PYANTI
}

validate_verbose_probe_log() {
    local log_path=$1
    local expected_shape=$2
    local expected_thread=$3

    python3 - "$log_path" "$expected_shape" "$expected_thread" <<'PYVERBOSE'
import re
import sys

path, expected_shape_s, expected_thread_s = sys.argv[1:]
expected_shape = int(expected_shape_s)
expected_thread = int(expected_thread_s)
text = open(path, encoding="utf-8", errors="replace").read()
lines = text.splitlines()

config_lines = [line for line in lines if line.startswith("[CONFIG] ")]
onednn_lines = [line for line in lines if line.startswith("[ONEDNN] ")]
env_lines = [line for line in lines if line.startswith("[ENV] ")]
result_lines = [line for line in lines if line.startswith("[CONV2D] ")]
checksum_lines = [line for line in lines if line.startswith("[CHECKSUM] ")]
verbose_lines = [
    line for line in lines
    if re.match(r"^(onednn_verbose|dnnl_verbose),", line.strip(), re.IGNORECASE)
]
conv_verbose = [line for line in verbose_lines if "convolution" in line.lower()]
info_lines = [line for line in verbose_lines if ",info," in line.lower()]

if len(config_lines) != 1 or len(onednn_lines) != 1 or len(env_lines) != 1:
    raise SystemExit(
        "verbose probe must contain exactly one CONFIG, ONEDNN and ENV line"
    )
if len(result_lines) != 1:
    raise SystemExit(f"verbose probe must contain one CONV2D row, got {len(result_lines)}")
if len(checksum_lines) != 2:  # warm-up rep=0 plus one measured repetition
    raise SystemExit(f"verbose probe must contain two CHECKSUM lines, got {len(checksum_lines)}")
if not verbose_lines or not conv_verbose or not info_lines:
    raise SystemExit("oneDNN verbose runtime/implementation output is incomplete")

fields = dict(token.split("=", 1) for token in config_lines[0].split()[1:] if "=" in token)
shape = int(fields["shape"])
requested = int(fields["threads_requested"])
observed = int(fields["threads_observed"])
if shape != expected_shape or requested != expected_thread or observed != expected_thread:
    raise SystemExit(
        f"verbose probe configuration mismatch: shape={shape}, requested={requested}, observed={observed}"
    )

onednn = onednn_lines[0]
for marker in (
    "cpu_threading_runtime=OpenMP",
    "scratchpad_mode=user",
    "execute_api=dnnl_primitive_execute",
    "src_layout=",
    "weight_layout=",
    "dst_layout=",
):
    if marker not in onednn:
        raise SystemExit(f"verbose probe ONEDNN line lacks {marker!r}")

if not any("runtime:openmp" in line.lower() for line in info_lines):
    raise SystemExit("oneDNN verbose did not report OpenMP runtime")
if not any("isa:" in line.lower() for line in info_lines):
    raise SystemExit("oneDNN verbose did not report CPU ISA")
if not all("gate=PASS" in line and "nonfinite=0" in line for line in checksum_lines):
    raise SystemExit("verbose probe checksum failed")

print(
    f"validated compact oneDNN verbose probe: shape={shape}, threads={requested}, "
    f"verbose_lines={len(verbose_lines)}, convolution_lines={len(conv_verbose)}"
)
PYVERBOSE
}

run_verbose_probe() {
    local output=$1
    local log=$2
    local session_id=$3
    local seed=$4

    rm -f -- "$output" "$log"
    env \
        -u GOMP_CPU_AFFINITY \
        -u OMP_NUM_THREADS \
        -u CONV2D_ANTI_COLLAPSE_PROBE \
        OMP_DYNAMIC=FALSE \
        OMP_PROC_BIND=spread \
        OMP_PLACES=cores \
        BENCH_SIZE_FILTER="$VERBOSE_PROBE_SHAPE" \
        BENCH_THREAD_FILTER="$VERBOSE_PROBE_THREAD" \
        CONV2D_DIAGNOSTICS=1 \
        ONEDNN_VERBOSE=1 \
        DNNL_VERBOSE=1 \
        stdbuf -oL -eL \
        "$BIN" "$output" 1 "$session_id" "$seed" \
        >"$log" 2>&1

}

run_measurement() {
    local output=$1
    local log=$2
    local reps=$3
    local session_id=$4
    local seed=$5
    local shape_filter=$6
    local thread_filter=$7
    local verbose=$8

    # GNU env requires all options (including -u) before the first
    # NAME=VALUE assignment.  Keep options and assignments in separate arrays
    # so the full-session path cannot accidentally treat "-u" as a command.
    local -a env_opts=(
        -u GOMP_CPU_AFFINITY
        -u OMP_NUM_THREADS
        -u CONV2D_ANTI_COLLAPSE_PROBE
        -u CONV2D_DIAGNOSTICS
    )
    local -a env_vars=(
        OMP_DYNAMIC=FALSE
        OMP_PROC_BIND=spread
        OMP_PLACES=cores
    )

    if [[ -n "$shape_filter" ]]; then
        env_vars+=(BENCH_SIZE_FILTER="$shape_filter")
    else
        env_opts+=(-u BENCH_SIZE_FILTER)
    fi
    if [[ -n "$thread_filter" ]]; then
        env_vars+=(BENCH_THREAD_FILTER="$thread_filter")
    else
        env_opts+=(-u BENCH_THREAD_FILTER)
    fi
    if (( verbose == 1 )); then
        env_vars+=(ONEDNN_VERBOSE=1 DNNL_VERBOSE=1)
    else
        env_opts+=(-u ONEDNN_VERBOSE -u DNNL_VERBOSE)
    fi

    env "${env_opts[@]}" "${env_vars[@]}" \
        stdbuf -oL -eL \
        "$BIN" "$output" "$reps" "$session_id" "$seed" \
        2>&1 | tee "$log"
}

run_anti_collapse() {
    local output=$1
    local log=$2
    local session_id=$3
    local seed=$4

    rm -f -- "$output"
    env \
        -u GOMP_CPU_AFFINITY \
        -u OMP_NUM_THREADS \
        -u BENCH_SIZE_FILTER \
        -u BENCH_THREAD_FILTER \
        -u ONEDNN_VERBOSE \
        -u DNNL_VERBOSE \
        -u CONV2D_DIAGNOSTICS \
        OMP_DYNAMIC=FALSE \
        OMP_PROC_BIND=spread \
        OMP_PLACES=cores \
        CONV2D_ANTI_COLLAPSE_PROBE=1 \
        stdbuf -oL -eL \
        "$BIN" "$output" 1 "$session_id" "$seed" \
        2>&1 | tee "$log"

    [[ ! -e "$output" ]] || die "anti-collapse mode unexpectedly created CSV: $output"
}

append_file_hash() {
    local manifest=$1
    local label=$2
    local path=$3
    printf '%s_sha256=%s\n' "$label" "$(sha256sum "$path" | awk '{print $1}')" >> "$manifest"
    printf '%s_path=%s\n' "$label" "$path" >> "$manifest"
}

stamp=$(date +%Y%m%d_%H%M%S)
campaign_id="conv2d_${PLATFORM_LC}_${stamp}"
manifest="$RUN_DIR/${campaign_id}_manifest.txt"

{
    echo "campaign_id=$campaign_id"
    echo "platform=$PLATFORM"
    echo "workload=CONV2D"
    echo "schema_version=cpu-gpu-v2"
    echo "created_at=$(date --iso-8601=seconds)"
    echo "hostname=$(hostname)"
    echo "kernel=$(uname -srmo)"
    echo "compiler=$($CXX --version | head -n1)"
    echo "all_shapes=$ALL_SHAPES"
    echo "all_threads=$ALL_THREADS"
    echo "sessions=$SESSIONS"
    echo "repetitions=$REPS"
    echo "quick_repetitions=$QUICK_REPS"
    echo "session_pause_s=$SESSION_PAUSE_S"
    echo "power_off_at_end=$POWER_OFF_AT_END"
    echo "quickcheck_only=$QUICKCHECK_ONLY"
    echo "verbose_probe_shape=$VERBOSE_PROBE_SHAPE"
    echo "verbose_probe_thread=$VERBOSE_PROBE_THREAD"
    echo "source_sha256=$(sha256sum "$SRC" | awk '{print $1}')"
    echo "header_sha256=$(sha256sum "$HEADER" | awk '{print $1}')"
    echo "runner_sha256=$(sha256sum "${BASH_SOURCE[0]}" | awk '{print $1}')"
    echo "binary_sha256=$(sha256sum "$BIN" | awk '{print $1}')"
    if git -C "$PROJECT_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        echo "git_commit=$(git -C "$PROJECT_DIR" rev-parse HEAD)"
        if [[ -n "$(git -C "$PROJECT_DIR" status --porcelain)" ]]; then
            echo "git_dirty=yes"
        else
            echo "git_dirty=no"
        fi
    else
        echo "git_commit=unavailable"
        echo "git_dirty=unavailable"
    fi
    echo "--- ldd ---"
    echo "$LDD_OUTPUT"
    echo "--- cpu ---"
    lscpu 2>/dev/null || true
    if command -v numactl >/dev/null 2>&1; then
        echo "--- numactl --hardware ---"
        numactl --hardware 2>&1 || true
        echo "--- numactl --show ---"
        numactl --show 2>&1 || true
    fi
} > "$manifest"


verbose_id="${campaign_id}_verbose_probe"
verbose_csv="$RUN_DIR/${verbose_id}.csv"
verbose_log="$RUN_DIR/${verbose_id}.log"

echo "[quickcheck] oneDNN preflight"
run_verbose_probe "$verbose_csv" "$verbose_log" "$verbose_id" "$VERBOSE_PROBE_SEED"
validate_csv "$verbose_csv" "$VERBOSE_PROBE_SHAPE" "$VERBOSE_PROBE_THREAD" 1 "$verbose_id"
validate_verbose_probe_log "$verbose_log" "$VERBOSE_PROBE_SHAPE" "$VERBOSE_PROBE_THREAD"
append_file_hash "$manifest" "verbose_probe_csv" "$verbose_csv"
append_file_hash "$manifest" "verbose_probe_log" "$verbose_log"
echo "[quickcheck] oneDNN preflight PASS"

quick_id="${campaign_id}_quickcheck"
quick_csv="$RUN_DIR/${quick_id}.csv"
quick_log="$RUN_DIR/${quick_id}.log"
quick_expected=$((6 * 2 * QUICK_REPS))

echo "[quickcheck] shapes=${QUICK_SHAPES}; threads=${QUICK_THREADS}; reps=${QUICK_REPS}; expected_rows=${quick_expected}"
run_measurement \
    "$quick_csv" "$quick_log" "$QUICK_REPS" "$quick_id" "$QUICK_SEED" \
    "$QUICK_SHAPES" "$QUICK_THREADS" 0
validate_csv "$quick_csv" "$QUICK_SHAPES" "$QUICK_THREADS" "$QUICK_REPS" "$quick_id"
validate_measurement_log "$quick_log" "$QUICK_SHAPES" "$QUICK_THREADS" "$QUICK_REPS" 0
append_file_hash "$manifest" "quickcheck_csv" "$quick_csv"
append_file_hash "$manifest" "quickcheck_log" "$quick_log"
echo "[quickcheck] measurement matrix PASS"

anti_id="${campaign_id}_anti_collapse"
anti_csv="$RUN_DIR/${anti_id}.csv"
anti_log="$RUN_DIR/${anti_id}.log"
echo "[quickcheck] anti-collapse"
run_anti_collapse "$anti_csv" "$anti_log" "$anti_id" "$ANTI_SEED"
validate_anti_collapse_log "$anti_log" "$MAX_THREAD"
append_file_hash "$manifest" "anti_collapse_log" "$anti_log"
echo "[quickcheck] anti-collapse PASS"

if (( QUICKCHECK_ONLY == 1 )); then
    POWER_OFF_AT_END=0
    echo "quickcheck_status=PASS" >> "$manifest"
    SUCCESS=1
    exit 0
fi

for ((session=1; session<=SESSIONS; session++)); do
    seed=$((SEED_BASE + session))
    session_id="${campaign_id}_session${session}"
    output="$RUN_DIR/${session_id}.csv"
    log="$RUN_DIR/${session_id}.log"
    expected_rows=$((6 * 9 * REPS))

    echo "[run] ${PLATFORM} CONV2D session ${session}/${SESSIONS}; seed=${seed}; reps=${REPS}; expected_rows=${expected_rows}"
    run_measurement "$output" "$log" "$REPS" "$session_id" "$seed" "" "" 0
    validate_csv "$output" "$ALL_SHAPES" "$ALL_THREADS" "$REPS" "$session_id"
    validate_measurement_log "$log" "$ALL_SHAPES" "$ALL_THREADS" "$REPS" 0
    append_file_hash "$manifest" "session${session}_csv" "$output"
    append_file_hash "$manifest" "session${session}_log" "$log"
    echo "[run] session ${session}/${SESSIONS} PASS"

    if (( session < SESSIONS )); then
        echo "[pause] Sleeping ${SESSION_PAUSE_S}s before the next independent session..."
        sleep "$SESSION_PAUSE_S"
    fi
done

{
    echo "official_status=PASS"
    echo "official_rows_total=$((SESSIONS * 6 * 9 * REPS))"
    echo "completed_at=$(date --iso-8601=seconds)"
} >> "$manifest"

echo "[done] ${PLATFORM} CONV2D: quickcheck, anti-collapse, and ${SESSIONS} validated sessions completed."
echo "[done] Official measurements per shape×thread configuration: $((SESSIONS * REPS))."
SUCCESS=1

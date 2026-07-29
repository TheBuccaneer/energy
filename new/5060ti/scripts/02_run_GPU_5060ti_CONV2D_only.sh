#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)
SRC="$SCRIPT_DIR/CONV2D/main_conv2d.cu"
SIBLING_SRC="$ROOT_DIR/../3090/scripts/CONV2D/main_conv2d.cu"
BUILD_DIR="$SCRIPT_DIR/CONV2D/.build"
BIN="$BUILD_DIR/main_conv2d"
RUN_DIR="$ROOT_DIR/runs/CONV2D"
RESTORE="$SCRIPT_DIR/03_disable_GPU_5060ti.sh"
GPU_LABEL="RTX 5060 Ti"
EXPECTED_GPU="${EXPECTED_GPU:-RTX 5060 Ti}"
GPU_INDEX="${GPU_INDEX:-0}"
REPS="${REPS:-10}"
SESSIONS="${SESSIONS:-5}"
SESSION_PAUSE_S="${SESSION_PAUSE_S:-60}"
POWER_OFF_AT_END="${POWER_OFF_AT_END:-1}"
SEED_BASE=20266000

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
        echo "[done] All ${GPU_LABEL} CONV2D sessions completed. Powering off."
        sudo systemctl poweroff
    elif (( SUCCESS == 1 && status == 0 )); then
        echo "[done] All ${GPU_LABEL} CONV2D sessions completed. POWER_OFF_AT_END=0."
    else
        echo "[abort] ${GPU_LABEL} CONV2D campaign incomplete; no automatic power-off." >&2
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
[[ "$REPS" -eq 10 ]] || die "official CONV2D campaign requires REPS=10"
[[ "$SESSIONS" -eq 5 ]] || die "official CONV2D campaign requires SESSIONS=5"
[[ "$SESSION_PAUSE_S" -eq 60 ]] || die "official CONV2D campaign requires SESSION_PAUSE_S=60"
[[ "$POWER_OFF_AT_END" =~ ^[01]$ ]] || die "POWER_OFF_AT_END must be 0 or 1"
[[ -f "$RESTORE" ]] || die "missing restore script: $RESTORE"

EXPECTED_SOURCE_SHA256="ff62fd03fd51770b89643988371fad4f318e95534423b1df7f23f408748e2ebb"

for command in nvcc python3 stdbuf tee sha256sum ldd nvidia-smi awk cmp git hostname uname sudo; do
    command -v "$command" >/dev/null || die "required command not found: $command"
done

[[ -f "$SRC" ]] || die "missing source: $SRC"
actual_source_sha=$(sha256sum "$SRC" | awk '{print $1}')
[[ "$actual_source_sha" == "$EXPECTED_SOURCE_SHA256" ]] || \
    die "source hash mismatch: got $actual_source_sha; expected $EXPECTED_SOURCE_SHA256"

if [[ -f "$SIBLING_SRC" ]]; then
    cmp -s "$SRC" "$SIBLING_SRC" || die "3090/5060 Ti CONV2D sources are not byte-identical"
    echo "[preflight] GPU source byte-identity: PASS"
else
    echo "[preflight] WARNING: sibling GPU source not found; byte-identity not checked: $SIBLING_SRC" >&2
fi

mkdir -p "$BUILD_DIR" "$RUN_DIR"
COMPILE_CMD=(nvcc -O3 -std=c++17 -lineinfo "$SRC" -lcudnn -lnvidia-ml -o "$BIN")
echo "[build] Compiling ${GPU_LABEL} CONV2D..."
printf '[build] command:'
printf ' %q' "${COMPILE_CMD[@]}"
printf '\n'
"${COMPILE_CMD[@]}"
ldd "$BIN" | grep -q 'libcudnn' || die "binary is not linked against libcudnn"
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

shapes = {
    1: dict(N=32, C=64,  H=56,  W=56,  K=64,  R=3, S=3, stride=1, pad=1),
    2: dict(N=32, C=64,  H=56,  W=56,  K=128, R=3, S=3, stride=2, pad=1),
    3: dict(N=32, C=128, H=28,  W=28,  K=256, R=3, S=3, stride=2, pad=1),
    4: dict(N=32, C=256, H=14,  W=14,  K=512, R=3, S=3, stride=2, pad=1),
    5: dict(N=32, C=3,   H=224, W=224, K=64,  R=7, S=7, stride=2, pad=3),
    6: dict(N=32, C=256, H=56,  W=56,  K=256, R=1, S=1, stride=1, pad=0),
}
allowed_algorithms = {
    "implicit_gemm",
    "implicit_precomp_gemm",
    "gemm",
    "direct",
    "fft",
    "fft_tiling",
    "winograd",
    "winograd_nonfused",
}
expected_rows = len(shapes) * reps

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

def parse_problem_spec(text, row_index):
    parts = text.split(";")
    expected_keys = [
        "shape_id", "N", "C", "H", "W", "K", "R", "S", "stride", "pad",
        "Hout", "Wout", "layout", "conv", "dtype", "math", "algo",
        "workspace_bytes",
    ]
    if len(parts) != len(expected_keys):
        raise SystemExit(
            f"row {row_index}: problem_spec has {len(parts)} fields; "
            f"expected {len(expected_keys)}: {text!r}"
        )
    parsed = {}
    keys = []
    for part in parts:
        if "=" not in part:
            raise SystemExit(f"row {row_index}: malformed problem_spec token {part!r}")
        key, value = part.split("=", 1)
        keys.append(key)
        parsed[key] = value
    if keys != expected_keys:
        raise SystemExit(
            f"row {row_index}: problem_spec key order mismatch: {keys}"
        )
    return parsed

counts = Counter()
repetition_sets = defaultdict(set)
sequence_values = []
run_ids = []
status_counts = Counter()
timing_warnings = 0
throttle_nonzero = 0
algo_workspace_by_shape = {}
batches_by_shape = defaultdict(list)

for index, row in enumerate(rows, start=1):
    if len(row) != 45:
        raise SystemExit(f"row {index}: expected 45 columns, got {len(row)}")
    if row["schema_version"] != "cpu-gpu-v2":
        raise SystemExit(f"row {index}: wrong schema_version {row['schema_version']!r}")
    if row["session_id"] != expected_session_id:
        raise SystemExit(f"row {index}: wrong session_id {row['session_id']!r}")
    if row["workload"] != "CONV2D":
        raise SystemExit(f"row {index}: wrong workload {row['workload']!r}")
    if row["implementation"] != "cudnn_convolution_fwd_fp32":
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
    shape_id = integer(row, "problem_size")
    batches = integer(row, "batches")
    sequence_values.append(sequence)
    run_ids.append(run_id)

    if shape_id not in shapes:
        raise SystemExit(f"row {index}: unexpected shape_id={shape_id}")
    if repetition < 1 or repetition > reps:
        raise SystemExit(f"row {index}: invalid repetition={repetition}")
    if batches <= 0 or batches > 100000:
        raise SystemExit(f"row {index}: invalid batches={batches}")

    shape = shapes[shape_id]
    hout = (shape["H"] + 2 * shape["pad"] - shape["R"]) // shape["stride"] + 1
    wout = (shape["W"] + 2 * shape["pad"] - shape["S"]) // shape["stride"] + 1
    spec = parse_problem_spec(row["problem_spec"], index)

    expected_static = {
        "shape_id": str(shape_id),
        "N": str(shape["N"]), "C": str(shape["C"]),
        "H": str(shape["H"]), "W": str(shape["W"]),
        "K": str(shape["K"]), "R": str(shape["R"]), "S": str(shape["S"]),
        "stride": str(shape["stride"]), "pad": str(shape["pad"]),
        "Hout": str(hout), "Wout": str(wout),
        "layout": "NCHW", "conv": "cross_correlation",
        "dtype": "f32", "math": "FMA",
    }
    for key, expected in expected_static.items():
        if spec[key] != expected:
            raise SystemExit(
                f"row {index}: problem_spec {key}={spec[key]!r}, expected {expected!r}"
            )
    if spec["algo"] not in allowed_algorithms:
        raise SystemExit(f"row {index}: unsupported algo={spec['algo']!r}")
    try:
        workspace_bytes = int(spec["workspace_bytes"])
    except ValueError:
        raise SystemExit(
            f"row {index}: invalid workspace_bytes={spec['workspace_bytes']!r}"
        )
    if workspace_bytes < 0:
        raise SystemExit(f"row {index}: negative workspace_bytes={workspace_bytes}")

    algo_workspace = (spec["algo"], workspace_bytes)
    previous_algo_workspace = algo_workspace_by_shape.setdefault(shape_id, algo_workspace)
    if previous_algo_workspace != algo_workspace:
        raise SystemExit(
            f"row {index}: algorithm/workspace changed within shape {shape_id}: "
            f"{previous_algo_workspace} -> {algo_workspace}"
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

    flops_per_op = (
        2 * shape["N"] * shape["K"] * shape["C"] * shape["R"] * shape["S"]
        * hout * wout
    )
    logical_bytes = 4 * (
        shape["N"] * shape["C"] * shape["H"] * shape["W"]
        + shape["K"] * shape["C"] * shape["R"] * shape["S"]
        + shape["N"] * shape["K"] * hout * wout
    )
    expected_flops_total = flops_per_op * batches

    if integer(row, "flops_total") != expected_flops_total:
        raise SystemExit(
            f"row {index}: flops_total={row['flops_total']}, expected={expected_flops_total}"
        )
    if integer(row, "logical_bytes_per_op") != logical_bytes:
        raise SystemExit(
            f"row {index}: logical_bytes_per_op={row['logical_bytes_per_op']}, "
            f"expected={logical_bytes}"
        )

    checks = [
        ("energy_per_op_j", num(row, "energy_per_op_j"), energy / batches),
        ("energy_per_second_j", num(row, "energy_per_second_j"), energy / wall),
        ("energy_per_flop_j", num(row, "energy_per_flop_j"), energy / expected_flops_total),
        ("time_per_op_ms_kernel", num(row, "time_per_op_ms_kernel"), 1000.0 * kernel / batches),
        ("time_per_op_ms_e2e", num(row, "time_per_op_ms_e2e"), 1000.0 * e2e / batches),
        ("gflops_per_s", num(row, "gflops_per_s"), expected_flops_total / kernel / 1.0e9),
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

    for key in (
        "pcie_gen", "pcie_width", "sm_clock_mhz", "clock_before_mhz",
        "clock_after_mhz", "mem_clock_mhz", "temp_c", "temp_before_c",
        "temp_after_c",
    ):
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

    counts[shape_id] += 1
    repetition_sets[shape_id].add(repetition)
    batches_by_shape[shape_id].append((repetition, batches))

if sequence_values != list(range(1, expected_rows + 1)):
    raise SystemExit(f"sequence_index must be 1..{expected_rows}")
if run_ids != sequence_values:
    raise SystemExit("run_id_global must equal sequence_index within each session")

for shape_id in shapes:
    if counts[shape_id] != reps:
        raise SystemExit(
            f"coverage shape={shape_id}: got {counts[shape_id]} rows, expected {reps}"
        )
    if repetition_sets[shape_id] != set(range(1, reps + 1)):
        raise SystemExit(
            f"repetition set shape={shape_id}: {sorted(repetition_sets[shape_id])}"
        )
    ordered = sorted(batches_by_shape[shape_id])
    batch_values = [batches for _, batches in ordered]
    if batch_values != sorted(batch_values):
        raise SystemExit(
            f"shape={shape_id}: batches decreased across repetitions: {batch_values}"
        )

algo_summary = ", ".join(
    f"shape{shape_id}:{algo}/{workspace}B"
    for shape_id, (algo, workspace) in sorted(algo_workspace_by_shape.items())
)
print(
    f"validated {len(rows)} CONV2D rows across {len(counts)} shapes; "
    f"runtime_status={dict(status_counts)}; "
    f"minor_timing_warnings={timing_warnings}; throttle_nonzero={throttle_nonzero}"
)
print(f"algorithms: {algo_summary}")
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
manifest="$RUN_DIR/conv2d_5060ti_${stamp}_manifest.txt"
[[ ! -e "$manifest" ]] || die "manifest already exists: $manifest"

repo_root=$(git -C "$ROOT_DIR" rev-parse --show-toplevel 2>/dev/null || true)
git_commit=$(git -C "$ROOT_DIR" rev-parse HEAD 2>/dev/null || echo unavailable)
git_status=$(git -C "$ROOT_DIR" status --porcelain 2>/dev/null || true)
nvcc_line=$(nvcc --version | tail -n 1)
driver_line=$(nvidia-smi --query-gpu=name,driver_version,pci.bus_id --format=csv,noheader -i "$GPU_INDEX" 2>/dev/null || echo unavailable)
cudnn_line=$(ldd "$BIN" | grep 'libcudnn' | head -n 1 || echo unavailable)
compile_text=$(printf '%q ' "${COMPILE_CMD[@]}")

{
    echo "workload=CONV2D"
    echo "implementation=cudnn_convolution_fwd_fp32"
    echo "execution_mode=gpu_resident"
    echo "platform=5060ti"
    echo "gpu_label=${GPU_LABEL}"
    echo "expected_gpu=${EXPECTED_GPU}"
    echo "gpu_index=${GPU_INDEX}"
    echo "campaign_stamp=${stamp}"
    echo "sessions=${SESSIONS}"
    echo "repetitions=${REPS}"
    echo "session_pause_s=${SESSION_PAUSE_S}"
    echo "shape_ids=1,2,3,4,5,6"
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
    echo "cudnn_link=${cudnn_line}"
    echo "gpu_driver=${driver_line}"
    echo "seed_base=${SEED_BASE}"
    echo "seed_rule=${SEED_BASE}+session"
} > "$manifest"

echo "[run] No quickcheck is executed by this 02 runner."
echo "[run] Manifest: $manifest"

for ((session=1; session<=SESSIONS; session++)); do
    seed=$((SEED_BASE + session))
    session_id="conv2d_5060ti_${stamp}_session${session}"
    output="$RUN_DIR/${session_id}.csv"
    log="$RUN_DIR/${session_id}.log"
    [[ ! -e "$output" && ! -e "$log" ]] || die "session output already exists: $session_id"

    echo "[run] ${GPU_LABEL} CONV2D session ${session}/${SESSIONS}; reps=${REPS}; seed=${seed}"
    env \
        -u BENCH_SIZE_FILTER \
        -u CONV2D_ANTI_COLLAPSE_PROBE \
        CUDA_VISIBLE_DEVICES="$GPU_INDEX" \
        NVIDIA_TF32_OVERRIDE=0 \
        BENCH_EXPECTED_GPU="$EXPECTED_GPU" \
        stdbuf -oL -eL \
        "$BIN" "$output" "$REPS" "$session_id" "$seed" \
        2>&1 | tee "$log"

    validate_csv "$output" "$REPS" "$session_id"

    order=$(python3 - "$output" <<'PYORDER'
import csv
import sys

seen = []
with open(sys.argv[1], newline="", encoding="utf-8") as handle:
    for row in csv.DictReader(handle):
        shape_id = int(row["problem_size"])
        if shape_id not in seen:
            seen.append(shape_id)
print(",".join(map(str, seen)))
PYORDER
)
    algorithms=$(python3 - "$output" <<'PYALGO'
import csv
import sys

seen = {}
with open(sys.argv[1], newline="", encoding="utf-8") as handle:
    for row in csv.DictReader(handle):
        shape_id = int(row["problem_size"])
        parts = dict(token.split("=", 1) for token in row["problem_spec"].split(";"))
        seen.setdefault(shape_id, (parts["algo"], parts["workspace_bytes"]))
print(",".join(
    f"shape{shape_id}:{algo}:{workspace}B"
    for shape_id, (algo, workspace) in sorted(seen.items())
))
PYALGO
)
    {
        echo "session_${session}_id=${session_id}"
        echo "session_${session}_seed=${seed}"
        echo "session_${session}_configuration_order=${order}"
        echo "session_${session}_algorithms=${algorithms}"
        echo "session_${session}_csv_sha256=$(sha256sum "$output" | awk '{print $1}')"
        echo "session_${session}_log_sha256=$(sha256sum "$log" | awk '{print $1}')"
    } >> "$manifest"

    echo "[run] Session ${session}/${SESSIONS} PASS: $output"
    if (( session < SESSIONS )); then
        echo "[pause] Sleeping ${SESSION_PAUSE_S} s before the next independent session..."
        sleep "$SESSION_PAUSE_S"
    fi
done

echo "[done] ${GPU_LABEL} CONV2D: ${SESSIONS} validated sessions completed."
echo "[done] Official measurements per shape: $((SESSIONS * REPS))."
SUCCESS=1

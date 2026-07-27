#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)
SRC="$SCRIPT_DIR/AXPY/main_axpy.cu"
SIBLING_SRC="$ROOT_DIR/../3090/scripts/AXPY/main_axpy.cu"
BUILD_DIR="$SCRIPT_DIR/AXPY/.build"
BIN="$BUILD_DIR/main_axpy"
RUN_DIR="$ROOT_DIR/runs/AXPY"
GPU_LABEL="RTX 5060 Ti"
EXPECTED_GPU="${EXPECTED_GPU:-RTX 5060 Ti}"
GPU_INDEX="${GPU_INDEX:-0}"
QUICK_REPS="${QUICK_REPS:-2}"
QUICK_SIZES="1000000,64000000,256000000"
QUICK_SEED=$((0x4158505900000900))

[[ "$QUICK_REPS" =~ ^[0-9]+$ ]] || { echo "ERROR: QUICK_REPS must be numeric" >&2; exit 2; }
[[ "$QUICK_REPS" -eq 2 ]] || { echo "ERROR: AXPY GPU quickcheck requires QUICK_REPS=2" >&2; exit 2; }

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

validate_quickcheck_log() {
    local log_path=$1

    python3 - "$log_path" <<'PYLOG'
import math
import re
import sys

path = sys.argv[1]
text = open(path, encoding="utf-8", errors="replace").read()

result_lines = [line for line in text.splitlines() if line.startswith("[AXPY] ")]
error_lines = [line for line in text.splitlines() if "max_abs_error=" in line]
if len(result_lines) != 6:
    raise SystemExit(f"quickcheck log: got {len(result_lines)} [AXPY] rows, expected 6")
if len(error_lines) != 6:
    raise SystemExit(f"quickcheck log: got {len(error_lines)} checksum lines, expected 6")

pattern = re.compile(r"max_abs_error=(?P<abs>\S+)\s+max_rel_error=(?P<rel>\S+)")
for index, line in enumerate(error_lines, start=1):
    match = pattern.search(line)
    if not match:
        raise SystemExit(f"checksum line {index}: malformed: {line!r}")
    abs_error = float(match.group("abs"))
    rel_error = float(match.group("rel"))
    if not math.isfinite(abs_error) or not math.isfinite(rel_error):
        raise SystemExit(f"checksum line {index}: non-finite diagnostic")
    if abs_error != 0.0 or rel_error != 0.0:
        raise SystemExit(
            f"checksum line {index}: expected exact zero errors, got {abs_error}, {rel_error}"
        )

probe_lines = [line for line in text.splitlines() if line.startswith("[ANTI_COLLAPSE] ")]
if len(probe_lines) != 1:
    raise SystemExit(f"expected exactly one anti-collapse line, got {len(probe_lines)}")

fields = {}
for token in probe_lines[0].split()[1:]:
    if "=" in token:
        key, value = token.split("=", 1)
        fields[key] = value
required = {
    "N", "device", "B_cal", "B_probe", "two_B_probe", "t1", "t2", "ratio",
    "time_basis", "sm_clock_mhz", "temp_c", "throttle_reasons", "checksum1",
    "checksum2", "gate",
}
missing = sorted(required - fields.keys())
if missing:
    raise SystemExit(f"anti-collapse line missing fields: {missing}")

n = int(fields["N"])
b_cal = int(fields["B_cal"])
b_probe = int(fields["B_probe"])
two_b_probe = int(fields["two_B_probe"])
t1 = float(fields["t1"])
t2 = float(fields["t2"])
ratio = float(fields["ratio"])

if n != 1_000_000 or fields["device"] != "gpu":
    raise SystemExit(f"anti-collapse wrong configuration: N={n}, device={fields['device']!r}")
if not (1 <= b_cal <= 250000):
    raise SystemExit(f"anti-collapse invalid B_cal={b_cal}")
if b_probe < 100 or two_b_probe != 2 * b_probe or two_b_probe > 250000:
    raise SystemExit(f"anti-collapse invalid probe batches: B={b_probe}, 2B={two_b_probe}")
if not all(math.isfinite(value) for value in (t1, t2, ratio)):
    raise SystemExit("anti-collapse non-finite timing or ratio")
if t1 < 0.020 or t2 < 0.020:
    raise SystemExit(f"anti-collapse duration too short: t1={t1}, t2={t2}")
if not (1.7 <= ratio <= 2.3):
    raise SystemExit(f"anti-collapse ratio outside [1.7,2.3]: {ratio}")
if fields["time_basis"] != "kernel_time_s":
    raise SystemExit(f"anti-collapse wrong time basis: {fields['time_basis']!r}")
if int(fields["sm_clock_mhz"]) <= 0 or int(fields["temp_c"]) <= 0:
    raise SystemExit("anti-collapse telemetry must be positive")
if not fields["throttle_reasons"].lower().startswith("0x"):
    raise SystemExit("anti-collapse throttle_reasons malformed")
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


stamp=$(date +%Y%m%d_%H%M%S)
quick_id="axpy_5060ti_${stamp}_quickcheck"
quick_csv="$RUN_DIR/${quick_id}.csv"
quick_log="$RUN_DIR/${quick_id}.log"
[[ ! -e "$quick_csv" && ! -e "$quick_log" ]] || die "quickcheck output already exists"

cat <<EOF
[quickcheck] GPU=${GPU_LABEL}
[quickcheck] sizes=${QUICK_SIZES}; reps=${QUICK_REPS}
[quickcheck] anti-collapse=enabled; time_basis=kernel_time_s
[quickcheck] no automatic shutdown
EOF

env \
    CUDA_VISIBLE_DEVICES="$GPU_INDEX" \
    NVIDIA_TF32_OVERRIDE=0 \
    BENCH_EXPECTED_GPU="$EXPECTED_GPU" \
    BENCH_SIZE_FILTER="$QUICK_SIZES" \
    AXPY_ANTI_COLLAPSE_PROBE=1 \
    stdbuf -oL -eL \
    "$BIN" "$quick_csv" "$QUICK_REPS" "$quick_id" "$QUICK_SEED" \
    2>&1 | tee "$quick_log"

validate_csv "$quick_csv" "$QUICK_REPS" "$quick_id"
validate_quickcheck_log "$quick_log"

echo "[quickcheck] PASS: $quick_csv"
echo "[quickcheck] Log:  $quick_log"
echo "[quickcheck] GPU settings were not restored and the machine will not power off."

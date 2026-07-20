#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
GPU_INDEX=${GPU_INDEX:-0}
EXPECTED_GPU=${EXPECTED_GPU:-RTX 5060 Ti}
SRC="$ROOT/scripts/STRIDED_GEMM/main_strided_gemm.cu"
BIN="$ROOT/scripts/STRIDED_GEMM/main_strided_gemm"
OUT="$ROOT/runs/STRIDED_GEMM/strided_gemm_5060ti_quickcheck.csv"

[[ -f "$SRC" ]] || { echo "ERROR: missing $SRC" >&2; exit 1; }

mkdir -p "$(dirname "$OUT")"
rm -f "$OUT"

nvcc -O3 -std=c++17 -lineinfo \
    "$SRC" -lcublas -lnvidia-ml -o "$BIN"

CUDA_VISIBLE_DEVICES="$GPU_INDEX" \
NVIDIA_TF32_OVERRIDE=0 \
BENCH_EXPECTED_GPU="$EXPECTED_GPU" \
BENCH_SIZE_FILTER=4096 \
stdbuf -oL -eL \
    "$BIN" "$OUT" 3 quickcheck_strided_5060ti 20264999

python3 - "$OUT" "$EXPECTED_GPU" <<'PY'
import csv
import math
import sys

path = sys.argv[1]
expected_gpu = sys.argv[2]
expected_header = [
    "schema_version", "timestamp", "session_id", "sequence_index", "run_id_global",
    "repetition", "workload", "implementation", "execution_mode", "device_name",
    "num_threads", "problem_size", "problem_spec", "batches", "e2e_time_s",
    "kernel_time_s", "wall_time_s", "device_energy_j", "total_energy_j",
    "dram_energy_j", "energy_per_op_j", "energy_per_second_j", "energy_per_flop_j",
    "time_per_op_ms_kernel", "time_per_op_ms_e2e", "flops_total", "gflops_per_s",
    "logical_bytes_per_op", "avg_power_w", "runtime_status", "pcie_gen", "pcie_width",
    "sm_clock_mhz", "clock_before_mhz", "clock_after_mhz", "mem_clock_mhz", "temp_c",
    "temp_before_c", "temp_after_c", "throttle_reasons", "cpu_cycles",
    "cpu_instructions", "cpu_ipc", "cpu_cache_misses", "checksum_ok",
]

with open(path, newline="", encoding="utf-8") as handle:
    reader = csv.DictReader(handle)
    assert reader.fieldnames == expected_header, (
        f"unexpected CSV header ({len(reader.fieldnames or [])} columns)"
    )
    rows = list(reader)

assert len(rows) == 3, f"expected 3 rows, got {len(rows)}"

positive_fields = [
    "batches", "e2e_time_s", "kernel_time_s", "wall_time_s", "device_energy_j",
    "total_energy_j", "energy_per_op_j", "energy_per_second_j", "energy_per_flop_j",
    "time_per_op_ms_kernel", "time_per_op_ms_e2e", "flops_total", "gflops_per_s",
    "logical_bytes_per_op", "avg_power_w",
]
integer_positive_fields = [
    "batches", "pcie_gen", "pcie_width", "sm_clock_mhz", "clock_before_mhz",
    "clock_after_mhz", "mem_clock_mhz", "temp_c", "temp_before_c", "temp_after_c",
]

for index, row in enumerate(rows, start=1):
    assert len(row) == 45, f"row {index}: expected 45 columns, got {len(row)}"
    assert row["schema_version"] == "cpu-gpu-v2"
    assert row["workload"] == "STRIDED_GEMM"
    assert row["implementation"] == "cublas_gemm_ex_fp32_pedantic_ld2n"
    assert row["execution_mode"] == "gpu_resident"
    assert expected_gpu.lower() in row["device_name"].lower(), (
        f"row {index}: unexpected GPU {row['device_name']!r}"
    )
    assert row["num_threads"] == "-1"
    assert row["problem_size"] == "4096"
    assert row["problem_spec"] == "N=4096;ld=8192"
    assert row["runtime_status"] in {"below", "in_range", "above"}
    assert row["checksum_ok"].lower() == "t"

    for field in positive_fields:
        value = float(row[field])
        assert math.isfinite(value) and value > 0.0, (
            f"row {index}: {field} must be finite and positive, got {row[field]!r}"
        )

    for field in integer_positive_fields:
        assert int(float(row[field])) > 0, (
            f"row {index}: {field} must be positive, got {row[field]!r}"
        )

    assert float(row["dram_energy_j"]) == -1.0
    assert row["cpu_cycles"] == "-1"
    assert row["cpu_instructions"] == "-1"
    assert float(row["cpu_ipc"]) == -1.0
    assert row["cpu_cache_misses"] == "-1"
    assert math.isclose(
        float(row["device_energy_j"]),
        float(row["total_energy_j"]),
        rel_tol=0.0,
        abs_tol=1.0e-6,
    )

    kernel = float(row["kernel_time_s"])
    wall = float(row["wall_time_s"])
    assert kernel <= wall * 1.05, (
        f"row {index}: kernel time {kernel} exceeds wall time {wall} implausibly"
    )

    throttle_text = row["throttle_reasons"]
    assert throttle_text.lower().startswith("0x")
    throttle_value = int(throttle_text, 16)
    severe_mask = 0x8 | 0x20 | 0x40 | 0x80
    assert throttle_value & severe_mask == 0, (
        f"row {index}: severe throttle reason(s): {throttle_text}"
    )

    print(
        f"N={row['problem_size']} ld=8192 rep={row['repetition']} "
        f"batches={row['batches']} "
        f"kernel_ms={1000 * kernel:.3f} "
        f"e2e_ms={1000 * float(row['e2e_time_s']):.3f} "
        f"GFLOP/s={float(row['gflops_per_s']):.2f} "
        f"energy_J={float(row['device_energy_j']):.3f} "
        f"power_W={float(row['avg_power_w']):.1f} "
        f"temp_C={row['temp_c']} throttle={throttle_text} "
        f"checksum={row['checksum_ok']}"
    )

print("Quickcheck PASS:", path)
PY

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
GPU_INDEX=${GPU_INDEX:-0}
EXPECTED_GPU=${EXPECTED_GPU:-RTX 5060 Ti}
SRC="$ROOT/scripts/STREAM/main_stream.cu"
BIN="$ROOT/scripts/STREAM/main_stream"
OUT="$ROOT/runs/STREAM/stream_5060ti_quickcheck.csv"

[[ -f "$SRC" ]] || { echo "ERROR: missing $SRC" >&2; exit 1; }

mkdir -p "$(dirname "$OUT")"
rm -f "$OUT"

nvcc -O3 -std=c++17 -lineinfo \
    "$SRC" -lnvidia-ml -o "$BIN"

CUDA_VISIBLE_DEVICES="$GPU_INDEX" \
NVIDIA_TF32_OVERRIDE=0 \
BENCH_EXPECTED_GPU="$EXPECTED_GPU" \
BENCH_SIZE_FILTER=64000000 \
stdbuf -oL -eL \
    "$BIN" "$OUT" 3 quickcheck_stream_5060ti 20265999

python3 - "$OUT" "$EXPECTED_GPU" <<'PY'
import csv
import math
import sys

path = sys.argv[1]
expected_gpu = sys.argv[2]
expected_n = 64_000_000
expected_logical_bytes = 12.0 * expected_n
expected_flops_per_op = 2.0 * expected_n

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
    assert row["workload"] == "STREAM"
    assert row["implementation"] == "cuda_stream_triad_fp32"
    assert row["execution_mode"] == "gpu_resident"
    assert expected_gpu.lower() in row["device_name"].lower(), (
        f"row {index}: unexpected GPU {row['device_name']!r}"
    )
    assert row["num_threads"] == "-1"
    assert int(row["problem_size"]) == expected_n
    assert row["problem_spec"] == f"elements={expected_n}"
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

    batches = int(row["batches"])
    kernel = float(row["kernel_time_s"])
    wall = float(row["wall_time_s"])
    e2e = float(row["e2e_time_s"])
    energy = float(row["device_energy_j"])
    flops_total = float(row["flops_total"])
    logical_bytes = float(row["logical_bytes_per_op"])

    assert kernel <= wall * 1.05, (
        f"row {index}: kernel time {kernel} exceeds wall time {wall} implausibly"
    )
    assert math.isclose(e2e, wall, rel_tol=0.0, abs_tol=1.0e-6)
    assert math.isclose(float(row["total_energy_j"]), energy, rel_tol=0.0, abs_tol=1.0e-6)
    assert math.isclose(logical_bytes, expected_logical_bytes, rel_tol=0.0, abs_tol=1.0)
    assert math.isclose(
        flops_total,
        expected_flops_per_op * batches,
        rel_tol=1.0e-12,
        abs_tol=1.0,
    )
    assert math.isclose(
        float(row["energy_per_op_j"]),
        energy / batches,
        rel_tol=1.0e-6,
        abs_tol=1.0e-9,
    )
    assert math.isclose(
        float(row["time_per_op_ms_kernel"]),
        1000.0 * kernel / batches,
        rel_tol=2.0e-6,
        abs_tol=1.0e-8,
    )
    assert math.isclose(
        float(row["gflops_per_s"]),
        flops_total / kernel / 1.0e9,
        rel_tol=2.0e-6,
        abs_tol=1.0e-6,
    )

    throttle_text = row["throttle_reasons"]
    assert throttle_text.lower().startswith("0x")
    throttle_value = int(throttle_text, 16)
    severe_mask = 0x8 | 0x20 | 0x40 | 0x80
    assert throttle_value & severe_mask == 0, (
        f"row {index}: severe throttle reason(s): {throttle_text}"
    )

    logical_bandwidth_gb_s = logical_bytes * batches / kernel / 1.0e9
    print(
        f"N={expected_n} rep={row['repetition']} batches={batches} "
        f"kernel_ms={1000 * kernel:.3f} e2e_ms={1000 * e2e:.3f} "
        f"logical_BW={logical_bandwidth_gb_s:.1f} GB/s "
        f"energy_J={energy:.3f} power_W={float(row['avg_power_w']):.1f} "
        f"temp_C={row['temp_c']} pcie=x{row['pcie_width']} "
        f"throttle={throttle_text} checksum={row['checksum_ok']}"
    )

print("Quickcheck PASS:", path)
PY

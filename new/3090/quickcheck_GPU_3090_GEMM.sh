#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")" && pwd)
GPU_INDEX=${GPU_INDEX:-0}
EXPECTED_GPU=${EXPECTED_GPU:-RTX 3090}
SRC="$ROOT/scripts/GEMM/main_gemm.cu"
BIN="$ROOT/scripts/GEMM/main_gemm"
OUT="$ROOT/runs/GEMM/gemm_3090_quickcheck.csv"

mkdir -p "$(dirname "$OUT")"
rm -f "$OUT"

nvcc -O3 -std=c++17 -lineinfo "$SRC" -lcublas -lnvidia-ml -o "$BIN"

CUDA_VISIBLE_DEVICES="$GPU_INDEX" \
NVIDIA_TF32_OVERRIDE=0 \
BENCH_EXPECTED_GPU="$EXPECTED_GPU" \
BENCH_SIZE_FILTER=4096 \
    "$BIN" "$OUT" 3 quickcheck_3090 20263999

python3 - "$OUT" <<'PY'
import csv, sys
path = sys.argv[1]
rows = list(csv.DictReader(open(path, newline="")))
assert len(rows) == 3, f"expected 3 rows, got {len(rows)}"
for r in rows:
    print(
        f"N={r['problem_size']} rep={r['repetition']} "
        f"kernel_ms={1000*float(r['kernel_time_s']):.3f} "
        f"e2e_ms={1000*float(r['e2e_time_s']):.3f} "
        f"GFLOP/s={float(r['gflops_per_s']):.2f} "
        f"energy_J={float(r['device_energy_j']):.3f} "
        f"power_W={float(r['avg_power_w']):.1f} "
        f"temp_C={r['temp_c']} checksum={r['checksum_ok']}"
    )
assert all(r['checksum_ok'].lower() in {'t','true','1'} for r in rows)
assert all(float(r['device_energy_j']) > 0 for r in rows)
print("Quickcheck PASS:", path)
PY

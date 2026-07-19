#!/usr/bin/env bash
set -euo pipefail
ROOT=${1:-$PWD}
if [ -d "$ROOT/scripts/GEMM/CPU/INTEL" ]; then PLATFORM=INTEL; OUTBASE="$ROOT/runs/GEMM/CPU/INTEL"
elif [ -d "$ROOT/scripts/GEMM/CPU/AMD" ]; then PLATFORM=AMD; OUTBASE="$ROOT/runs2/GEMM/CPU/AMD"
else echo "ERROR: project root not found" >&2; exit 1; fi
COMMON="$ROOT/scripts/common"; SRC="$ROOT/scripts/GEMM/CPU/$PLATFORM/main_gemm.cpp"; BIN="$ROOT/scripts/GEMM/CPU/$PLATFORM/main_gemm"
mkdir -p "$OUTBASE"
g++ -O3 -march=native -std=c++17 -I"$COMMON" "$SRC" -lopenblas -lpthread -lm -o "$BIN"
echo "Linked libraries:"; ldd "$BIN" | grep -E 'openblas|gomp|omp|pthread' || true
OUT="$OUTBASE/gemm_openblas_scaling_quickcheck.csv"; rm -f "$OUT"
BENCH_SIZE_FILTER=4096 BENCH_THREAD_FILTER=1,4,10 env -u OMP_PROC_BIND -u OMP_PLACES -u GOMP_CPU_AFFINITY OMP_DYNAMIC=FALSE stdbuf -oL -eL "$BIN" "$OUT" 1 openblas_fix_check 424242
python3 - "$OUT" <<'PYSUMMARY'
import csv,sys
rows=list(csv.DictReader(open(sys.argv[1])))
for r in sorted(rows,key=lambda x:int(x['num_threads'])):
 print(f"threads={r['num_threads']:>2} time_ms={float(r['time_per_op_ms_e2e']):9.3f} GFLOP/s={float(r['gflops_per_s']):9.2f} power_W={float(r['avg_power_w']):7.1f} checksum={r['checksum_ok']}")
base=next(float(r['time_per_op_ms_e2e']) for r in rows if int(r['num_threads'])==1)
for t in (4,10):
 cur=next(float(r['time_per_op_ms_e2e']) for r in rows if int(r['num_threads'])==t)
 print(f"speedup_{t}={base/cur:.2f}x")
PYSUMMARY
echo "CSV: $OUT"

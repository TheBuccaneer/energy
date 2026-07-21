#!/usr/bin/env bash
set -euo pipefail

PLATFORM="Intel"
PLATFORM_LC="intel"
ALL_THREADS="1,2,4,8,10,16,20"
QUICK_THREADS="1,20"
ENERGY_MODE="intel"
REPS=${REPS:-10}
SESSIONS=${SESSIONS:-5}
QUICK_REPS=${QUICK_REPS:-2}
SEED_BASE=${SEED_BASE:-20263300}
POWER_OFF_AT_END=${POWER_OFF_AT_END:-1}
QUICKCHECK_ONLY=${QUICKCHECK_ONLY:-0}

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)
SRC="$SCRIPT_DIR/REDUCTION/main_reduction_${PLATFORM_LC}.cpp"
COMMON="$SCRIPT_DIR/common"
BUILD_DIR="$SCRIPT_DIR/REDUCTION/.build"
BIN="$BUILD_DIR/main_reduction_${PLATFORM_LC}"
RUN_DIR="$ROOT_DIR/runs/REDUCTION"
RESTORE="$SCRIPT_DIR/03_disable_CPU_${PLATFORM}.sh"
SIZES="1000000,2000000,4000000,8000000,16000000,32000000,64000000,128000000,256000000"
QUICK_SIZES="1000000,64000000,256000000"
SUCCESS=0
KEEPALIVE_PID=""

cleanup() {
    local status=$?
    trap - EXIT INT TERM
    if [[ -n "$KEEPALIVE_PID" ]]; then
        kill "$KEEPALIVE_PID" 2>/dev/null || true
        wait "$KEEPALIVE_PID" 2>/dev/null || true
    fi
    echo "[cleanup] Restoring ${PLATFORM} CPU settings and RAPL permissions..."
    if ! sudo bash "$RESTORE"; then status=1; fi
    if (( SUCCESS == 1 && status == 0 && POWER_OFF_AT_END == 1 )); then
        echo "[done] REDUCTION quickcheck and five validated sessions completed. Powering off."
        sudo systemctl poweroff
    elif (( SUCCESS == 1 && status == 0 )); then
        echo "[done] REDUCTION completed. POWER_OFF_AT_END=0; staying online."
    else
        echo "[abort] REDUCTION campaign incomplete or failed; no automatic power-off." >&2
    fi
    exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

for required in "$SRC" "$COMMON/benchmark_common.hpp" "$RESTORE"; do
    [[ -f "$required" ]] || { echo "ERROR: missing required file: $required" >&2; exit 2; }
done
for command in g++ python3 stdbuf tee ldd; do
    command -v "$command" >/dev/null || { echo "ERROR: required command not found: $command" >&2; exit 2; }
done
[[ "$SESSIONS" -eq 5 ]] || { echo "ERROR: official campaign requires SESSIONS=5." >&2; exit 2; }
[[ "$REPS" -eq 10 ]] || { echo "ERROR: official campaign requires REPS=10." >&2; exit 2; }
[[ "$QUICK_REPS" -eq 2 ]] || { echo "ERROR: REDUCTION quickcheck requires QUICK_REPS=2." >&2; exit 2; }
[[ "$QUICKCHECK_ONLY" =~ ^[01]$ ]] || { echo "ERROR: QUICKCHECK_ONLY must be 0 or 1." >&2; exit 2; }

mkdir -p "$BUILD_DIR" "$RUN_DIR"
echo "[build] Compiling ${PLATFORM} REDUCTION with OpenMP..."
g++ -O3 -march=native -std=c++17 -fopenmp -I"$COMMON" "$SRC" -lpthread -lm -o "$BIN"
ldd "$BIN" | grep -q 'libgomp' || { echo "ERROR: libgomp is not linked." >&2; exit 2; }
if ldd "$BIN" | grep -qi 'openblas'; then echo "ERROR: REDUCTION must not link against OpenBLAS." >&2; exit 2; fi

echo "[preflight] Build OK; OpenMP linked; OpenBLAS absent."
sudo -v
( while true; do sudo -n true 2>/dev/null || exit; sleep 60; kill -0 "$$" 2>/dev/null || exit; done ) &
KEEPALIVE_PID=$!

validate_csv() {
    local path=$1 sizes=$2 threads=$3 reps=$4
    python3 - "$path" "$sizes" "$threads" "$reps" "$ENERGY_MODE" <<'PYVALID'
import csv, math, sys
from collections import Counter, defaultdict
path, sizes_s, threads_s, reps_s, energy_mode = sys.argv[1:]
sizes={int(x) for x in sizes_s.split(',') if x}; threads={int(x) for x in threads_s.split(',') if x}; reps=int(reps_s)
header=['schema_version','timestamp','session_id','sequence_index','run_id_global','repetition','workload','implementation','execution_mode','device_name','num_threads','problem_size','problem_spec','batches','e2e_time_s','kernel_time_s','wall_time_s','total_energy_j','device_energy_j','dram_energy_j','energy_per_op_j','energy_per_second_j','energy_per_flop_j','time_per_op_ms_kernel','time_per_op_ms_e2e','flops_total','gflops_per_s','logical_bytes_per_op','avg_power_w','runtime_status','pcie_gen','pcie_width','sm_clock_mhz','clock_before_mhz','clock_after_mhz','mem_clock_mhz','temp_c','temp_before_c','temp_after_c','throttle_reasons','cpu_cycles','cpu_instructions','cpu_ipc','cpu_cache_misses','checksum_ok']
with open(path,newline='',encoding='utf-8') as f:
    rd=csv.DictReader(f); rows=list(rd)
    if rd.fieldnames!=header: raise SystemExit(f'header mismatch: {rd.fieldnames}')
if len(rows)!=len(sizes)*len(threads)*reps: raise SystemExit(f'row-count mismatch: got {len(rows)}')
def num(r,k):
    try:v=float(r[k])
    except Exception:raise SystemExit(f'invalid {k}: {r.get(k)!r}')
    if not math.isfinite(v):raise SystemExit(f'non-finite {k}: {v}')
    return v
def close(a,b,rel=2e-3,abs_=1e-9):return math.isclose(a,b,rel_tol=rel,abs_tol=abs_)
def truth(v):return str(v).strip().lower() in {'1','true','yes','t'}
counts=Counter(); seen=defaultdict(set)
for i,r in enumerate(rows,1):
    if r['schema_version']!='cpu-gpu-v2' or r['workload']!='REDUCTION' or r['implementation']!='openmp_blocked_sum_fp32' or r['execution_mode']!='cpu_native':raise SystemExit(f'row {i}: identity failed')
    n=int(r['problem_size']); t=int(r['num_threads']); rep=int(r['repetition']); b=int(r['batches'])
    if n not in sizes or t not in threads or not(1<=rep<=reps) or b<=0:raise SystemExit(f'row {i}: unexpected configuration')
    if r['problem_spec']!=f'elements={n}' or not truth(r['checksum_ok']):raise SystemExit(f'row {i}: spec/checksum failed')
    e2e=num(r,'e2e_time_s'); ker=num(r,'kernel_time_s'); wall=num(r,'wall_time_s'); total=num(r,'total_energy_j'); device=num(r,'device_energy_j'); dram=num(r,'dram_energy_j')
    if min(e2e,ker,wall,total,device)<=0 or not close(ker,e2e,rel=2e-5) or not close(wall,e2e,rel=2e-5):raise SystemExit(f'row {i}: invalid CPU time/energy')
    expected_total=device if dram==-1 else device+dram
    if energy_mode=='intel' and (dram < -1 or not close(total,expected_total)):raise SystemExit(f'row {i}: Intel energy semantics failed')
    if energy_mode=='amd' and (dram!=-1 or not close(total,device)):raise SystemExit(f'row {i}: AMD energy semantics failed')
    flops=(n-1)*b; bytes_=4*n+4
    checks=[(num(r,'flops_total'),flops),(num(r,'logical_bytes_per_op'),bytes_),(num(r,'energy_per_op_j'),total/b),(num(r,'energy_per_second_j'),total/e2e),(num(r,'energy_per_flop_j'),total/flops),(num(r,'time_per_op_ms_kernel'),1000*ker/b),(num(r,'time_per_op_ms_e2e'),1000*e2e/b),(num(r,'gflops_per_s'),flops/ker/1e9),(num(r,'avg_power_w'),total/e2e)]
    if any(not close(a,e,abs_=1e-6) for a,e in checks):raise SystemExit(f'row {i}: derived metric failed')
    status='below' if e2e<.75 else('in_range' if e2e<=1.25 else'above')
    if r['runtime_status']!=status:raise SystemExit(f'row {i}: runtime_status failed')
    counts[(n,t)]+=1;seen[(n,t)].add(rep)
for n in sizes:
  for t in threads:
    if counts[(n,t)]!=reps or seen[(n,t)]!=set(range(1,reps+1)):raise SystemExit(f'coverage failed for N={n}, threads={t}')
print(f'validated {len(rows)} REDUCTION CPU rows')
PYVALID
}

run_reduction() {
    local output=$1 log=$2 reps=$3 session_id=$4 seed=$5 size_filter=$6 thread_filter=$7
    if [[ -n "$size_filter" ]]; then
        env -u GOMP_CPU_AFFINITY OMP_DYNAMIC=FALSE OMP_PROC_BIND=spread OMP_PLACES=cores BENCH_SIZE_FILTER="$size_filter" BENCH_THREAD_FILTER="$thread_filter" stdbuf -oL -eL "$BIN" "$output" "$reps" "$session_id" "$seed" 2>&1 | tee "$log"
    else
        env -u GOMP_CPU_AFFINITY -u BENCH_SIZE_FILTER -u BENCH_THREAD_FILTER OMP_DYNAMIC=FALSE OMP_PROC_BIND=spread OMP_PLACES=cores stdbuf -oL -eL "$BIN" "$output" "$reps" "$session_id" "$seed" 2>&1 | tee "$log"
    fi
}

stamp=$(date +%Y%m%d_%H%M%S)
quick_id="reduction_${PLATFORM_LC}_${stamp}_quickcheck"
quick_csv="$RUN_DIR/${quick_id}.csv"; quick_log="$RUN_DIR/${quick_id}.log"
echo "[quickcheck] sizes=${QUICK_SIZES}; threads=${QUICK_THREADS}; reps=${QUICK_REPS}"
run_reduction "$quick_csv" "$quick_log" "$QUICK_REPS" "$quick_id" "$((SEED_BASE+900))" "$QUICK_SIZES" "$QUICK_THREADS"
validate_csv "$quick_csv" "$QUICK_SIZES" "$QUICK_THREADS" "$QUICK_REPS"
echo "[quickcheck] PASS: $quick_csv"
if (( QUICKCHECK_ONLY == 1 )); then
    echo "[quickcheck] QUICKCHECK_ONLY=1; official sessions were not started."
    POWER_OFF_AT_END=0
    SUCCESS=1
    exit 0
fi
for ((session=1; session<=SESSIONS; session++)); do
    seed=$((SEED_BASE+session-1)); session_id="reduction_${PLATFORM_LC}_${stamp}_session${session}"
    output="$RUN_DIR/${session_id}.csv"; log="$RUN_DIR/${session_id}.log"
    echo "[run] ${PLATFORM} REDUCTION session ${session}/${SESSIONS}; seed=${seed}; reps=${REPS}"
    run_reduction "$output" "$log" "$REPS" "$session_id" "$seed" "" ""
    validate_csv "$output" "$SIZES" "$ALL_THREADS" "$REPS"
    echo "[run] Session ${session}/${SESSIONS} PASS: $output"
done
SUCCESS=1
echo "[done] ${PLATFORM} REDUCTION: quickcheck plus ${SESSIONS} validated sessions completed."

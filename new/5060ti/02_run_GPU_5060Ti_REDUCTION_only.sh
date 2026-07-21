#!/usr/bin/env bash
set -euo pipefail
GPU_LABEL="RTX 5060 Ti"
GPU_SLUG="5060ti"
GPU_INDEX=${GPU_INDEX:-0}
EXPECTED_GPU=${EXPECTED_GPU:-RTX 5060 Ti}
REPS=${REPS:-10}
SESSIONS=${SESSIONS:-5}
QUICK_REPS=${QUICK_REPS:-2}
SEED_BASE=${SEED_BASE:-20265400}
POWER_OFF_AT_END=${POWER_OFF_AT_END:-1}
QUICKCHECK_ONLY=${QUICKCHECK_ONLY:-0}
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SRC="$ROOT/scripts/REDUCTION/main_reduction.cu"
BIN="$ROOT/scripts/REDUCTION/main_reduction"
OUTDIR="$ROOT/runs/REDUCTION"
RESTORE="$ROOT/03_disable_GPU_5060Ti.sh"
for candidate in 03_disable_GPU_5060Ti.sh 03_disable_GPU_5060ti.sh 03_disable_GPU_5060TI.sh; do
    [[ -f "$ROOT/$candidate" ]] && { RESTORE="$ROOT/$candidate"; break; }
done
SIZES="1000000,2000000,4000000,8000000,16000000,32000000,64000000,128000000,256000000"
QUICK_SIZES="1000000,64000000,256000000"
SUCCESS=0
KEEPALIVE_PID=""
cleanup() {
    local status=$?
    trap - EXIT INT TERM
    if [[ -n "$KEEPALIVE_PID" ]]; then kill "$KEEPALIVE_PID" 2>/dev/null || true; wait "$KEEPALIVE_PID" 2>/dev/null || true; fi
    sudo env GPU_INDEX="$GPU_INDEX" bash "$RESTORE" || status=1
    if (( SUCCESS==1 && status==0 && POWER_OFF_AT_END==1 )); then echo "[done] ${GPU_LABEL} REDUCTION completed. Powering off."; sudo systemctl poweroff
    elif (( SUCCESS==1 && status==0 )); then echo "[done] ${GPU_LABEL} REDUCTION completed. POWER_OFF_AT_END=0."
    else echo "[abort] ${GPU_LABEL} REDUCTION failed; GPU restored; no shutdown." >&2; fi
    exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
for required in "$SRC" "$RESTORE"; do [[ -f "$required" ]] || { echo "ERROR: missing required file: $required" >&2; exit 2; }; done
for command in nvcc python3 stdbuf tee ldd; do command -v "$command" >/dev/null || { echo "ERROR: required command not found: $command" >&2; exit 2; }; done
[[ "$SESSIONS" -eq 5 ]] || { echo "ERROR: official campaign requires SESSIONS=5." >&2; exit 2; }
[[ "$REPS" -eq 10 ]] || { echo "ERROR: official campaign requires REPS=10." >&2; exit 2; }
[[ "$QUICK_REPS" -eq 2 ]] || { echo "ERROR: REDUCTION quickcheck requires QUICK_REPS=2." >&2; exit 2; }
[[ "$QUICKCHECK_ONLY" =~ ^[01]$ ]] || { echo "ERROR: QUICKCHECK_ONLY must be 0 or 1." >&2; exit 2; }
mkdir -p "$OUTDIR" "$(dirname "$BIN")"
echo "[build] Compiling ${GPU_LABEL} REDUCTION with CUB/NVML..."
nvcc -O3 -std=c++17 -lineinfo "$SRC" -lnvidia-ml -o "$BIN"
ldd "$BIN" | grep -E 'nvidia-ml|cuda' || true
sudo -v
( while true; do sudo -n true 2>/dev/null || exit; sleep 60; kill -0 "$$" 2>/dev/null || exit; done ) &
KEEPALIVE_PID=$!
validate_csv() {
    local path=$1 sizes_s=$2 reps=$3
    python3 - "$path" "$sizes_s" "$reps" <<'PYVALID'
import csv,math,sys
from collections import Counter,defaultdict
path,sizes_s,reps_s=sys.argv[1:];sizes={int(x) for x in sizes_s.split(',') if x};reps=int(reps_s)
header=['schema_version','timestamp','session_id','sequence_index','run_id_global','repetition','workload','implementation','execution_mode','device_name','num_threads','problem_size','problem_spec','batches','e2e_time_s','kernel_time_s','wall_time_s','total_energy_j','device_energy_j','dram_energy_j','energy_per_op_j','energy_per_second_j','energy_per_flop_j','time_per_op_ms_kernel','time_per_op_ms_e2e','flops_total','gflops_per_s','logical_bytes_per_op','avg_power_w','runtime_status','pcie_gen','pcie_width','sm_clock_mhz','clock_before_mhz','clock_after_mhz','mem_clock_mhz','temp_c','temp_before_c','temp_after_c','throttle_reasons','cpu_cycles','cpu_instructions','cpu_ipc','cpu_cache_misses','checksum_ok']
with open(path,newline='',encoding='utf-8') as f:
 rd=csv.DictReader(f);rows=list(rd)
 if rd.fieldnames!=header:raise SystemExit(f'header mismatch: {rd.fieldnames}')
if len(rows)!=len(sizes)*reps:raise SystemExit(f'row-count mismatch: got {len(rows)}')
def num(r,k):
 try:v=float(r[k])
 except Exception:raise SystemExit(f'invalid {k}: {r.get(k)!r}')
 if not math.isfinite(v):raise SystemExit(f'non-finite {k}: {v}')
 return v
def close(a,b,rel=2e-3,abs_=1e-9):return math.isclose(a,b,rel_tol=rel,abs_tol=abs_)
def truth(v):return str(v).strip().lower() in {'1','true','yes','t'}
counts=Counter();seen=defaultdict(set)
for i,r in enumerate(rows,1):
 if r['schema_version']!='cpu-gpu-v2' or r['workload']!='REDUCTION' or r['implementation']!='cub_device_reduce_sum_fp32' or r['execution_mode']!='gpu_resident':raise SystemExit(f'row {i}: identity failed')
 n=int(r['problem_size']);rep=int(r['repetition']);b=int(r['batches'])
 if n not in sizes or not(1<=rep<=reps) or b<=0 or int(r['num_threads'])!=-1:raise SystemExit(f'row {i}: unexpected configuration')
 if r['problem_spec']!=f'elements={n}' or not truth(r['checksum_ok']):raise SystemExit(f'row {i}: spec/checksum failed')
 e2e=num(r,'e2e_time_s');ker=num(r,'kernel_time_s');wall=num(r,'wall_time_s');total=num(r,'total_energy_j');device=num(r,'device_energy_j');dram=num(r,'dram_energy_j')
 if min(e2e,ker,wall,total,device)<=0 or not close(wall,e2e,rel=2e-5):raise SystemExit(f'row {i}: invalid time/energy')
 if dram!=-1 or not close(total,device):raise SystemExit(f'row {i}: GPU energy semantics failed')
 if ker-e2e>max(.0005,.005*e2e):raise SystemExit(f'row {i}: material kernel/e2e mismatch')
 flops=(n-1)*b;bytes_=4*n+4
 checks=[(num(r,'flops_total'),flops),(num(r,'logical_bytes_per_op'),bytes_),(num(r,'energy_per_op_j'),total/b),(num(r,'energy_per_second_j'),total/e2e),(num(r,'energy_per_flop_j'),total/flops),(num(r,'time_per_op_ms_kernel'),1000*ker/b),(num(r,'time_per_op_ms_e2e'),1000*e2e/b),(num(r,'gflops_per_s'),flops/ker/1e9),(num(r,'avg_power_w'),total/e2e)]
 if any(not close(a,e,abs_=1e-6) for a,e in checks):raise SystemExit(f'row {i}: derived metric failed')
 status='below' if e2e<.75 else('in_range' if e2e<=1.25 else'above')
 if r['runtime_status']!=status:raise SystemExit(f'row {i}: runtime_status failed')
 if int(r['pcie_gen'])<=0 or int(r['pcie_width'])<=0 or int(r['sm_clock_mhz'])<=0:raise SystemExit(f'row {i}: missing GPU telemetry')
 counts[n]+=1;seen[n].add(rep)
for n in sizes:
 if counts[n]!=reps or seen[n]!=set(range(1,reps+1)):raise SystemExit(f'coverage failed for N={n}')
print(f'validated {len(rows)} REDUCTION GPU rows')
PYVALID
}
run_reduction() {
    local output=$1 log=$2 reps=$3 session_id=$4 seed=$5 size_filter=$6
    if [[ -n "$size_filter" ]]; then
        CUDA_VISIBLE_DEVICES="$GPU_INDEX" NVIDIA_TF32_OVERRIDE=0 BENCH_EXPECTED_GPU="$EXPECTED_GPU" BENCH_CUDA_DEVICE=0 BENCH_SIZE_FILTER="$size_filter" stdbuf -oL -eL "$BIN" "$output" "$reps" "$session_id" "$seed" 2>&1 | tee "$log"
    else
        env -u BENCH_SIZE_FILTER CUDA_VISIBLE_DEVICES="$GPU_INDEX" NVIDIA_TF32_OVERRIDE=0 BENCH_EXPECTED_GPU="$EXPECTED_GPU" BENCH_CUDA_DEVICE=0 stdbuf -oL -eL "$BIN" "$output" "$reps" "$session_id" "$seed" 2>&1 | tee "$log"
    fi
}
stamp=$(date +%Y%m%d_%H%M%S)
quick_id="reduction_${GPU_SLUG}_${stamp}_quickcheck";quick_csv="$OUTDIR/${quick_id}.csv";quick_log="$OUTDIR/${quick_id}.log"
echo "[quickcheck] sizes=${QUICK_SIZES}; reps=${QUICK_REPS}"
run_reduction "$quick_csv" "$quick_log" "$QUICK_REPS" "$quick_id" "$((SEED_BASE+900))" "$QUICK_SIZES"
validate_csv "$quick_csv" "$QUICK_SIZES" "$QUICK_REPS"
echo "[quickcheck] PASS: $quick_csv"
if (( QUICKCHECK_ONLY == 1 )); then
    echo "[quickcheck] QUICKCHECK_ONLY=1; official sessions were not started."
    POWER_OFF_AT_END=0
    SUCCESS=1
    exit 0
fi
for ((session=1;session<=SESSIONS;session++)); do
 seed=$((SEED_BASE+session-1));session_id="reduction_${GPU_SLUG}_${stamp}_session${session}";output="$OUTDIR/${session_id}.csv";log="$OUTDIR/${session_id}.log"
 echo "[run] ${GPU_LABEL} REDUCTION session ${session}/${SESSIONS}; seed=${seed}; reps=${REPS}"
 run_reduction "$output" "$log" "$REPS" "$session_id" "$seed" ""
 validate_csv "$output" "$SIZES" "$REPS"
 echo "[run] Session ${session}/${SESSIONS} PASS: $output"
done
SUCCESS=1
echo "[done] ${GPU_LABEL} REDUCTION: quickcheck plus ${SESSIONS} validated sessions completed."

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from gpu_gemm_common import *


def check_source_provenance(checks: list[dict], out: Path) -> None:
    root3090 = platform_root(__file__)
    source = locate_source(root3090)
    runner = locate_runner(root3090)
    if source is None:
        add_check(checks, 'provenance', 'cuda_source_present', 'WARN', False,
                  'not found', 'scripts/GEMM[/GPU]/main_gemm.cu')
    else:
        text = source.read_text(encoding='utf-8', errors='replace')
        tokens = {
            'pedantic_compute':'CUBLAS_COMPUTE_32F_PEDANTIC',
            'pedantic_math_mode':'CUBLAS_PEDANTIC_MATH',
            'direct_nvml_energy':'nvmlDeviceGetTotalEnergyConsumption',
            'resident_mode':'gpu_resident',
            'cublas_gemm_ex':'cublasGemmEx',
            'expected_sizes':'64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384',
        }
        for name, token in tokens.items():
            add_check(checks, 'provenance', name, 'FAIL', token in text,
                      'present' if token in text else 'missing', token)
        (out/'audited_cuda_source_path.txt').write_text(str(source)+'\n', encoding='utf-8')
    if runner is None:
        add_check(checks, 'provenance', 'runner_present', 'WARN', False,
                  'not found', '02_run_GPU_3090_GEMM_only.sh')
    else:
        text = runner.read_text(encoding='utf-8', errors='replace')
        add_check(checks, 'provenance', 'tf32_override_disabled', 'FAIL',
                  'NVIDIA_TF32_OVERRIDE=0' in text, 'checked', 'NVIDIA_TF32_OVERRIDE=0')
        add_check(checks, 'provenance', 'default_sessions', 'FAIL',
                  bool(re.search(r'SESSIONS=\$\{SESSIONS:-5\}', text)), 'checked', 'default 5')
        add_check(checks, 'provenance', 'default_repetitions', 'FAIL',
                  bool(re.search(r'REPS=\$\{REPS:-10\}', text)), 'checked', 'default 10')
        found = [token for token in ['SESSION_PAUSE','sleep 300','Cooling for'] if token in text]
        add_check(checks, 'provenance', 'no_session_pause', 'FAIL', not found,
                  found or 'none', 'none')
        (out/'audited_runner_path.txt').write_text(str(runner)+'\n', encoding='utf-8')


def main() -> None:
    parser = argparse.ArgumentParser(description='Validate RTX 3090 GEMM campaign')
    parser.add_argument('--campaign')
    args = parser.parse_args()
    out = results_dir(__file__)
    campaign = load_campaign(__file__, args.campaign)
    raw = campaign.dataframe
    df = add_derived(raw)
    checks: list[dict] = []

    add_check(checks, 'campaign', 'session_files', 'FAIL',
              campaign.sessions == list(range(1,6)), campaign.sessions, list(range(1,6)))
    missing = sorted(set(GPU_COLUMNS)-set(raw.columns))
    extras = sorted(set(raw.columns)-set(GPU_COLUMNS)-{'source_file','session_number','checksum_bool'})
    add_check(checks, 'schema', 'required_columns', 'FAIL', not missing, f'missing={missing}', 'all 45')
    add_check(checks, 'schema', 'unexpected_columns', 'WARN', not extras, extras or 'none', 'none')
    observed_order = [c for c in raw.columns if c not in {'source_file','session_number','checksum_bool'}]
    add_check(checks, 'schema', 'gpu_source_column_order', 'WARN', observed_order == GPU_COLUMNS,
              'exact' if observed_order == GPU_COLUMNS else 'different', 'current GPU source order')
    add_check(checks, 'schema', 'canonical_cross_platform_order', 'WARN', observed_order == CANONICAL_V2_COLUMNS,
              'canonical' if observed_order == CANONICAL_V2_COLUMNS else 'device_energy_j precedes total_energy_j',
              'same order as CPU v2')

    expected_rows = len(SIZES)*EXPECTED_REPETITIONS
    counts = raw.groupby('source_file').size().to_dict()
    add_check(checks, 'coverage', 'rows_per_session', 'FAIL',
              len(counts)==5 and all(v==expected_rows for v in counts.values()), counts, f'{expected_rows} each')
    sizes = sorted(raw.problem_size.dropna().astype(int).unique().tolist())
    add_check(checks, 'coverage', 'problem_sizes', 'FAIL', sizes == SIZES, sizes, SIZES)
    duplicates = int(raw.duplicated(['session_number','problem_size','repetition']).sum())
    add_check(checks, 'coverage', 'duplicate_repetitions', 'FAIL', duplicates == 0, duplicates, 0)
    grouped = raw.groupby(['session_number','problem_size'])['repetition']
    rep_counts = grouped.nunique()
    add_check(checks, 'coverage', 'repetitions_per_size', 'FAIL',
              len(rep_counts)==45 and bool((rep_counts==10).all()),
              f'groups={len(rep_counts)}, min={rep_counts.min()}, max={rep_counts.max()}', '45 groups; 10 each')
    bad_rep = sum(tuple(sorted(g.astype(int))) != tuple(range(1,11)) for _, g in grouped)
    add_check(checks, 'coverage', 'repetition_ids', 'FAIL', bad_rep == 0, bad_rep, 0)

    expected_sets = {
        'schema_version':{'cpu-gpu-v2'}, 'workload':{'GEMM'},
        'implementation':{'cublas_gemm_ex_fp32_pedantic'}, 'execution_mode':{'gpu_resident'},
    }
    for column, expected in expected_sets.items():
        observed = set(raw[column].astype(str))
        add_check(checks, 'semantics', column, 'FAIL', observed == expected, sorted(observed), sorted(expected))
    names = sorted(raw.device_name.dropna().astype(str).unique())
    add_check(checks, 'semantics', 'device_name', 'FAIL',
              len(names)==1 and 'RTX 3090' in names[0], names, 'one RTX 3090')
    add_check(checks, 'semantics', 'num_threads_sentinel', 'FAIL',
              bool((raw.num_threads==-1).all()), sorted(raw.num_threads.unique()), -1)
    expected_spec = raw.problem_size.map(lambda n: f'N={int(n)}')
    add_check(checks, 'semantics', 'problem_spec', 'FAIL',
              bool((raw.problem_spec.astype(str)==expected_spec).all()), 'checked', 'N=<size>')
    add_check(checks, 'correctness', 'checksum', 'FAIL', bool(raw.checksum_bool.all()),
              f'failed={int((~raw.checksum_bool).sum())}', 0)

    positive = ['batches','e2e_time_s','kernel_time_s','wall_time_s','device_energy_j','flops_total','gflops_per_s','avg_power_w']
    bad_positive = int((~np.isfinite(raw[positive]) | (raw[positive] <= 0)).any(axis=1).sum())
    add_check(checks, 'correctness', 'finite_positive_measurements', 'FAIL', bad_positive==0, bad_positive, 0)

    expected_flops = 2.0*raw.problem_size.astype(float)**3*raw.batches.astype(float)
    expected_bytes = 12.0*raw.problem_size.astype(float)**2
    expected_status = np.where(raw.e2e_time_s<0.75, 'below', np.where(raw.e2e_time_s>1.25, 'above', 'in_range'))
    e2e_per_op = 1000.0*raw.e2e_time_s/raw.batches
    kernel_per_op = 1000.0*raw.kernel_time_s/raw.batches
    time_atol = 0.5e-9 + 1000.0*0.5e-6/raw.batches
    formulas = {
        'e2e_equals_wall':rel_close(raw.e2e_time_s, raw.wall_time_s, 1e-9, 1e-9),
        'kernel_not_above_e2e':raw.kernel_time_s <= raw.e2e_time_s + 2e-4,
        'device_equals_total_energy':rel_close(raw.device_energy_j, raw.total_energy_j, 2e-6, 2e-6),
        'dram_gpu_sentinel':raw.dram_energy_j == -1,
        'energy_per_op':rel_close(raw.energy_per_op_j, raw.device_energy_j/raw.batches, 2e-6, 1e-9),
        'energy_per_second':rel_close(raw.energy_per_second_j, raw.device_energy_j/raw.e2e_time_s, 3e-6, 1e-7),
        'energy_per_flop':rel_close(raw.energy_per_flop_j, raw.device_energy_j/expected_flops, 3e-6, 1e-20),
        'time_per_op_e2e':(raw.time_per_op_ms_e2e-e2e_per_op).abs() <= time_atol,
        'time_per_op_kernel':(raw.time_per_op_ms_kernel-kernel_per_op).abs() <= time_atol,
        'flops_total':rel_close(raw.flops_total, expected_flops, 2e-9, 1e-2),
        'gflops_per_s':rel_close(raw.gflops_per_s, raw.flops_total/raw.kernel_time_s/1e9, 3e-6, 1e-4),
        'logical_bytes':rel_close(raw.logical_bytes_per_op, expected_bytes, 2e-9, 1e-3),
        'avg_power':rel_close(raw.avg_power_w, raw.device_energy_j/raw.e2e_time_s, 3e-6, 1e-4),
        'runtime_status':raw.runtime_status.astype(str).to_numpy() == expected_status,
    }
    failure_parts = []
    detail = ['source_file','session_number','sequence_index','repetition','problem_size','batches','e2e_time_s','kernel_time_s','device_energy_j']
    for name, mask in formulas.items():
        mask = pd.Series(mask, index=raw.index).fillna(False)
        failed = int((~mask).sum())
        add_check(checks, 'formula', name, 'FAIL', failed==0, f'failed_rows={failed}', 0)
        if failed:
            part = raw.loc[~mask, detail].copy()
            part.insert(0, 'failed_formula', name)
            failure_parts.append(part)
    failures = pd.concat(failure_parts, ignore_index=True) if failure_parts else pd.DataFrame(columns=['failed_formula',*detail])
    failures.to_csv(out/'formula_failures.csv', index=False)

    sentinels = all(bool((raw[c]==-1).all()) for c in ['cpu_cycles','cpu_instructions','cpu_ipc','cpu_cache_misses'])
    add_check(checks, 'semantics', 'cpu_counter_sentinels', 'FAIL', sentinels, 'checked', -1)

    all_share = float((raw.runtime_status=='in_range').mean())
    actionable = raw[~((raw.batches==1) & (raw.e2e_time_s>TARGET_HIGH_S))]
    action_share = float((actionable.runtime_status=='in_range').mean()) if len(actionable) else 1.0
    add_check(checks, 'plausibility', 'target_runtime_share_all', 'WARN', all_share>=0.90, f'{100*all_share:.2f}%', '>=90%')
    add_check(checks, 'plausibility', 'target_runtime_share_actionable', 'WARN', action_share>=0.90,
              f'{100*action_share:.2f}%', '>=90% excluding unavoidable batch=1 rows')
    max_temp = float(raw.temp_c.max())
    add_check(checks, 'plausibility', 'maximum_temperature', 'WARN', max_temp<90, f'{max_temp:.1f} C', '<90 C')
    pmin, pmax = float(raw.avg_power_w.min()), float(raw.avg_power_w.max())
    add_check(checks, 'plausibility', 'board_power_range', 'WARN', pmin>1 and pmax<450,
              f'{pmin:.1f}..{pmax:.1f} W', '1..450 W')
    kmin = float(df.kernel_fraction.min())
    add_check(checks, 'plausibility', 'resident_kernel_fraction', 'WARN', kmin>=0.98, f'min={kmin:.5f}', '>=0.98')
    serious = int(df.serious_throttle.sum())
    add_check(checks, 'plausibility', 'serious_throttle_reasons', 'WARN', serious==0, f'rows={serious}', 0)

    seq_bad = sid_bad = 0
    for session, group in raw.groupby('session_number'):
        if sorted(group.sequence_index.astype(int)) != list(range(1,expected_rows+1)):
            seq_bad += 1
        if set(group.session_id.astype(str)) != {f'{campaign.stamp}_session{session}'}:
            sid_bad += 1
    add_check(checks, 'provenance', 'sequence_indices', 'FAIL', seq_bad==0, seq_bad, 0)
    add_check(checks, 'provenance', 'session_ids_match_files', 'FAIL', sid_bad==0, sid_bad, 0)
    check_source_provenance(checks, out)

    throttle = df.groupby('throttle_mask').size().reset_index(name='rows').sort_values('rows', ascending=False)
    throttle['hex_mask'] = throttle.throttle_mask.map(lambda x: f'0x{int(x):X}')
    throttle['decoded'] = throttle.throttle_mask.map(lambda x: throttle_labels(int(x)))
    throttle.to_csv(out/'throttle_reason_summary.csv', index=False)

    pd.DataFrame([{
        'campaign':campaign.stamp, 'files':len(campaign.files), 'rows':len(raw),
        'run_directory':str(find_run_dir(platform_root(__file__))), 'device_name':'; '.join(names),
    }]).to_csv(out/'campaign_manifest.csv', index=False)
    check_df = pd.DataFrame(checks)
    check_df.to_csv(out/'validation_checks.csv', index=False)
    hard = check_df[(check_df.severity=='FAIL') & (check_df.status=='FAIL')]
    warns = check_df[(check_df.severity=='WARN') & (check_df.status=='WARN')]
    verdict = 'FAIL' if len(hard) else ('PASS WITH WARNINGS' if len(warns) else 'PASS')
    report = (
        '# RTX 3090 GEMM validation report\n\n'
        f'- Campaign: `{campaign.stamp}`\n- Files: {len(campaign.files)}\n- Rows: {len(raw)}\n'
        f'- Expected rows/session: {expected_rows}\n- Overall verdict: **{verdict}**\n\n'
        f'## Failed checks\n\n{markdown_table(hard)}\n\n'
        f'## Warnings\n\n{markdown_table(warns)}\n\n'
        f'## All checks\n\n{markdown_table(check_df, 200)}\n\n'
        '## Metric semantics\n\n'
        'This campaign measures `gpu_resident` execution. Allocation, initialization and PCIe transfers are outside the measured interval. '
        'Energy is the direct NVML board-energy delta and is therefore not the same domain as CPU package-only RAPL.\n'
    )
    (out/'validation_report.md').write_text(report, encoding='utf-8')
    print(f'[RTX 3090] validation: {verdict}')
    print(out/'validation_report.md')
    if verdict == 'FAIL':
        sys.exit(2)

if __name__ == '__main__':
    main()

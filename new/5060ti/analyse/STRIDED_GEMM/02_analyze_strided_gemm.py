#!/usr/bin/env python3
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gpu_strided_common import *

METRICS = ['runtime_per_op_s','device_energy_per_op_j','throughput_gflops_exact','efficiency_gflop_per_j_exact','avg_power_w','temp_c','sm_clock_mhz','edp_j_s']


def build_summary(session: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for n, group in session.groupby('problem_size'):
        row = {'problem_size':int(n), 'sessions':int(group.session_number.nunique())}
        for metric in METRICS:
            values = group[metric].to_numpy(float)
            lo, hi = exact_bootstrap_ci(values)
            row[f'{metric}_median'] = float(np.median(values))
            row[f'{metric}_ci95_low'] = lo
            row[f'{metric}_ci95_high'] = hi
            row[f'{metric}_session_cv_pct'] = cv_pct(values)
        raw_n = raw[raw.problem_size==n]
        row['run_runtime_cv_pct'] = cv_pct(raw_n.runtime_per_op_s)
        row['run_energy_cv_pct'] = cv_pct(raw_n.device_energy_per_op_j)
        row['in_range_share'] = float((raw_n.runtime_status=='in_range').mean())
        row['batch_median'] = float(raw_n.batches.median())
        row['serious_throttle_rows'] = int(raw_n.serious_throttle.sum())
        row['stable_runtime'] = row['runtime_per_op_s_session_cv_pct'] <= 5.0
        row['stable_energy'] = row['device_energy_per_op_j_session_cv_pct'] <= 5.0
        row['stable_throughput'] = row['throughput_gflops_exact_session_cv_pct'] <= 5.0
        rows.append(row)
    return pd.DataFrame(rows).sort_values('problem_size')


def plot_metric(summary, metric, ylabel, filename, figdir, title, logy=True):
    x = summary.problem_size.to_numpy()
    med = summary[f'{metric}_median'].to_numpy()
    lo = summary[f'{metric}_ci95_low'].to_numpy()
    hi = summary[f'{metric}_ci95_high'].to_numpy()
    plt.figure(figsize=(8,5))
    plt.errorbar(x, med, yerr=[med-lo,hi-med], marker='o', capsize=3)
    plt.xscale('log', base=2)
    if logy:
        plt.yscale('log')
    plt.xlabel('Matrix size N')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.savefig(figdir/filename, dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description='Analyze one GPU STRIDED_GEMM campaign')
    parser.add_argument('--campaign')
    args = parser.parse_args()
    _, platform, _, _, out, cfg = context(__file__)
    checks_path = out/'validation_checks.csv'
    if not checks_path.is_file():
        raise SystemExit('Run 01_validate_strided_gemm.py first.')
    checks = pd.read_csv(checks_path)
    if ((checks.severity=='FAIL') & (checks.status=='FAIL')).any():
        raise SystemExit('Validation has hard failures; analysis aborted.')

    campaign = load_campaign(__file__, args.campaign)
    df = add_derived(campaign.dataframe)
    outlier_parts = []
    for _, group in df.groupby(['session_number','problem_size']):
        mask = robust_outlier_mask(group, ['runtime_per_op_s','device_energy_per_op_j','throughput_gflops_exact','avg_power_w'])
        if mask.any():
            part = group.loc[mask].copy()
            part['outlier_scope'] = 'within_session_and_size'
            outlier_parts.append(part)
    outliers = pd.concat(outlier_parts, ignore_index=True) if outlier_parts else pd.DataFrame(columns=list(df.columns)+['outlier_scope'])
    outliers.to_csv(out/'robust_outliers.csv', index=False)

    session = df.groupby(['session_number','problem_size'], as_index=False)[
        METRICS+['kernel_fraction','clock_change_pct','temperature_rise_c','batches']
    ].median()
    # Compatibility aliases used by the all-platform pipeline.
    session['gflops_per_s'] = session['throughput_gflops_exact']
    session.to_csv(out/'session_medians_by_size.csv', index=False)
    summary = build_summary(session, df)
    summary.to_csv(out/'size_summary.csv', index=False)

    overview = df.groupby('session_number', as_index=False).agg(
        rows=('problem_size','size'), median_gflops=('throughput_gflops_exact','median'),
        median_power_w=('avg_power_w','median'), median_temp_c=('temp_c','median'),
        max_temp_c=('temp_c','max'), median_sm_clock_mhz=('sm_clock_mhz','median'),
        serious_throttle_rows=('serious_throttle','sum'),
        in_range_share=('runtime_status', lambda s: float((s=='in_range').mean())),
    )
    overview.to_csv(out/'session_overview.csv', index=False)
    throttle = df.groupby('throttle_mask', as_index=False).agg(rows=('problem_size','size'))
    throttle['hex_mask'] = throttle.throttle_mask.map(lambda x: f'0x{int(x):X}')
    throttle['decoded'] = throttle.throttle_mask.map(lambda x: throttle_labels(int(x)))
    throttle.to_csv(out/'throttle_analysis.csv', index=False)

    figdir = out/'figures'
    plot_metric(summary, 'throughput_gflops_exact', 'Throughput (GFLOP/s)', 'throughput_by_size.png', figdir, f"{cfg['label']} STRIDED_GEMM: throughput", False)
    plot_metric(summary, 'device_energy_per_op_j', 'Board energy per GEMM (J)', 'energy_per_gemm.png', figdir, f"{cfg['label']} STRIDED_GEMM: energy")
    plot_metric(summary, 'runtime_per_op_s', 'Runtime per GEMM (s)', 'runtime_per_gemm.png', figdir, f"{cfg['label']} STRIDED_GEMM: runtime")
    plot_metric(summary, 'edp_j_s', 'Energy-delay product (J·s)', 'edp_by_size.png', figdir, f"{cfg['label']} STRIDED_GEMM: EDP")

    x = summary.problem_size.to_numpy()
    plt.figure(figsize=(8,5))
    plt.plot(x, summary.avg_power_w_median, marker='o', label='Board power (W)')
    plt.plot(x, summary.temp_c_median, marker='s', label='Temperature (°C)')
    plt.xscale('log', base=2)
    plt.xlabel('Matrix size N')
    plt.ylabel('Median value')
    plt.title(f"{cfg['label']} STRIDED_GEMM: power and temperature")
    plt.grid(True, which='both', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figdir/'power_temperature_by_size.png', dpi=180)
    plt.close()

    pivot = session.pivot(index='problem_size', columns='session_number', values='throughput_gflops_exact')
    plt.figure(figsize=(8,5))
    for column in pivot.columns:
        plt.plot(pivot.index, pivot[column], marker='o', alpha=0.75, label=f'Session {column}')
    plt.xscale('log', base=2)
    plt.xlabel('Matrix size N')
    plt.ylabel('GFLOP/s')
    plt.title(f"{cfg['label']} STRIDED_GEMM: session repeatability")
    plt.grid(True, which='both', alpha=0.3)
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(figdir/'session_throughput_repeatability.png', dpi=180)
    plt.close()

    peak = summary.loc[summary.throughput_gflops_exact_median.idxmax()]
    best_eff = summary.loc[summary.efficiency_gflop_per_j_exact_median.idxmax()]
    unstable_runtime = summary.loc[~summary.stable_runtime,'problem_size'].astype(int).tolist()
    unstable_energy = summary.loc[~summary.stable_energy,'problem_size'].astype(int).tolist()
    unstable_throughput = summary.loc[~summary.stable_throughput,'problem_size'].astype(int).tolist()
    unavoidable = (df.batches==1) & (df.e2e_time_s>TARGET_HIGH_S)
    actionable = df.loc[~unavoidable]
    report = (
        f"# {cfg['label']} STRIDED_GEMM scientific analysis\n\n"
        f'- Campaign: `{campaign.stamp}`\n- Measurements: {len(df)}\n- Sessions: {df.session_number.nunique()}\n'
        '- Mode: `gpu_resident`\n- Implementation: `cublas_gemm_ex_fp32_pedantic_ld2n`\n'
        '- Layout: logical N×N matrices, physical leading dimension `ld=2N`\n\n'
        '## Quality and repeatability\n\n'
        f'- Overall target-window share: {100*(df.runtime_status=="in_range").mean():.2f}%\n'
        f'- Actionable target-window share: {100*(actionable.runtime_status=="in_range").mean():.2f}%\n'
        f'- Robust outlier share: {100*len(outliers)/len(df):.2f}%\n'
        f'- Maximum temperature: {df.temp_c.max():.1f} °C\n'
        f'- Median before/after SM-clock decline: {np.nanmedian(-df.clock_change_pct):.2f}%\n'
        f'- Serious throttle rows: {int(df.serious_throttle.sum())}\n'
        f'- Runtime-unstable sizes: {unstable_runtime or "none"}\n'
        f'- Energy-unstable sizes: {unstable_energy or "none"}\n'
        f'- Throughput-unstable sizes: {unstable_throughput or "none"}\n\n'
        '## Main findings\n\n'
        f'- Peak median throughput: {peak.throughput_gflops_exact_median:.2f} GFLOP/s at N={int(peak.problem_size)}.\n'
        f'- Peak board efficiency: {best_eff.efficiency_gflop_per_j_exact_median:.2f} GFLOP/J at N={int(best_eff.problem_size)}.\n\n'
        '## Interpretation contract\n\n'
        'Runtime and throughput are inverse views of the same fixed-work axis. Board energy and GFLOP/J are inverse views of the same fixed-work axis. '
        'They are not counted as independent votes. EDP is a composite of runtime and energy. '
        'The 12N² logical-byte value is a semantic anchor, while the allocated footprint is 24N²; neither is measured physical memory traffic. '
        'PCIe transfers are outside the measured interval. NVML energy is board-level.\n'
    )
    (out/'scientific_summary.md').write_text(report, encoding='utf-8')
    print(f"[{cfg['label']}] STRIDED_GEMM scientific analysis written to {out}")

if __name__ == '__main__':
    main()

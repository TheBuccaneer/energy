#!/usr/bin/env python3
from __future__ import annotations
import math
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cpu_strided_common import PRACTICAL_TOLERANCE, independent_ratio_ci, probability_superiority, cliffs_delta, markdown_table, write_text

def classify_ratio(point, lo, hi):
    low_eq, high_eq = 1/(1+PRACTICAL_TOLERANCE), 1+PRACTICAL_TOLERANCE
    if hi < low_eq: return 'clear_intel'
    if lo > high_eq: return 'clear_amd'
    if lo >= low_eq and hi <= high_eq: return 'practically_equivalent'
    return 'uncertain_intel_advantage' if point < 1 else ('uncertain_amd_advantage' if point > 1 else 'uncertain')

def select_best(summary, metric, objective='min'):
    idx = summary.groupby('problem_size')[metric].idxmin() if objective=='min' else summary.groupby('problem_size')[metric].idxmax()
    return summary.loc[idx].sort_values('problem_size').reset_index(drop=True)

def main():
    root = Path(__file__).resolve().parents[3]
    out = root/'ALL AUDIT'/'STRIDED_GEMM'/'results'; out.mkdir(parents=True, exist_ok=True); (out/'figures').mkdir(exist_ok=True)
    data={}
    for p in ['INTEL','AMD']:
        r=root/p/'results'/'STRIDED_GEMM'
        data[p]={'summary':pd.read_csv(r/'configuration_summary.csv'),'sessions':pd.read_csv(r/'session_configuration_medians.csv'),'trade':pd.read_csv(r/'within_platform_energy_runtime_tradeoffs.csv')}
    common_threads=sorted(set(data['INTEL']['summary'].num_threads)&set(data['AMD']['summary'].num_threads))
    rows=[]
    metrics=[('runtime_per_op_s','min'),('total_energy_per_op_j','min'),('package_energy_per_op_j','min'),('edp_total_j_s','min')]
    for n in sorted(data['INTEL']['summary'].problem_size.unique()):
      for t in common_threads:
       for metric,obj in metrics:
        vals={}
        for p in ['INTEL','AMD']:
          s=data[p]['sessions']; vals[p]=s[(s.problem_size==n)&(s.num_threads==t)][metric].to_numpy(float)
        point,lo,hi=independent_ratio_ci(vals['INTEL'],vals['AMD'],seed=int(n+t+len(metric)))
        rows.append({'problem_size':int(n),'num_threads':int(t),'metric':metric,'intel_over_amd_ratio':point,'ci95_low':lo,'ci95_high':hi,'classification':classify_ratio(point,lo,hi),'probability_intel_better':probability_superiority(vals['INTEL'],vals['AMD'],True),'cliffs_delta_intel_minus_amd':cliffs_delta(vals['INTEL'],vals['AMD'])})
    common=pd.DataFrame(rows); common.to_csv(out/'cross_common_thread_comparison.csv',index=False)

    native_rows=[]
    for metric,obj in metrics:
      best={p:select_best(data[p]['summary'],metric+'_median',obj) for p in ['INTEL','AMD']}
      for n in sorted(best['INTEL'].problem_size.unique()):
        vals={}; threads={}
        for p in ['INTEL','AMD']:
          row=best[p][best[p].problem_size==n].iloc[0]; threads[p]=int(row.num_threads)
          s=data[p]['sessions']; vals[p]=s[(s.problem_size==n)&(s.num_threads==threads[p])][metric].to_numpy(float)
        point,lo,hi=independent_ratio_ci(vals['INTEL'],vals['AMD'],seed=int(n+len(metric)*13))
        native_rows.append({'problem_size':int(n),'metric':metric,'intel_best_threads':threads['INTEL'],'amd_best_threads':threads['AMD'],'intel_over_amd_ratio':point,'ci95_low':lo,'ci95_high':hi,'classification':classify_ratio(point,lo,hi),'probability_intel_better':probability_superiority(vals['INTEL'],vals['AMD'],True),'cliffs_delta_intel_minus_amd':cliffs_delta(vals['INTEL'],vals['AMD']),'analysis_type':'descriptive_native_best_post_selection'})
    native=pd.DataFrame(native_rows); native.to_csv(out/'cross_native_best_comparison.csv',index=False)

    placement=native[native.metric.isin(['runtime_per_op_s','total_energy_per_op_j','edp_total_j_s'])].pivot(index='problem_size',columns='metric',values='classification').reset_index()
    placement.columns=['problem_size']+[f'{c}_classification' for c in placement.columns[1:]]
    placement.to_csv(out/'cpu_strided_placement_by_size.csv',index=False)
    trade=pd.concat([data[p]['trade'].assign(platform=p) for p in ['INTEL','AMD']],ignore_index=True); trade.to_csv(out/'within_platform_tradeoffs_combined.csv',index=False)

    sens=[]
    for p in ['INTEL','AMD']:
      s=data[p]['summary']
      total=select_best(s,'total_energy_per_op_j_median'); package=select_best(s,'package_energy_per_op_j_median')
      m=total[['problem_size','num_threads']].rename(columns={'num_threads':'total_best_threads'}).merge(package[['problem_size','num_threads']].rename(columns={'num_threads':'package_best_threads'}),on='problem_size')
      m['platform']=p; m['winner_changes']=m.total_best_threads!=m.package_best_threads; sens.append(m)
    pd.concat(sens,ignore_index=True).to_csv(out/'package_vs_total_energy_winner_sensitivity.csv',index=False)

    for metric,label,file in [('runtime_per_op_s','Runtime per GEMM [s]','native_best_runtime.png'),('total_energy_per_op_j','Package + DRAM energy [J]','native_best_total_energy.png')]:
      fig,ax=plt.subplots(figsize=(8,5))
      for p in ['INTEL','AMD']:
        b=select_best(data[p]['summary'],metric+'_median')
        ax.plot(b.problem_size,b[metric+'_median'],marker='o',label=p)
      ax.set_xscale('log',base=2); ax.set_yscale('log'); ax.set_xlabel('N'); ax.set_ylabel(label); ax.grid(True,alpha=.3); ax.legend(); fig.tight_layout(); fig.savefig(out/'figures'/file,dpi=180); plt.close(fig)

    report=f"""# Intel–AMD STRIDED_GEMM comparison

Primary energy is package + DRAM; package-only is a sensitivity analysis. Native-best comparisons are descriptive post-selection estimates over five session medians.

## Native-best classifications

{markdown_table(native,80)}

## Within-platform trade-offs

{markdown_table(trade,40)}
"""
    write_text(out/'CPU_STRIDED_GEMM_COMPARISON.md',report)
    print(f'[CPU STRIDED_GEMM] comparison written to {out}')
if __name__=='__main__': main()

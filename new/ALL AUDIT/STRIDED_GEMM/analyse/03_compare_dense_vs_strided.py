#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from cpu_strided_common import add_derived,campaign_summary,load_project_campaign,session_medians,independent_ratio_ci,probability_superiority,cliffs_delta,PRACTICAL_TOLERANCE,markdown_table,write_text

def classify(point,lo,hi):
    a,b=1/(1+PRACTICAL_TOLERANCE),1+PRACTICAL_TOLERANCE
    if hi<a:return 'clear_strided_lower'
    if lo>b:return 'clear_strided_higher'
    if lo>=a and hi<=b:return 'practically_equivalent'
    return 'uncertain_strided_lower' if point<1 else 'uncertain_strided_higher'

def best(summary,metric): return summary.loc[summary.groupby('problem_size')[metric].idxmin()].sort_values('problem_size')
def main():
 root=Path(__file__).resolve().parents[3]; out=root/'ALL AUDIT'/'STRIDED_GEMM'/'results'; (out/'figures').mkdir(parents=True,exist_ok=True)
 all_native=[]; all_matched=[]; changes=[]
 for p in ['INTEL','AMD']:
  camps={w:load_project_campaign(root,p,w) for w in ['GEMM','STRIDED_GEMM']}
  sess={w:session_medians(add_derived(camps[w].data)) for w in camps}
  metric_cols=['runtime_per_op_s','total_energy_per_op_j','package_energy_per_op_j','edp_total_j_s']
  summ={w:sess[w].groupby(['problem_size','num_threads'],as_index=False)[metric_cols].median().rename(columns={m:m+'_median' for m in metric_cols}) for w in camps}
  for metric in ['runtime_per_op_s','total_energy_per_op_j','package_energy_per_op_j','edp_total_j_s']:
   bd=best(summ['GEMM'],metric+'_median'); bs=best(summ['STRIDED_GEMM'],metric+'_median')
   for n in sorted(bd.problem_size.unique()):
    td=int(bd[bd.problem_size==n].iloc[0].num_threads); ts=int(bs[bs.problem_size==n].iloc[0].num_threads)
    vd=sess['GEMM'][(sess['GEMM'].problem_size==n)&(sess['GEMM'].num_threads==td)][metric]
    vs=sess['STRIDED_GEMM'][(sess['STRIDED_GEMM'].problem_size==n)&(sess['STRIDED_GEMM'].num_threads==ts)][metric]
    pt,lo,hi=independent_ratio_ci(vs,vd,seed=int(n+len(metric)))
    all_native.append({'platform':p,'problem_size':int(n),'metric':metric,'dense_best_threads':td,'strided_best_threads':ts,'strided_over_dense_ratio':pt,'ci95_low':lo,'ci95_high':hi,'classification':classify(pt,lo,hi),'probability_strided_lower':probability_superiority(vs,vd,True),'cliffs_delta_strided_minus_dense':cliffs_delta(vs,vd),'analysis_type':'descriptive_native_best_post_selection'})
  common=sorted(set(sess['GEMM'].num_threads)&set(sess['STRIDED_GEMM'].num_threads))
  for n in sorted(sess['GEMM'].problem_size.unique()):
   for t in common:
    for metric in ['runtime_per_op_s','total_energy_per_op_j','package_energy_per_op_j','edp_total_j_s']:
     vd=sess['GEMM'][(sess['GEMM'].problem_size==n)&(sess['GEMM'].num_threads==t)][metric]; vs=sess['STRIDED_GEMM'][(sess['STRIDED_GEMM'].problem_size==n)&(sess['STRIDED_GEMM'].num_threads==t)][metric]
     pt,lo,hi=independent_ratio_ci(vs,vd,seed=int(n+t+len(metric)))
     all_matched.append({'platform':p,'problem_size':int(n),'num_threads':int(t),'metric':metric,'strided_over_dense_ratio':pt,'ci95_low':lo,'ci95_high':hi,'classification':classify(pt,lo,hi)})
 native=pd.DataFrame(all_native); matched=pd.DataFrame(all_matched); native.to_csv(out/'dense_vs_strided_native_best.csv',index=False); matched.to_csv(out/'dense_vs_strided_configuration_matched.csv',index=False)
 # CPU placement changes by workload
 for w in ['GEMM','STRIDED_GEMM']:
  summaries={}
  for p in ['INTEL','AMD']:
   ss=session_medians(add_derived(load_project_campaign(root,p,w).data))
   summaries[p]=ss.groupby(['problem_size','num_threads'],as_index=False)[['runtime_per_op_s','total_energy_per_op_j']].median().rename(columns={'runtime_per_op_s':'runtime_per_op_s_median','total_energy_per_op_j':'total_energy_per_op_j_median'})
  for metric in ['runtime_per_op_s_median','total_energy_per_op_j_median']:
   for n in sorted(summaries['INTEL'].problem_size.unique()):
    vals={p:float(best(summaries[p],metric).query('problem_size==@n').iloc[0][metric]) for p in ['INTEL','AMD']}
    changes.append({'workload':w,'problem_size':int(n),'metric':metric,'winner':'INTEL' if vals['INTEL']<vals['AMD'] else 'AMD','intel_value':vals['INTEL'],'amd_value':vals['AMD'],'intel_over_amd_ratio':vals['INTEL']/vals['AMD']})
 ch=pd.DataFrame(changes); piv=ch.pivot(index=['problem_size','metric'],columns='workload',values='winner').reset_index(); piv['placement_changes']=piv.GEMM!=piv.STRIDED_GEMM; piv.to_csv(out/'layout_induced_cpu_placement_changes.csv',index=False)
 for metric,file,label in [('runtime_per_op_s','dense_vs_strided_runtime_ratio.png','Strided / dense runtime'),('total_energy_per_op_j','dense_vs_strided_total_energy_ratio.png','Strided / dense total energy')]:
  fig,ax=plt.subplots(figsize=(8,5))
  q=native[native.metric==metric]
  for p,g in q.groupby('platform'):ax.plot(g.problem_size,g.strided_over_dense_ratio,marker='o',label=p)
  ax.axhline(1,linestyle='--');ax.set_xscale('log',base=2);ax.set_xlabel('N');ax.set_ylabel(label);ax.grid(True,alpha=.3);ax.legend();fig.tight_layout();fig.savefig(out/'figures'/file,dpi=180);plt.close(fig)
 write_text(out/'DENSE_VS_STRIDED_REPORT.md',f"""# Dense GEMM versus STRIDED_GEMM

Ratios are STRIDED_GEMM / GEMM. Values above one mean a layout penalty. Native-best comparisons are descriptive post-selection; configuration-matched comparisons hold thread count fixed.

## Native-best ratios

{markdown_table(native,100)}

## CPU placement changes

{markdown_table(piv,50)}
""")
 print(f'[CPU STRIDED_GEMM] dense-vs-strided comparison written to {out}')
if __name__=='__main__':main()

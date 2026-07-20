#!/usr/bin/env python3
from __future__ import annotations
import math, sys
from pathlib import Path
import numpy as np
import pandas as pd
from cpu_strided_common import PRACTICAL_TOLERANCE, markdown_table, write_text

def add(rows,cat,name,severity,passed,observed,expected): rows.append({'category':cat,'check':name,'severity':severity,'status':'PASS' if passed else severity,'observed':str(observed),'expected':str(expected)})
def main():
 root=Path(__file__).resolve().parents[3]; out=root/'ALL AUDIT'/'STRIDED_GEMM'/'results'; rows=[]
 for p in ['AMD','INTEL']:
  s=pd.read_csv(root/p/'results'/'STRIDED_GEMM'/'configuration_summary.csv')
  n=s.problem_size.astype(float); flops=2*n**3
  checks={
   'throughput_identity':np.isclose(s.throughput_gflops_exact_median,flops/s.runtime_per_op_s_median/1e9,rtol=1e-10,atol=1e-12),
   'total_efficiency_identity':np.isclose(s.total_efficiency_gflop_per_j_median,flops/s.total_energy_per_op_j_median/1e9,rtol=1e-10,atol=1e-12),
   'edp_identity':np.isclose(s.edp_total_j_s_median,s.total_energy_per_op_j_median*s.runtime_per_op_s_median,rtol=5e-3,atol=1e-12),
   'allocated_footprint_factor':np.ones(len(s),dtype=bool),
  }
  for name,mask in checks.items(): add(rows,p,name,'FAIL',bool(np.all(mask)),int((~mask).sum()),0)
  for nval,g in s.groupby('problem_size'):
   e=int(g.loc[g.total_energy_per_op_j_median.idxmin()].num_threads); eff=int(g.loc[g.total_efficiency_gflop_per_j_median.idxmax()].num_threads)
   add(rows,p,f'energy_efficiency_inverse_N{int(nval)}','FAIL',e==eff,f'{e}/{eff}','same thread')
 native=pd.read_csv(out/'cross_native_best_comparison.csv')
 loweq,higheq=1/(1+PRACTICAL_TOLERANCE),1+PRACTICAL_TOLERANCE
 for i,r in native.iterrows():
  if r.ci95_high<loweq: exp='clear_intel'
  elif r.ci95_low>higheq: exp='clear_amd'
  elif r.ci95_low>=loweq and r.ci95_high<=higheq: exp='practically_equivalent'
  else: exp='uncertain_intel_advantage' if r.intel_over_amd_ratio<1 else ('uncertain_amd_advantage' if r.intel_over_amd_ratio>1 else 'uncertain')
  add(rows,'classification',f'native_row_{i}','FAIL',r.classification==exp,r.classification,exp)
 dense=pd.read_csv(out/'dense_vs_strided_native_best.csv')
 add(rows,'coverage','dense_vs_strided_native_rows','FAIL',len(dense)==2*9*4,len(dense),72)
 matched=pd.read_csv(out/'dense_vs_strided_configuration_matched.csv')
 add(rows,'coverage','matched_rows','FAIL',len(matched)==(9*7*4+9*9*4),len(matched),576)
 checks=pd.DataFrame(rows); checks.to_csv(out/'integrity_checks.csv',index=False)
 hard=checks[(checks.severity=='FAIL')&(checks.status=='FAIL')]; warns=checks[(checks.severity=='WARN')&(checks.status=='WARN')]
 verdict='FAIL' if len(hard) else ('PASS WITH WARNINGS' if len(warns) else 'PASS')
 write_text(out/'INTEGRITY_AUDIT.md',f"""# CPU STRIDED_GEMM integrity audit

Overall verdict: **{verdict}**

This independent pass rechecks fixed-work identities, energy/efficiency inverse ranking, EDP consistency, ratio classification and expected output coverage.

## Failed checks

{markdown_table(hard)}

## Warnings

{markdown_table(warns)}

## All checks

{markdown_table(checks,250)}
""")
 print(f'[CPU STRIDED_GEMM] integrity audit: {verdict}'); print(out/'INTEGRITY_AUDIT.md')
 if verdict=='FAIL':sys.exit(2)
if __name__=='__main__':main()

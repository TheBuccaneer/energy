#!/usr/bin/env python3
from __future__ import annotations
import json
from all_conv2d_common import *

def main() -> None:
    project,result,figures=roots(__file__)
    summary,sessions,provenance=load_individual(project)
    envelopes=build_envelopes(summary)
    leaders=cross_platform_leaders(envelopes)
    pareto=global_pareto(summary)
    ratios=pairwise_ratios(envelopes)
    penalties=platform_penalties(envelopes)
    summary.to_csv(result/"all_platform_configuration_summary.csv",index=False)
    sessions.to_csv(result/"all_platform_session_medians.csv",index=False)
    provenance.to_csv(result/"all_platform_provenance.csv",index=False)
    envelopes.to_csv(result/"platform_envelopes_by_shape.csv",index=False)
    leaders.to_csv(result/"cross_platform_leaders_by_shape.csv",index=False)
    pareto.to_csv(result/"global_pareto_by_shape.csv",index=False)
    ratios.to_csv(result/"pairwise_ratios.csv",index=False)
    penalties.to_csv(result/"platform_penalties.csv",index=False)
    plot_envelopes(envelopes,pareto,figures)

    conflicts=[]
    for shape in sorted(envelopes.problem_size.unique()):
        sub=leaders[leaders.problem_size==shape]
        fastest=set(sub[sub.objective=="runtime"].platform)
        greenest=set(sub[sub.objective=="energy"].platform)
        conflicts.append({"problem_size":int(shape),"runtime_tie_set":"|".join(sorted(fastest)),"energy_tie_set":"|".join(sorted(greenest)),"same_tie_set":fastest==greenest,"overlap":"|".join(sorted(fastest&greenest))})
    conflicts=pd.DataFrame(conflicts)
    conflicts.to_csv(result/"runtime_energy_conflicts.csv",index=False)

    report=f"""# CONV2D all-platform comparison

## Scope

Validated five-session campaigns from AMD Threadripper 3970X, Intel i9-7900X, RTX 3090 and RTX 5060 Ti are combined only after each platform pipeline passes.

## Statistical contract

- Ten repetitions within a session are technical repetitions.
- The median within each session is the primary scientific unit.
- Five session medians form each platform/configuration summary.
- CPU thread counts are optimized independently for each shape and objective before platform-level comparison.
- Practical ties use a ±2% tolerance.

## Cross-platform leaders

{markdown_table(leaders,100)}

## Runtime-versus-energy conflicts

{markdown_table(conflicts,50)}

## Global runtime-energy Pareto status

{markdown_table(pareto,100)}

## Measurement-domain caveat

CPU energy is package RAPL plus DRAM RAPL when available; GPU energy is NVML board energy. The comparison is therefore a comparison of the study's measured device domains, not a whole-system AC-wall comparison. Runtime and energy are kept as separate primitive objectives; EDP is reported only as a composite.

## CONV2D-specific interpretation

The six shapes differ in geometry and operational intensity. cuDNN and oneDNN are allowed to select different legal algorithms or primitives per shape and platform. The comparison therefore evaluates the best observed platform implementation under the frozen mathematical CONV2D semantics, not one identical low-level algorithm.
"""
    (result/"comparison_report.md").write_text(report,encoding="utf-8")
    (result/"ANALYSIS_COMPLETE.json").write_text(json.dumps({"status":"PASS","platforms":PLATFORMS,"shapes":sorted(map(int,envelopes.problem_size.unique())),"practical_tolerance":PRACTICAL_TOLERANCE,"configuration_rows":len(summary),"session_medians":len(sessions)},indent=2),encoding="utf-8")
    print(f"[CONV2D all-platform] comparison PASS -> {result}")
if __name__=="__main__": main()

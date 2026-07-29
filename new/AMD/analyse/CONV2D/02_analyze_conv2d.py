#!/usr/bin/env python3
from __future__ import annotations
import json
from conv2d_analysis_common import *

def main() -> None:
    args=parse_args("Analyze one validated CONV2D campaign")
    ctx=context(__file__)
    campaign=load_campaign(ctx,args.campaign)
    checks,_=validate_campaign(ctx,campaign)
    hard=checks[(checks.severity=="FAIL")&(checks.status=="FAIL")]
    if len(hard): raise SystemExit("Validation failed; run 01_validate_conv2d.py and inspect validation_report.md")
    d=add_derived(campaign.data,ctx)
    session,summary,leaders,pareto= summarize_platform(d)
    d.to_csv(ctx.result_dir/"normalized_rows.csv",index=False)
    session.to_csv(ctx.result_dir/"session_medians.csv",index=False)
    summary.to_csv(ctx.result_dir/"configuration_summary.csv",index=False)
    leaders.to_csv(ctx.result_dir/"leaders_by_shape.csv",index=False)
    pareto.to_csv(ctx.result_dir/"pareto_and_stability.csv",index=False)
    telemetry=d.groupby(["problem_size","configuration"],dropna=False).agg(
        rows=("problem_size","size"), avg_power_w_median=("avg_power_w","median"),
        temp_c_median=("temp_c","median"), temp_c_max=("temp_c","max"),
        sm_clock_mhz_median=("sm_clock_mhz","median"),
        clock_change_pct_median=("clock_change_pct","median"),
        throttle_reasons_unique=("throttle_reasons",lambda x:"|".join(sorted(set(map(str,x))))),
    ).reset_index()
    telemetry.to_csv(ctx.result_dir/"telemetry_summary.csv",index=False)
    plot_platform(summary,ctx)
    report=f"""# CONV2D platform analysis — {ctx.config['label']}

## Campaign

- Selected campaign: `{campaign.stamp}`
- Raw measurements: {len(d)}
- Session medians: {len(session)}
- Configurations summarized: {len(summary)}
- Energy domain: {ctx.config['energy_domain']}

## Tie-aware leaders (2% practical tolerance)

{markdown_table(leaders,100)}

## Strict and practical Pareto status

{markdown_table(pareto,150)}

## Interpretation guardrails

- Ten repetitions are technical repetitions; the five session medians are the scientific units.
- Runtime, total measured energy and EDP are reported separately.
- cuDNN and oneDNN may select different legal algorithms or primitives per shape and platform; `problem_spec` and `implementation` retain that provenance.
- Cross-platform energy comparison must retain the meter-domain caveat: CPU package/DRAM RAPL versus GPU board NVML.
"""
    (ctx.result_dir/"analysis_report.md").write_text(report,encoding="utf-8")
    (ctx.result_dir/"ANALYSIS_COMPLETE.json").write_text(json.dumps({"status":"PASS","platform":ctx.platform,"campaign":campaign.stamp,"raw_rows":len(d),"session_medians":len(session),"practical_tolerance":PRACTICAL_TOLERANCE},indent=2),encoding="utf-8")
    print(f"[{ctx.platform} CONV2D] analysis PASS -> {ctx.result_dir}")
if __name__=="__main__": main()

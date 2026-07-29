#!/usr/bin/env python3
from __future__ import annotations
import sys
from conv2d_analysis_common import *

def main() -> None:
    args=parse_args("Validate one complete CONV2D campaign")
    ctx=context(__file__)
    try: campaign=load_campaign(ctx,args.campaign)
    except Exception as exc:
        print(f"FATAL: {exc}",file=sys.stderr); raise SystemExit(2)
    checks,failures=validate_campaign(ctx,campaign)
    checks.to_csv(ctx.result_dir/"validation_checks.csv",index=False)
    failures.to_csv(ctx.result_dir/"formula_failures.csv",index=False)
    write_manifest(ctx,campaign)
    hard=checks[(checks.severity=="FAIL")&(checks.status=="FAIL")]
    warnings=checks[(checks.severity=="WARN")&(checks.status=="WARN")]
    verdict="FAIL" if len(hard) else ("PASS WITH WARNINGS" if len(warnings) else "PASS")
    report=f"""# CONV2D validation report — {ctx.config['label']}

## Verdict

**{verdict}**

- Campaign: `{campaign.stamp}`
- Files: {len(campaign.files)}
- Raw rows: {len(campaign.data)}
- Expected sessions: 5
- Expected repetitions per configuration: 10
- Energy domain: {ctx.config['energy_domain']}
- Execution mode: `{ctx.config['mode']}`

## Hard failures

{markdown_table(hard)}

## Warnings

{markdown_table(warnings)}

## Measurement contract

- Six frozen CONV2D shapes, FP32 NCHW/OIHW, cross-correlation, no bias or activation.
- Logical FLOPs: `2*N*K*C*R*S*Hout*Wout` per convolution.
- Logical bytes: input + weights + output, each counted once.
- CPU primary measurement: RAPL package plus DRAM when available.
- GPU primary measurement: NVML board energy in `gpu_resident` mode.
- The scientific unit used downstream is the median of ten repetitions within each session.
"""
    (ctx.result_dir/"validation_report.md").write_text(report,encoding="utf-8")
    print(f"[{ctx.platform} CONV2D] validation: {verdict}")
    if len(hard): raise SystemExit(2)
if __name__=="__main__": main()

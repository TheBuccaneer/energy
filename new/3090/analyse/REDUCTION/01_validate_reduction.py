#!/usr/bin/env python3
from __future__ import annotations

import sys
import pandas as pd

from reduction_analysis_common import *


def main() -> None:
    args = parse_args("Validate one complete REDUCTION campaign")
    ctx = context(__file__)
    try:
        campaign = load_campaign(ctx, args.campaign)
    except Exception as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        raise SystemExit(2)

    checks, failures = validate_campaign(ctx, campaign)
    checks.to_csv(ctx.result_dir / "validation_checks.csv", index=False)
    failures.to_csv(ctx.result_dir / "formula_failures.csv", index=False)
    write_manifest(ctx, campaign)

    hard = checks[(checks.severity == "FAIL") & (checks.status == "FAIL")]
    warnings = checks[(checks.severity == "WARN") & (checks.status == "WARN")]
    verdict = "FAIL" if len(hard) else ("PASS WITH WARNINGS" if len(warnings) else "PASS")

    report = f"""# REDUCTION validation report — {ctx.config['label']}

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

## All checks

{markdown_table(checks, 300)}

## Measurement contract

- One operation is one complete FP32 sum of x[0:N] to one FP32 scalar.
- Logical work is `N-1` FP32 additions per operation.
- Logical data volume is `4*N+4` bytes per operation.
- Logical bytes are not measured physical memory traffic.
- CPU primary energy is package RAPL; GPU primary energy is NVML board energy.
- GPU mode is resident and excludes allocations and PCIe transfers.
"""
    (ctx.result_dir / "validation_report.md").write_text(report, encoding="utf-8")

    print(f"[{ctx.platform} REDUCTION] validation: {verdict}")
    print(ctx.result_dir / "validation_report.md")
    if len(hard):
        raise SystemExit(2)


if __name__ == "__main__":
    main()

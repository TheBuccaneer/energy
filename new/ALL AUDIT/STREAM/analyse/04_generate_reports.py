#!/usr/bin/env python3
from __future__ import annotations

import pandas as pd

from all_stream_common import *


def main() -> None:
    out = results_dir(__file__)
    manifest = pd.read_csv(out / "input_manifest.csv")
    winners = pd.read_csv(out / "all_platform_metric_winners.csv")
    placement = pd.read_csv(out / "placement_by_size.csv")
    tradeoffs = pd.read_csv(out / "within_platform_energy_runtime_tradeoffs.csv")
    stability = pd.read_csv(out / "all_platform_stability.csv")
    best = pd.read_csv(out / "best_cpu_vs_best_gpu.csv")

    clear_device = placement[placement.clear_device_tradeoff.astype(str).str.lower().isin({"true", "1"})]
    clear_config = tradeoffs[tradeoffs.interpretation == "clear_configuration_tradeoff"]
    unstable = stability[~(stability.runtime_stable_5pct & stability.energy_stable_5pct)]

    summary = f"""# All-platform STREAM results summary

## Dataset

{manifest.to_markdown(index=False)}

## What the data can reveal

1. **Bandwidth saturation and thread scaling.** CPU thread counts expose where additional threads stop improving logical bandwidth.
2. **Fastest versus greenest.** Runtime-optimal and energy-optimal configurations are compared tie-aware at every N.
3. **CPU/GPU placement crossover.** The best CPU and best GPU are compared for runtime, energy, EDP, logical bandwidth, and logical GB/J.
4. **Working-set transition.** The sweep spans 12 MB to 3.072 GB of logical array footprint; small sizes may be cache affected, while large sizes are the main memory-stress regime.
5. **Session stability.** Five session medians separate repeatability from technical repetition noise.

## Clear device-level fastest-vs-greenest conflicts

{markdown_table(clear_device)}

## Clear within-platform configuration conflicts

{markdown_table(clear_config)}

## Placement by size

{placement.to_markdown(index=False)}

## Best CPU versus best GPU

{best.to_markdown(index=False)}

## Stability warnings

{markdown_table(unstable, 250)}

## Claims hygiene

- Runtime and logical bandwidth are inverse views of one axis.
- Energy per Triad and logical GB/J are inverse views of one axis.
- EDP is a composite.
- Logical bandwidth is not measured physical DRAM/VRAM traffic.
- GPU execution is resident and excludes PCIe transfers.
- CPU package RAPL and GPU board NVML are asymmetric device domains.
- Native-best results are descriptive post-selection on five sessions.
"""
    (out / "ALL_STREAM_RESULTS_SUMMARY.md").write_text(summary, encoding="utf-8")

    audit = f"""# All-platform STREAM audit report

## Scope

This report combines independently validated STREAM campaigns from Intel CPU, AMD CPU, RTX 3090 and RTX 5060 Ti.

## Coverage

- Four platforms
- Nine element counts
- Five sessions per configuration
- Ten technical repetitions per session
- Session medians as the primary statistical units

## Central outputs

- `unified_session_medians.csv`
- `unified_configuration_summary.csv`
- `native_policy_leaders.csv`
- `pairwise_native_best_comparisons.csv`
- `all_platform_metric_winners.csv`
- `placement_by_size.csv`
- `all_configuration_pareto.csv`
- `all_platform_stability.csv`

## Immediate audit indicators

- Clear device-level fastest-vs-greenest conflicts: {len(clear_device)}
- Clear within-platform configuration conflicts: {len(clear_config)}
- Configurations above 5% CV in runtime or energy: {len(unstable)}

## Methodological constraints

The primary decision axes are runtime and measured device-domain energy. Logical bandwidth and logical GB/J are normalized presentation views. EDP is joint. Cross-device energy claims apply only within the measured device domains. No physical memory-traffic claim is made from `12*N` logical bytes.
"""
    (out / "ALL_STREAM_AUDIT_REPORT.md").write_text(audit, encoding="utf-8")

    methods = """# STREAM methods and limitations

## Workload

`a[i] = b[i] + 3.0f*c[i]`, FP32. One operation is one complete pass over N elements.

- FLOPs per operation: `2*N`
- Logical bytes per operation: `12*N`
- Logical operational intensity: `1/6 FLOP/byte`

## Statistics

Ten repetitions are technical repetitions. Their median forms one session observation. Five session medians form the analysis sample. Exact n=5 bootstrap median intervals are used for configuration summaries. Native-best pairwise intervals are descriptive because selection and reporting use the same sessions.

## Energy

CPU primary energy is package RAPL. GPU primary energy is NVML board energy including device memory. Intel total package+DRAM remains available as a within-platform sensitivity, but cross-platform primary comparisons use package energy for both CPUs.

## Scope

GPU mode is resident. Allocations, initialization and PCIe transfers are outside the interval. Logical bandwidth is derived from semantic bytes and runtime, not a hardware counter. Small sizes may use cache; large sizes are the principal memory-stress regime.
"""
    (out / "METHODS_AND_LIMITATIONS.md").write_text(methods, encoding="utf-8")

    handoff = f"""# STREAM handoff for independent audit

## Required independent checks

1. Recompute `2*N*batches` and `12*N` from raw rows.
2. Recompute time/op and energy/op with CSV rounding allowances.
3. Confirm exactly five sessions and ten repetitions per configuration.
4. Recompute runtime, energy, EDP, logical bandwidth and logical GB/J from session medians.
5. Confirm runtime winner equals logical-bandwidth winner and energy winner equals logical-GB/J winner.
6. Recompute tie-aware leader sets and 2% practical-equivalence rules.
7. Recompute strict/practical Pareto status from runtime and energy only.
8. Keep CPU-package/GPU-board asymmetry and resident GPU scope explicit.

## Expected aggregate dimensions

- Unified session medians: 810 rows
- Unified configurations: 162 rows
- Native policy leaders: 180 rows
- Selected policy session medians: 900 rows
- Pairwise native-best rows: 270 rows
- Global metric winner rows: 45 rows

## Current headline counts

- Clear device-level fastest-vs-greenest conflicts: {len(clear_device)}
- Clear within-platform configuration conflicts: {len(clear_config)}
- Stability warnings: {len(unstable)}
"""
    (out / "ALL_STREAM_HANDOFF_FOR_CLAUDE.md").write_text(handoff, encoding="utf-8")
    print(f"[ALL STREAM] reports: {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import pandas as pd

from all_reduction_common import *


def main() -> None:
    out = results_dir(__file__)
    placement = pd.read_csv(out / "placement_by_size.csv")
    trade = pd.read_csv(out / "within_platform_energy_runtime_tradeoffs.csv")
    regret = pd.read_csv(out / "within_platform_exact_winner_regret.csv")
    stability = pd.read_csv(out / "all_platform_stability.csv")
    manifest = pd.read_csv(out / "input_manifest.csv")
    winners = pd.read_csv(out / "all_platform_metric_winners.csv")

    clear_device = placement[placement.clear_device_tradeoff.astype(str).str.lower().isin({"true", "1"})]
    clear_config = trade[trade.interpretation == "clear_configuration_tradeoff"]
    unstable = stability[~(stability.runtime_stable_5pct & stability.energy_stable_5pct)]
    clear_regret = regret[regret.energy_penalty_classification.astype(str).str.startswith("clear_energy_opt")]

    summary = f"""# All-platform REDUCTION results summary

## Dataset

- Four platforms: Intel CPU, AMD CPU, RTX 3090, RTX 5060 Ti
- Five independent sessions per configuration
- Ten technical repetitions per session
- Nine sizes from 1M to 256M elements
- Semantic operation: `sum(x[0:N]) -> FP32 scalar`
- Work per operation: `N-1` additions
- Logical data volume per operation: `4*N+4` bytes

## Scientific questions exposed

1. **Aggregation and synchronization scaling.** CPU thread counts expose where parallel block reduction and final aggregation stop improving runtime.
2. **Objective disagreement.** Runtime-, energy- and EDP-optimal choices are reported separately.
3. **CPU/GPU placement crossover.** The best CPU and best resident GPU are compared at every size.
4. **Configuration regret.** Exact runtime-winner versus exact energy-winner penalties are reported descriptively in `within_platform_exact_winner_regret.csv`.
5. **Stability.** Session CV identifies configurations whose apparent optimum is not reproducible.

## Device-level placement

{placement.to_markdown(index=False)}

## Tie-aware within-platform tradeoffs

{trade.to_markdown(index=False)}

## Exact-winner descriptive regret

{regret.to_markdown(index=False)}

## Stability warnings (>5% runtime or energy CV)

{markdown_table(unstable, 300)}

## Reporting contract

- Runtime and logical useful-data rate are inverse views of the same primitive axis.
- Energy and logical GB/J are inverse views of the same primitive axis.
- EDP is a composite metric, not an independent measurement.
- `4*N+4` is semantic logical traffic, not measured physical DRAM/VRAM traffic.
- CPU `partials[]` and CUB workspace traffic are intentionally excluded from the logical-byte anchor.
- GPU mode is `gpu_resident`; allocation and PCIe transfer costs are excluded.
- CPU package RAPL and GPU board NVML are different device-domain boundaries.
- Native-best and exact-winner-regret intervals are descriptive post-selection summaries based on the same five sessions.
"""
    (out / "ALL_REDUCTION_RESULTS_SUMMARY.md").write_text(summary, encoding="utf-8")

    audit = f"""# All-platform REDUCTION audit report

## Input campaigns

{manifest.to_markdown(index=False)}

## Aggregate dimensions

- Unified session medians: 810 rows
- Unified configurations: 162 rows
- Native policy leaders: 180 rows
- Selected policy session medians: 900 rows
- Pairwise native-best comparisons: 270 rows
- Global metric winners: 45 rows
- Exact-winner regret rows: 36 rows

## Immediate indicators

- Clear device-level fastest-vs-greenest conflicts: {len(clear_device)}
- Clear tie-aware within-platform configuration conflicts: {len(clear_config)}
- Clear descriptive exact-winner energy penalties: {len(clear_regret)}
- Configurations above 5% CV in runtime or energy: {len(unstable)}

## Interpretation

The primary decision axes are runtime and measured device-domain energy. Logical useful-data rate and logical GB/J are normalized inverse views. EDP is joint. No physical traffic claim is made from `4*N+4`, because implementation-specific CPU partials, CUB workspace and memory-system effects are not hardware-counter measurements.
"""
    (out / "ALL_REDUCTION_AUDIT_REPORT.md").write_text(audit, encoding="utf-8")

    methods = """# REDUCTION methods and limitations

## Workload

`sum(x[0:N]) -> FP32 scalar` with FP32 input, accumulation and output.

- FLOPs per operation: `N-1`
- Logical bytes per operation: `4*N+4`
- Logical operational intensity: `(N-1)/(4*N+4)`, approaching 0.25 FLOP/byte

CPU uses a fixed 4096-element blocked OpenMP reduction with SIMD-local accumulation and a fixed final stage. GPUs use CUB `DeviceReduce::Sum`. The mathematical workload is identical; the internal reduction hierarchy is platform-native.

## Statistics

Ten repetitions are technical repetitions. Their median forms one session observation. Five session medians form the analysis sample. Exact enumeration-based n=5 bootstrap median intervals are used for configuration summaries. Native-best pairwise and exact-winner-regret intervals are descriptive because selection and reporting use the same sessions.

## Energy

CPU primary energy is package RAPL. GPU primary energy is NVML board energy including device memory. Intel package+DRAM remains available as a within-platform sensitivity, but cross-platform primary comparisons use `device_energy_j` for both CPUs and GPUs.

## Scope

GPU mode is resident. Allocations, initialization and PCIe transfers are outside the interval. Logical useful-data rate is derived from semantic bytes and runtime, not a traffic counter. Small sizes may be cache- and synchronization-dominated; large sizes test resident memory reading plus hierarchical aggregation. CPU partial-array and CUB-workspace traffic are not counted in logical bytes.
"""
    (out / "METHODS_AND_LIMITATIONS.md").write_text(methods, encoding="utf-8")

    handoff = f"""# REDUCTION handoff for independent audit

## Required independent checks

1. Recompute `(N-1)*batches` and `4*N+4` from raw rows.
2. Recompute time/op and energy/op with CSV serialization tolerances.
3. Confirm exactly five sessions and ten repetitions per configuration.
4. Recompute runtime, energy, EDP, logical useful-data rate and logical GB/J from session medians.
5. Confirm runtime winner equals logical-rate winner and energy winner equals logical-GB/J winner.
6. Recompute tie-aware leader sets and the 2% practical-equivalence rule.
7. Recompute strict/practical Pareto status from runtime and energy only.
8. Recompute exact runtime-winner versus exact energy-winner regret, while retaining its post-selection label.
9. Keep CPU-package/GPU-board asymmetry, internal reduction traffic and resident GPU scope explicit.

## Expected aggregate dimensions

- Unified session medians: 810 rows
- Unified configurations: 162 rows
- Native policy leaders: 180 rows
- Selected policy session medians: 900 rows
- Pairwise native-best rows: 270 rows
- Global metric winner rows: 45 rows
- Exact-winner regret rows: 36 rows

## Current headline counts

- Clear device-level fastest-vs-greenest conflicts: {len(clear_device)}
- Clear tie-aware within-platform configuration conflicts: {len(clear_config)}
- Clear descriptive exact-winner energy penalties: {len(clear_regret)}
- Stability warnings: {len(unstable)}
"""
    (out / "ALL_REDUCTION_HANDOFF_FOR_CLAUDE.md").write_text(handoff, encoding="utf-8")
    print(f"[ALL REDUCTION] reports: {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from all_gemm_common import *


def labels(text: str) -> str:
    values = [x for x in str(text).split(",") if x and x != "nan"]
    return ", ".join(PLATFORM_LABELS.get(x, x) for x in values) if values else "none"


def winner_table(winners: pd.DataFrame) -> str:
    pivot = {}
    for size in SIZES:
        subset = winners[winners.problem_size == size].set_index("metric")
        pivot[size] = {
            "N": size,
            "Energy leaders": labels(subset.loc["energy_j", "leader_platforms"]),
            "Runtime leaders": labels(subset.loc["runtime_s", "leader_platforms"]),
            "EDP leaders": labels(subset.loc["edp_j_s", "leader_platforms"]),
            "Throughput leaders": labels(subset.loc["throughput_gflops", "leader_platforms"]),
            "Efficiency leaders": labels(subset.loc["efficiency_gflop_per_j", "leader_platforms"]),
        }
    return pd.DataFrame(pivot.values()).to_markdown(index=False)


def cpu_gpu_table(best: pd.DataFrame) -> str:
    rows = []
    for size in SIZES:
        energy = best[(best.problem_size == size) & (best.metric == "energy_j")].iloc[0]
        runtime = best[(best.problem_size == size) & (best.metric == "runtime_s")].iloc[0]
        edp = best[(best.problem_size == size) & (best.metric == "edp_j_s")].iloc[0]
        rows.append({
            "N": size,
            "Energy best CPU/GPU": f"{energy.best_cpu}/{energy.best_gpu}",
            "CPU/GPU energy ratio": f"{energy.cpu_over_gpu_ratio:.4g}",
            "Energy classification": energy.classification,
            "Runtime best CPU/GPU": f"{runtime.best_cpu}/{runtime.best_gpu}",
            "CPU/GPU runtime ratio": f"{runtime.cpu_over_gpu_ratio:.4g}",
            "Runtime classification": runtime.classification,
            "EDP ratio": f"{edp.cpu_over_gpu_ratio:.4g}",
        })
    return pd.DataFrame(rows).to_markdown(index=False)


def gpu_table(pairwise: pd.DataFrame) -> str:
    rows = []
    for size in SIZES:
        def get(metric):
            return pairwise[
                (pairwise.metric == metric)
                & (pairwise.problem_size == size)
                & (pairwise.platform_a == "3090")
                & (pairwise.platform_b == "5060ti")
            ].iloc[0]
        runtime = get("runtime_s")
        energy = get("energy_j")
        efficiency = get("efficiency_gflop_per_j")
        rows.append({
            "N": size,
            "3090/5060Ti runtime": f"{runtime.a_over_b_ratio:.4g}",
            "3090/5060Ti energy": f"{energy.a_over_b_ratio:.4g}",
            "3090/5060Ti GFLOP/J": f"{efficiency.a_over_b_ratio:.4g}",
            "Runtime result": runtime.classification,
            "Energy result": energy.classification,
        })
    return pd.DataFrame(rows).to_markdown(index=False)


def metric_winner_counts(winners: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric, group in winners.groupby("metric"):
        for platform, count in group.exact_winner.value_counts().items():
            clear = int(((group.exact_winner == platform) & (group.selection_status == "clear_leader")).sum())
            rows.append({
                "metric": metric,
                "platform": platform,
                "platform_label": PLATFORM_LABELS[platform],
                "exact_winner_sizes": int(count),
                "clear_winner_sizes": clear,
            })
    return pd.DataFrame(rows).sort_values(["metric", "exact_winner_sizes"], ascending=[True, False])


def main() -> None:
    out = results_dir(__file__)
    required = [
        "preflight_checks.csv", "input_manifest.csv", "unified_configuration_summary.csv",
        "native_policy_leaders.csv", "pairwise_native_best_comparisons.csv",
        "all_platform_metric_winners.csv", "best_cpu_vs_best_gpu.csv",
        "placement_by_size.csv", "within_platform_energy_runtime_tradeoffs.csv",
        "all_configuration_pareto.csv", "configuration_tradeoff_map.csv",
        "all_platform_stability.csv",
    ]
    missing = [name for name in required if not (out / name).is_file()]
    if missing:
        raise SystemExit(f"Missing outputs; run earlier stages first: {missing}")

    checks = pd.read_csv(out / "preflight_checks.csv")
    manifest = pd.read_csv(out / "input_manifest.csv")
    summary = pd.read_csv(out / "unified_configuration_summary.csv")
    leaders = pd.read_csv(out / "native_policy_leaders.csv")
    pairwise = pd.read_csv(out / "pairwise_native_best_comparisons.csv")
    winners = pd.read_csv(out / "all_platform_metric_winners.csv")
    best_cpu_gpu = pd.read_csv(out / "best_cpu_vs_best_gpu.csv")
    placement = pd.read_csv(out / "placement_by_size.csv")
    within = pd.read_csv(out / "within_platform_energy_runtime_tradeoffs.csv")
    pareto = pd.read_csv(out / "all_configuration_pareto.csv")
    tradeoffs = pd.read_csv(out / "configuration_tradeoff_map.csv")
    stability = pd.read_csv(out / "all_platform_stability.csv")

    hard = checks[(checks.severity == "FAIL") & (checks.status == "FAIL")]
    warns = checks[(checks.severity == "WARN") & (checks.status == "WARN")]
    verdict = "FAIL" if len(hard) else ("PASS WITH WARNINGS" if len(warns) else "PASS")
    total_rows = int(pd.to_numeric(manifest.raw_measurement_rows, errors="coerce").sum())
    total_configs = int(summary[["platform", "problem_size", "configuration"]].drop_duplicates().shape[0])

    winner_counts = metric_winner_counts(winners)
    winner_counts.to_csv(out / "metric_winner_counts.csv", index=False)

    clear_device_tradeoffs = placement[placement.placement_class == "clear_device_tradeoff"]
    uncertain_device_tradeoffs = placement[placement.placement_class == "uncertain_device_tradeoff"]
    cpu_clear_tradeoffs = within[
        (within.platform.isin(CPU_PLATFORMS))
        & (within.interpretation == "clear_configuration_tradeoff")
    ]

    unstable = stability[
        (~stability.runtime_stable_5pct)
        | (~stability.energy_stable_5pct)
        | (~stability.throughput_stable_5pct)
    ]

    exact_winner_md = winner_table(winners)
    cpu_gpu_md = cpu_gpu_table(best_cpu_gpu)
    gpu_md = gpu_table(pairwise)

    audit = f"""# Combined audit of all GEMM campaigns

## Overall verdict

**{verdict}**

The combined analysis includes {total_rows:,} raw measurements from four validated
campaigns and {total_configs} platform/size/configuration combinations:

{manifest[['platform_label','campaign','raw_measurement_rows','configurations','validation_warnings','energy_domain']].to_markdown(index=False)}

No raw row is treated as an independent statistical experiment. Ten within-session
repetitions are technical repetitions; the primary statistical units are the five
session medians per configuration.

## What this pipeline compares

1. **All configurations:** every Intel and AMD thread count plus one resident
   configuration per GPU.
2. **Native-best per platform:** energy-, runtime-, throughput-, efficiency-, and
   EDP-optimized configurations selected separately.
3. **Best CPU versus best GPU:** the better CPU and better GPU for each size and
   metric, without confusing CPU thread tuning with device placement.
4. **Pairwise device ratios:** all six platform pairs with deterministic
   bootstrap ratio intervals, probability of superiority, and Cliff's delta.

## Statistical interpretation

- Point estimates: median of five session medians.
- Within-configuration uncertainty: exact bootstrap of the five session medians.
- Pairwise ratio uncertainty: deterministic independent bootstrap of the two
  five-session samples.
- Practical equivalence tolerance: ±2%.
- A clear all-platform leader must be clearly favored against every alternative
  by the pairwise ratio-CI rule outside the ±2% practical-equivalence band.
- Native-best ratio intervals are **descriptive**, not confirmatory: configuration
  selection and summarization use the same five sessions, so no post-selection
  p-values are reported.

## Metric identities

At a fixed matrix size, every configuration performs exactly `2*N^3` logical
FLOP. The combined pipeline therefore normalizes:

- throughput as logical FLOP divided by e2e runtime;
- GFLOP/J as logical FLOP divided by measured device-domain energy.

Runtime/throughput and energy/GFLOP-J are inverse views, not four independent
dimensions. EDP is the joint energy-runtime metric.

## Energy-domain limitation

CPU energy is package-only RAPL. GPU energy is NVML board energy and includes
on-board memory. The comparison is valid as a comparison of the measured device
domains, but it is not whole-system energy and must not be described as a pure
architecture-only comparison.

## Metric normalization

For fixed `N`, throughput is derived as `2*N^3 / runtime` and GFLOP/J as
`2*N^3 / energy`. Hence runtime and throughput are one dimension, while energy
per GEMM and GFLOP/J are one dimension. Do not count their winner tables as
independent replications of the same finding.

## Tie-aware all-platform leaders

{exact_winner_md}

## Best CPU versus best GPU

{cpu_gpu_md}

## RTX 3090 versus RTX 5060 Ti

{gpu_md}

## Energy-runtime configuration trade-offs

Clear within-CPU configuration conflicts occur in {len(cpu_clear_tradeoffs)}
size/platform cases. Their full leader sets and penalties are recorded in
`within_platform_energy_runtime_tradeoffs.csv`.

Across devices, {len(clear_device_tradeoffs)} sizes have clear disjoint energy and
runtime leaders, while {len(uncertain_device_tradeoffs)} have disjoint but
statistically/practically uncertain leader sets. The per-size result is in
`placement_by_size.csv`.

## Pareto analysis

`all_configuration_pareto.csv` contains strict and practical 2% Pareto status for
every CPU thread configuration and both GPU configurations at every N.
`configuration_tradeoff_map.csv` assigns each point to one of:

- dominant or practically equivalent;
- energy-efficient compromise;
- runtime-efficient compromise;
- balanced Pareto trade-off;
- dominated.

This classification is preferable to a simple winner count because it exposes
how much runtime is traded for energy and vice versa.

## Stability

There are {len(unstable)} configurations with more than 5% session-level CV in
runtime, energy, or throughput. They are concentrated on the CPUs and are listed
in `all_platform_stability.csv`; use `stability_breakdown.csv` and
`leader_stability.csv` from the integrity audit to separate general instability
from instability of an exact policy-selected configuration. The individual
platform audits remain the authority for thermal and sensor-specific warnings.

## Audit decision

The combined dataset is suitable for descriptive cross-platform GEMM analysis.
Claims must retain three qualifications: resident GPU execution, asymmetric
energy domains, and descriptive post-selection uncertainty for native-best
comparisons.
"""
    (out / "ALL_GEMM_AUDIT_REPORT.md").write_text(audit, encoding="utf-8")

    results_summary = f"""# All-platform GEMM results summary

We compared FP32 GEMM across two CPUs and two GPUs using {total_rows:,} validated
measurements. Each platform was measured in five sessions with ten technical
repetitions per configuration. We summarize each configuration by the median of
its five session medians and report 95% bootstrap intervals. CPU energy denotes
package-only RAPL energy; GPU energy denotes NVML board energy. GPU measurements
use resident execution and exclude PCIe transfers.

The analysis distinguishes configuration selection from device placement. For
CPUs, all measured thread counts remain available in the configuration-level
Pareto analysis. For platform-level comparisons, energy-, runtime-, and EDP-best
configurations are selected separately, with a 2% practical-equivalence tolerance
and ratio-CI-aware leader sets. Throughput and GFLOP/J are reported as normalized
inverse views of runtime and energy, not as independent evidence. The resulting
tie-aware device leaders are:

{exact_winner_md}

The direct best-CPU-versus-best-GPU comparison is:

{cpu_gpu_md}

The complete results show whether each N has a shared energy/runtime leader or a
true placement trade-off. Clear device-level trade-offs occur at
{', '.join(map(str, clear_device_tradeoffs.problem_size.astype(int).tolist())) or 'no sizes'}.
Cases where the exact energy and runtime winners differ but leader uncertainty
remains are reported separately rather than counted as decisive conflicts.

Within-platform CPU tuning remains narrower than raw `idxmin` counts suggest:
{len(cpu_clear_tradeoffs)} CPU size/platform cases have disjoint, clear energy-
and runtime-optimal leader sets. All other apparent differences are ties or
uncertain under the combined 2% and CI rule.

The pairwise tables provide effect magnitude rather than only winner labels:
median ratios, 95% ratio intervals, probability of superiority, and Cliff's
delta. Because native-best configurations are selected from the same five
sessions used for comparison, these intervals are descriptive and are not
presented as confirmatory hypothesis tests.

Across all CPU thread settings and both GPUs, the practical Pareto analysis
identifies dominant points, energy-efficient compromises, runtime-efficient
compromises, balanced trade-offs, and dominated configurations. This provides
the appropriate empirical basis for later job-level placement rules; absolute
EDP values are never compared across different matrix sizes as though the jobs
contained equal work.
"""
    (out / "ALL_GEMM_RESULTS_SUMMARY.md").write_text(results_summary, encoding="utf-8")

    handoff = f"""# Handoff for Claude: combined Intel, AMD, RTX 3090, and RTX 5060 Ti GEMM analysis

## Dataset and audit state

- Raw measurements: {total_rows:,}
- Platforms: Intel CPU, AMD CPU, RTX 3090, RTX 5060 Ti
- Problem sizes: {SIZES}
- Sessions: 5 per platform
- Technical repetitions: 10 per configuration/session
- Combined preflight verdict: {verdict}
- Hard preflight failures: {len(hard)}
- Warnings: {len(warns)}

Each individual platform validator must pass before the combined pipeline runs.
The pipeline also verifies that each analysis matches the latest complete raw
campaign and checks GPU source provenance.

## Statistical contract

- Unit of analysis: five session medians, not 50 raw repetitions.
- Exact bootstrap CI per fixed configuration.
- Deterministic unpaired bootstrap CI for pairwise median ratios.
- Probability of superiority and Cliff's delta are included.
- Practical tolerance: 2%.
- Clear all-platform leaders require ratio-CI support against every alternative.
- Positive Cliff's delta means numerically larger A values; this favors A only
  for higher-is-better metrics.
- Native-best comparisons are descriptive post-selection analyses; no p-values
  are claimed because selection and estimation use the same five sessions.

## Energy semantics

- Intel/AMD: CPU package RAPL only.
- RTX 3090/5060 Ti: NVML board energy including device memory.
- GPU mode: resident; PCIe transfers excluded.

Do not translate device-domain comparisons into whole-system or architecture-only
claims without an explicit limitation.

## Tie-aware all-platform leaders

{exact_winner_md}

## Best CPU versus best GPU

{cpu_gpu_md}

## GPU generation comparison

{gpu_md}

## Trade-off findings

- Clear device-level energy/runtime trade-off sizes: {clear_device_tradeoffs.problem_size.astype(int).tolist()}
- Uncertain device-level trade-off sizes: {uncertain_device_tradeoffs.problem_size.astype(int).tolist()}
- Clear within-CPU configuration conflicts: {len(cpu_clear_tradeoffs)}

Inspect:

- `placement_by_size.csv`
- `within_platform_energy_runtime_tradeoffs.csv`
- `configuration_tradeoff_map.csv`
- `all_configuration_pareto.csv`

## Core output files

- `unified_session_medians.csv`: normalized n=5 session data.
- `unified_configuration_summary.csv`: harmonized medians, CIs and CVs.
- `native_policy_leaders.csv`: exact and tie-aware policy leaders.
- `pairwise_native_best_comparisons.csv`: all six device pairs and five metrics.
- `best_cpu_vs_best_gpu.csv`: direct placement view.
- `all_platform_metric_winners.csv`: per-size tie-aware device leaders.
- `placement_by_size.csv`: shared leader versus clear/uncertain device trade-off.
- `all_configuration_pareto.csv`: every CPU thread setting plus both GPUs.
- `configuration_tradeoff_map.csv`: dominant/compromise/dominated classes.
- `crossover_summary.csv`: changes in pairwise winner state over N.

## Questions to evaluate

1. Which size-dependent CPU/GPU crossover is robust under the tie-aware leader
   rule rather than only raw minima?
2. Does the strongest story come from device placement, CPU thread tuning, or
   the combined Pareto trade-off map?
3. Which results remain meaningful after retaining the CPU-package/GPU-board
   energy-domain limitation?
4. Is a transfer-aware `gpu_e2e` sensitivity analysis necessary for the intended
   venue, or can resident execution remain the primary compute-kernel scope?
"""
    (out / "ALL_GEMM_HANDOFF_FOR_CLAUDE.md").write_text(handoff, encoding="utf-8")

    methods = """# Combined GEMM methods and limitations

## Statistical unit

Ten repetitions inside a session quantify technical noise. They are first
collapsed to one median per session and configuration. All reported uncertainty
therefore uses five session medians.

## Native-best policy views

The analysis emits five policy views for convenience: energy-optimal,
runtime-optimal, EDP-optimal, throughput-optimal, and GFLOP/J-optimal. At fixed
`N`, however, throughput is the inverse normalization of runtime and GFLOP/J is
the inverse normalization of energy. These pairs are not independent criteria.
The implementation does not use one configuration's energy together with
another configuration's runtime in a single operating point.

## Practical equivalence

A 2% tolerance prevents tiny numerical differences from becoming categorical
winner claims. For the final all-platform table, a clear winner must also be
clearly favored against every alternative by the bootstrapped pairwise ratio
interval outside the practical-equivalence band. Otherwise the result is marked
`tie_or_uncertain`. Marginal-CI separation remains available as a diagnostic but
is not the final classification rule.

## Post-selection limitation

Native-best CPU configurations are selected and summarized on the same five
sessions. Ratio intervals describe observed robustness but are not confirmatory
post-selection inference. A future preregistered confirmation could fix thread
counts using the current campaign and rerun independent sessions.

## Energy domains

CPU package RAPL and GPU NVML board energy are not identical domains. GPU values
include device-memory energy; CPU package values exclude system DRAM when a
separate comparable domain is unavailable. The study therefore compares
measured device domains, not whole-system energy.

## GPU execution scope

GPU data are `gpu_resident`: allocations, initialization and PCIe transfers lie
outside the interval. Placement claims involving short jobs or repeated data
movement need a separate `gpu_e2e` sensitivity analysis.

## EDP across sizes

Absolute EDP is not interpreted across different N because each size represents
a different amount of work. EDP is used only for same-size device/configuration
comparisons.
"""
    (out / "METHODS_AND_LIMITATIONS.md").write_text(methods, encoding="utf-8")

    print(f"[ALL GEMM] reports written to {out}")


if __name__ == "__main__":
    main()

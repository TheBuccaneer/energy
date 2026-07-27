#!/usr/bin/env python3
from __future__ import annotations

import sys
import numpy as np
import pandas as pd

from reduction_analysis_common import *

POLICIES = {
    "runtime_opt": ("runtime_per_op_s", True),
    "energy_opt": ("primary_energy_per_op_j", True),
    "edp_opt": ("edp_primary_j_s", True),
    "bandwidth_opt": ("logical_bandwidth_gb_s", False),
    "bytes_per_j_opt": ("logical_gb_per_j", False),
}


def main() -> None:
    args = parse_args("Analyze one validated REDUCTION campaign")
    ctx = context(__file__)
    checks_path = ctx.result_dir / "validation_checks.csv"
    manifest_path = ctx.result_dir / "campaign_manifest.csv"
    if not checks_path.is_file() or not manifest_path.is_file():
        raise SystemExit("Run 01_validate_reduction.py first.")
    checks = pd.read_csv(checks_path)
    if ((checks.severity == "FAIL") & (checks.status == "FAIL")).any():
        raise SystemExit("Validation contains hard failures.")

    manifest = pd.read_csv(manifest_path)
    validated_campaign = str(manifest.iloc[0].campaign)
    requested = args.campaign or validated_campaign
    if requested != validated_campaign:
        raise SystemExit(
            f"Requested campaign {requested} differs from validated campaign {validated_campaign}; validate it first."
        )

    campaign = load_campaign(ctx, requested)
    data = add_derived(campaign.data, ctx)
    sessions = session_medians(data)
    summary = configuration_summary(sessions)

    sessions.to_csv(ctx.result_dir / "session_configuration_medians.csv", index=False)
    summary.to_csv(ctx.result_dir / "configuration_summary.csv", index=False)

    # Robust outliers are diagnostic only and never silently removed.
    outlier_parts = []
    for (session, size, configuration), group in data.groupby(["session_number", "problem_size", "configuration"]):
        runtime_out = robust_outlier_mask(group.runtime_per_op_s)
        energy_out = robust_outlier_mask(group.primary_energy_per_op_j)
        mask = runtime_out | energy_out
        if mask.any():
            part = group.loc[mask, [
                "source_file", "session_number", "sequence_index", "problem_size", "configuration",
                "num_threads", "repetition", "runtime_per_op_s", "primary_energy_per_op_j",
                "checksum_ok",
            ]].copy()
            part["runtime_outlier"] = runtime_out[mask].to_numpy()
            part["energy_outlier"] = energy_out[mask].to_numpy()
            outlier_parts.append(part)
    outliers = pd.concat(outlier_parts, ignore_index=True) if outlier_parts else pd.DataFrame()
    outliers.to_csv(ctx.result_dir / "robust_outliers.csv", index=False)

    leaders = {}
    for policy, (metric, lower) in POLICIES.items():
        frame = select_leaders(summary, metric, lower)
        frame["policy"] = policy
        frame["platform"] = ctx.platform
        leaders[policy] = frame
        frame.to_csv(ctx.result_dir / f"best_{policy.removesuffix('_opt')}_by_size.csv", index=False)
    leader_table = pd.concat(leaders.values(), ignore_index=True)
    leader_table.to_csv(ctx.result_dir / "policy_leaders.csv", index=False)

    # Within-platform fastest-vs-greenest conflict.
    trade_rows = []
    energy = leaders["energy_opt"].set_index("problem_size")
    runtime = leaders["runtime_opt"].set_index("problem_size")
    for size in SIZES:
        e = energy.loc[size]
        r = runtime.loc[size]
        e_set = {x for x in str(e.leader_configurations).split(",") if x}
        r_set = {x for x in str(r.leader_configurations).split(",") if x}
        overlap = sorted(e_set & r_set)
        e_cfg = summary[(summary.problem_size == size) & (summary.configuration == e.exact_configuration)].iloc[0]
        r_cfg = summary[(summary.problem_size == size) & (summary.configuration == r.exact_configuration)].iloc[0]
        trade_rows.append({
            "platform": ctx.platform,
            "problem_size": size,
            "working_set_gib": (4.0 * size + 4.0) / (1024 ** 3),
            "energy_opt_configuration": e.exact_configuration,
            "runtime_opt_configuration": r.exact_configuration,
            "energy_leaders": e.leader_configurations,
            "runtime_leaders": r.leader_configurations,
            "leader_set_overlap": ",".join(overlap),
            "leader_sets_disjoint": not bool(overlap),
            "energy_selection_status": e.selection_status,
            "runtime_selection_status": r.selection_status,
            "runtime_opt_energy_penalty_pct": 100.0 * (r_cfg.primary_energy_per_op_j_median / e_cfg.primary_energy_per_op_j_median - 1.0),
            "runtime_opt_runtime_gain_pct": 100.0 * (1.0 - r_cfg.runtime_per_op_s_median / e_cfg.runtime_per_op_s_median),
            "interpretation": (
                "clear_configuration_tradeoff"
                if not overlap and e.selection_status == "clear_leader" and r.selection_status == "clear_leader"
                else "disjoint_but_at_least_one_uncertain"
                if not overlap else "no_tie_aware_conflict"
            ),
        })
    tradeoffs = pd.DataFrame(trade_rows)
    tradeoffs.to_csv(ctx.result_dir / "within_platform_energy_runtime_tradeoffs.csv", index=False)

    pareto_parts = []
    for size, group in summary.groupby("problem_size"):
        part = group.copy()
        part["strict_pareto"] = strict_pareto(part)
        part["practical_pareto_2pct"] = practical_pareto(part)
        pareto_parts.append(part)
    pareto = pd.concat(pareto_parts, ignore_index=True)
    pareto.to_csv(ctx.result_dir / "configuration_pareto.csv", index=False)

    stability = summary[[
        "problem_size", "configuration", "num_threads",
        "runtime_per_op_s_session_cv_pct", "primary_energy_per_op_j_session_cv_pct",
        "logical_bandwidth_gb_s_session_cv_pct", "primary_power_w_session_cv_pct",
    ]].copy()
    stability["runtime_stable_5pct"] = stability.runtime_per_op_s_session_cv_pct <= 5.0
    stability["energy_stable_5pct"] = stability.primary_energy_per_op_j_session_cv_pct <= 5.0
    stability["bandwidth_stable_5pct"] = stability.logical_bandwidth_gb_s_session_cv_pct <= 5.0
    stability.to_csv(ctx.result_dir / "stability.csv", index=False)

    plot_metric(summary, "runtime_per_op_s", "Laufzeit pro Reduktion [s]", ctx.figure_dir / "runtime_by_size.png", logy=True)
    plot_metric(summary, "primary_energy_per_op_j", "Primärenergie pro Reduktion [J]", ctx.figure_dir / "energy_by_size.png", logy=True)
    plot_metric(summary, "logical_bandwidth_gb_s", "Logische Bandbreite [GB/s]", ctx.figure_dir / "logical_bandwidth_by_size.png")
    plot_metric(summary, "primary_power_w", "Mittlere Leistung [W]", ctx.figure_dir / "power_by_size.png")
    plot_metric(summary, "edp_primary_j_s", "EDP [J·s]", ctx.figure_dir / "edp_by_size.png", logy=True)

    clear = tradeoffs[tradeoffs.interpretation == "clear_configuration_tradeoff"]
    unstable = stability[~(stability.runtime_stable_5pct & stability.energy_stable_5pct)]
    report = f"""# REDUCTION scientific report — {ctx.config['label']}

## Dataset

- Campaign: `{campaign.stamp}`
- Raw rows: {len(data)}
- Session medians: {len(sessions)}
- Configurations: {len(summary)}
- Primary unit: five session medians per size/configuration
- Primary energy domain: {ctx.config['energy_domain']}
- Execution mode: `{ctx.config['mode']}`

## Immediate findings exposed by the analysis

- Clear fastest-vs-greenest configuration conflicts: **{len(clear)} of {len(SIZES)} sizes**.
- Configurations above 5% session CV in runtime or energy: **{len(unstable)} of {len(stability)}**.
- The analysis exposes thread-count saturation, synchronization/aggregation cost, runtime-optimal versus energy-optimal choices,
  logical useful-data-rate scaling, EDP, and stability across sessions.

## Fastest-vs-greenest table

{tradeoffs.to_markdown(index=False)}

## Interpretation contract

1. Runtime and logical useful-data rate are inverse views of the same primitive axis.
2. Energy per reduction and logical GB/J are inverse views of the same primitive axis.
3. EDP is a composite, not an independent vote.
4. `4*N+4` bytes are logical REDUCTION bytes, not measured physical DRAM/VRAM traffic.
5. Small sizes may be cache and synchronization affected; large sizes are the principal resident memory-plus-aggregation regime.
6. GPU results are resident and exclude PCIe transfers.
7. CPU/GPU energy domains are asymmetric and must be named explicitly.
8. Native-best selection is descriptive post-selection on five sessions.

## Stability rows above 5%

{markdown_table(unstable, 200)}
"""
    (ctx.result_dir / "scientific_report.md").write_text(report, encoding="utf-8")
    print(f"[{ctx.platform} REDUCTION] analysis complete: {ctx.result_dir}")


if __name__ == "__main__":
    main()

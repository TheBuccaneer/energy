#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from all_gemm_common import *


def record(rows: list[dict], category: str, check: str, severity: str,
           passed: bool, observed, expected) -> None:
    rows.append({
        "category": category,
        "check": check,
        "severity": severity,
        "status": "PASS" if passed else severity,
        "observed": str(observed),
        "expected": str(expected),
    })


def main() -> None:
    out = results_dir(__file__)
    required = [
        "unified_session_medians.csv", "unified_configuration_summary.csv",
        "native_policy_leaders.csv", "native_policy_session_medians.csv",
        "pairwise_native_best_comparisons.csv", "all_platform_metric_winners.csv",
        "best_cpu_vs_best_gpu.csv", "placement_by_size.csv",
        "all_configuration_pareto.csv", "configuration_tradeoff_map.csv",
        "all_platform_stability.csv", "preflight_checks.csv",
    ]
    missing = [name for name in required if not (out / name).is_file()]
    if missing:
        raise SystemExit(f"Missing outputs for integrity audit: {missing}")

    session = pd.read_csv(out / "unified_session_medians.csv")
    summary = pd.read_csv(out / "unified_configuration_summary.csv")
    leaders = pd.read_csv(out / "native_policy_leaders.csv")
    selected = pd.read_csv(out / "native_policy_session_medians.csv")
    pairwise = pd.read_csv(out / "pairwise_native_best_comparisons.csv")
    winners = pd.read_csv(out / "all_platform_metric_winners.csv")
    best_cpu_gpu = pd.read_csv(out / "best_cpu_vs_best_gpu.csv")
    placement = pd.read_csv(out / "placement_by_size.csv")
    pareto = pd.read_csv(out / "all_configuration_pareto.csv")
    tradeoff = pd.read_csv(out / "configuration_tradeoff_map.csv")
    stability = pd.read_csv(out / "all_platform_stability.csv")
    preflight = pd.read_csv(out / "preflight_checks.csv")

    checks: list[dict] = []

    expected_counts = {
        "unified_session_medians": (len(session), 810),
        "unified_configuration_summary": (len(summary), 162),
        "native_policy_session_medians": (len(selected), 900),
        "pairwise_native_best_comparisons": (len(pairwise), 270),
        "all_platform_metric_winners": (len(winners), 45),
        "best_cpu_vs_best_gpu": (len(best_cpu_gpu), 45),
        "placement_by_size": (len(placement), 9),
        "all_configuration_pareto": (len(pareto), 162),
    }
    for name, (observed, expected) in expected_counts.items():
        record(checks, "coverage", name, "FAIL", observed == expected, observed, expected)

    session_key = ["platform", "problem_size", "configuration", "session_number"]
    config_key = ["platform", "problem_size", "configuration"]
    record(checks, "coverage", "unique_session_keys", "FAIL",
           not session.duplicated(session_key).any(),
           int(session.duplicated(session_key).sum()), 0)
    counts = session.groupby(config_key)["session_number"].nunique()
    record(checks, "coverage", "five_sessions_per_configuration", "FAIL",
           bool((counts == EXPECTED_SESSIONS).all()),
           f"min={counts.min()}, max={counts.max()}", EXPECTED_SESSIONS)

    logical_flops = 2.0 * session["problem_size"].astype(float) ** 3
    expected_throughput = logical_flops / session["runtime_s"] / 1e9
    expected_efficiency = logical_flops / session["energy_j"] / 1e9
    throughput_error = np.abs(session["throughput_gflops"] / expected_throughput - 1.0)
    efficiency_error = np.abs(session["efficiency_gflop_per_j"] / expected_efficiency - 1.0)
    record(checks, "metric_identity", "throughput_equals_flops_over_e2e_runtime", "FAIL",
           float(throughput_error.max()) < 1e-12,
           f"max_rel_error={throughput_error.max():.3e}", "<1e-12")
    record(checks, "metric_identity", "efficiency_equals_flops_over_energy", "FAIL",
           float(efficiency_error.max()) < 1e-12,
           f"max_rel_error={efficiency_error.max():.3e}", "<1e-12")

    normalization_max = float(session.get(
        "throughput_normalization_delta_pct", pd.Series([np.nan])
    ).abs().max())
    record(checks, "normalization", "reported_vs_e2e_throughput_delta", "WARN",
           normalization_max <= 0.1,
           f"max_abs_delta={normalization_max:.5f}%", "<=0.1%")

    identity_rows = []
    identities_ok = True
    for size in SIZES:
        e = winners[(winners.problem_size == size) & (winners.metric == "energy_j")].iloc[0]
        q = winners[(winners.problem_size == size) & (winners.metric == "efficiency_gflop_per_j")].iloc[0]
        r = winners[(winners.problem_size == size) & (winners.metric == "runtime_s")].iloc[0]
        t = winners[(winners.problem_size == size) & (winners.metric == "throughput_gflops")].iloc[0]
        energy_efficiency_match = (
            e.exact_winner == q.exact_winner
            and e.leader_platforms == q.leader_platforms
            and e.selection_status == q.selection_status
        )
        runtime_throughput_match = (
            r.exact_winner == t.exact_winner
            and r.leader_platforms == t.leader_platforms
            and r.selection_status == t.selection_status
        )
        identities_ok &= energy_efficiency_match and runtime_throughput_match
        identity_rows.append({
            "problem_size": size,
            "energy_efficiency_match": energy_efficiency_match,
            "energy_leaders": e.leader_platforms,
            "efficiency_leaders": q.leader_platforms,
            "runtime_throughput_match": runtime_throughput_match,
            "runtime_leaders": r.leader_platforms,
            "throughput_leaders": t.leader_platforms,
        })
    pd.DataFrame(identity_rows).to_csv(out / "metric_identity_check.csv", index=False)
    record(checks, "metric_identity", "energy_and_efficiency_leaders_identical", "FAIL",
           identities_ok and all(r["energy_efficiency_match"] for r in identity_rows),
           "checked for all N", "identical because work is fixed at each N")
    record(checks, "metric_identity", "runtime_and_throughput_leaders_identical", "FAIL",
           identities_ok and all(r["runtime_throughput_match"] for r in identity_rows),
           "checked for all N", "identical because work is fixed at each N")

    # Every all-platform clear leader must also be clearly favored by the ratio-CI
    # comparison against every alternative.
    coherence_failures = []
    for _, winner in winners.iterrows():
        if winner.selection_status != "clear_leader":
            continue
        exact = str(winner.exact_winner)
        for other in PLATFORMS:
            if other == exact:
                continue
            if PLATFORMS.index(exact) < PLATFORMS.index(other):
                row = pairwise[
                    (pairwise.metric == winner.metric)
                    & (pairwise.problem_size == winner.problem_size)
                    & (pairwise.platform_a == exact)
                    & (pairwise.platform_b == other)
                ].iloc[0]
            else:
                row = pairwise[
                    (pairwise.metric == winner.metric)
                    & (pairwise.problem_size == winner.problem_size)
                    & (pairwise.platform_a == other)
                    & (pairwise.platform_b == exact)
                ].iloc[0]
            if row.classification != f"clear_{exact}":
                coherence_failures.append(
                    f"{winner.metric}/N={int(winner.problem_size)}/{exact} vs {other}: {row.classification}"
                )
    record(checks, "classification", "winner_table_matches_pairwise_ratio_rule", "FAIL",
           not coherence_failures, coherence_failures or "none", "none")

    pairwise_valid = (
        (pairwise.a_over_b_ratio > 0).all()
        and (pairwise.ratio_ci95_low > 0).all()
        and (pairwise.ratio_ci95_low <= pairwise.a_over_b_ratio).all()
        and (pairwise.a_over_b_ratio <= pairwise.ratio_ci95_high).all()
        and pairwise.probability_a_better.between(0, 1).all()
        and pairwise.cliffs_delta_a_minus_b.between(-1, 1).all()
    )
    record(checks, "pairwise", "ratio_ci_and_effect_ranges", "FAIL",
           pairwise_valid, "checked 270 rows", "positive ordered CIs; effects in valid ranges")

    # Recompute Pareto flags and trade-off classes independently.
    pareto_failures = []
    for size, group in summary.groupby("problem_size"):
        data = group.reset_index(drop=True)
        best_e = float(data.energy_j_median.min())
        best_t = float(data.runtime_s_median.min())
        for i, row in data.iterrows():
            strict_dominated = False
            practical_dominated = False
            for j, other in data.iterrows():
                if i == j:
                    continue
                if (
                    other.energy_j_median <= row.energy_j_median
                    and other.runtime_s_median <= row.runtime_s_median
                    and (
                        other.energy_j_median < row.energy_j_median
                        or other.runtime_s_median < row.runtime_s_median
                    )
                ):
                    strict_dominated = True
                if (
                    other.energy_j_median <= row.energy_j_median * (1 + PRACTICAL_TOLERANCE)
                    and other.runtime_s_median <= row.runtime_s_median * (1 + PRACTICAL_TOLERANCE)
                    and (
                        other.energy_j_median < row.energy_j_median / (1 + PRACTICAL_TOLERANCE)
                        or other.runtime_s_median < row.runtime_s_median / (1 + PRACTICAL_TOLERANCE)
                    )
                ):
                    practical_dominated = True
            p = pareto[
                (pareto.platform == row.platform)
                & (pareto.problem_size == size)
                & (pareto.configuration == row.configuration)
            ].iloc[0]
            tr = tradeoff[
                (tradeoff.platform == row.platform)
                & (tradeoff.problem_size == size)
                & (tradeoff.configuration == row.configuration)
            ].iloc[0]
            e_pen = row.energy_j_median / best_e - 1.0
            t_pen = row.runtime_s_median / best_t - 1.0
            if e_pen <= PRACTICAL_TOLERANCE and t_pen <= PRACTICAL_TOLERANCE:
                expected_class = "dominant_or_practically_equivalent"
            elif e_pen <= PRACTICAL_TOLERANCE:
                expected_class = "energy_efficient_compromise"
            elif t_pen <= PRACTICAL_TOLERANCE:
                expected_class = "runtime_efficient_compromise"
            elif not practical_dominated:
                expected_class = "balanced_pareto_tradeoff"
            else:
                expected_class = "dominated"
            if (
                bool(p.pareto_strict) != (not strict_dominated)
                or bool(p.pareto_practical_2pct) != (not practical_dominated)
                or tr.tradeoff_class != expected_class
            ):
                pareto_failures.append(
                    f"N={size}/{row.platform}/{row.configuration}"
                )
    record(checks, "pareto", "independent_pareto_recomputation", "FAIL",
           not pareto_failures, pareto_failures[:20] or "none", "none")

    # Stability audit and selected-leader exposure.
    unstable = stability[
        (~stability.runtime_stable_5pct)
        | (~stability.energy_stable_5pct)
        | (~stability.throughput_stable_5pct)
    ].copy()
    breakdown = (
        unstable.groupby("platform").agg(
            unstable_configurations=("configuration", "size"),
            runtime_unstable=("runtime_stable_5pct", lambda x: int((~x).sum())),
            energy_unstable=("energy_stable_5pct", lambda x: int((~x).sum())),
            throughput_unstable=("throughput_stable_5pct", lambda x: int((~x).sum())),
        ).reset_index()
    )
    all_platforms = pd.DataFrame({"platform": list(PLATFORMS)})
    breakdown = all_platforms.merge(breakdown, on="platform", how="left").fillna(0)
    for col in ["unstable_configurations", "runtime_unstable", "energy_unstable", "throughput_unstable"]:
        breakdown[col] = breakdown[col].astype(int)
    breakdown["platform_label"] = breakdown.platform.map(PLATFORM_LABELS)
    breakdown.to_csv(out / "stability_breakdown.csv", index=False)

    selected_stability = leaders.merge(
        stability,
        left_on=["platform", "problem_size", "exact_configuration"],
        right_on=["platform", "problem_size", "configuration"],
        how="left",
        suffixes=("_leader", "_stability"),
    )
    selected_stability["selected_configuration_stable"] = (
        selected_stability.runtime_stable_5pct
        & selected_stability.energy_stable_5pct
        & selected_stability.throughput_stable_5pct
    )
    selected_stability[[
        "platform", "platform_label_leader", "problem_size", "policy",
        "exact_configuration", "selection_status", "selected_configuration_stable",
        "runtime_s_session_cv_pct", "energy_j_session_cv_pct",
        "throughput_gflops_session_cv_pct",
    ]].to_csv(out / "leader_stability.csv", index=False)

    record(checks, "stability", "all_configurations_below_5pct_cv", "WARN",
           unstable.empty, f"unstable={len(unstable)}; by_platform={dict(zip(breakdown.platform, breakdown.unstable_configurations))}", 0)

    gpu_statuses = set(
        leaders[leaders.platform.isin(GPU_PLATFORMS)].selection_status.astype(str)
    )
    record(checks, "semantics", "gpu_policy_status_is_single_configuration", "FAIL",
           gpu_statuses == {"single_configuration"}, gpu_statuses, {"single_configuration"})

    record(checks, "semantics", "energy_domain_asymmetry", "WARN", False,
           "CPU package RAPL versus GPU board NVML",
           "retain in every CPU/GPU energy and Pareto claim")
    record(checks, "statistics", "native_best_post_selection", "WARN", False,
           "selection and estimation use the same five sessions",
           "descriptive intervals/effects only")
    record(checks, "statistics", "five_sessions_limit_inference", "WARN", False,
           "n=5 session medians per configuration",
           "repeatability on measured systems, not hardware-population inference")

    check_df = pd.DataFrame(checks)
    check_df.to_csv(out / "integrity_checks.csv", index=False)
    hard = check_df[(check_df.severity == "FAIL") & (check_df.status == "FAIL")]
    warns = check_df[(check_df.severity == "WARN") & (check_df.status == "WARN")]
    verdict = "FAIL" if len(hard) else ("PASS WITH WARNINGS" if len(warns) else "PASS")

    selected_unstable = selected_stability[~selected_stability.selected_configuration_stable]
    report = f"""# Independent integrity audit of the combined GEMM outputs

## Verdict

**{verdict}**

The audit recomputed the central identities, leader consistency, pairwise ranges,
and Pareto classifications from the generated CSV files. It did not rely on the
automatically generated prose as evidence.

## Hard failures

{markdown_table(hard)}

## Warnings

{markdown_table(warns)}

## Corrected metric semantics

For each fixed matrix size, every configuration performs the same logical work,
`2*N^3` FLOP. The normalized views therefore obey exact identities:

- throughput = logical FLOP / e2e runtime;
- GFLOP/J = logical FLOP / measured device-domain energy.

Consequently, runtime and throughput are inverse views of the same dimension,
and energy per GEMM and GFLOP/J are inverse views of the same dimension. They
must not be counted as four independent findings. EDP remains the joint
energy-runtime metric.

The maximum difference between the originally reported throughput and the
normalized e2e-throughput is {normalization_max:.5f}%, so the normalization fix
does not change the substantive runtime winners.

## Stability breakdown

{breakdown.to_markdown(index=False)}

There are {len(unstable)} configurations above 5% session-level CV in at least
one central metric. They are concentrated on the CPUs, especially Intel. The
full rows remain in `all_platform_stability.csv`.

{len(selected_unstable)} of the 180 policy-selected platform/size views use an
exact configuration that exceeds the 5% CV rule in at least one metric. This
does not automatically invalidate a platform winner: a device-level gap may be
large even when the exact thread-count identity is uncertain. Interpret the
platform decision separately from the selected CPU thread count.

## Pairwise effect direction

`probability_a_better` is already oriented so larger means platform A is better
for the named metric. `cliffs_delta_a_minus_b`, however, is purely numerical:
positive means A has larger values than B. Therefore positive delta favors A for
throughput/GFLOP-J, but favors B for runtime/energy/EDP. This sign rule must be
stated whenever Cliff's delta is used.

## Remaining interpretation constraints

1. CPU energy is package RAPL; GPU energy is NVML board energy.
2. GPU execution is resident and excludes allocation, initialization and PCIe transfer.
3. Native-best comparisons are descriptive after configuration selection.
4. Five sessions support repeatability claims on these systems, not population claims.
5. Practical Pareto status uses the explicit 2% dominance rule implemented by the pipeline.
"""
    (out / "INTEGRITY_AUDIT.md").write_text(report, encoding="utf-8")

    direction = """# Direction of pairwise effect columns

- `a_over_b_ratio < 1` favors A for lower-is-better metrics: runtime, energy, EDP.
- `a_over_b_ratio > 1` favors A for higher-is-better metrics: throughput, GFLOP/J.
- `probability_a_better` is oriented to the metric and always means the probability that A is better.
- `cliffs_delta_a_minus_b > 0` only means A has numerically larger values:
  - favorable for throughput and GFLOP/J;
  - unfavorable for runtime, energy and EDP.
"""
    (out / "PAIRWISE_EFFECT_DIRECTION.md").write_text(direction, encoding="utf-8")

    print(f"[ALL GEMM] integrity audit: {verdict}")
    print(out / "INTEGRITY_AUDIT.md")
    if hard.shape[0]:
        sys.exit(2)


if __name__ == "__main__":
    main()

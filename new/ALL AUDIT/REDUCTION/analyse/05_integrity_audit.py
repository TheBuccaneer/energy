#!/usr/bin/env python3
from __future__ import annotations

import numpy as np
import pandas as pd

from all_reduction_common import *


def record(rows, category, check, severity, passed, observed, expected):
    add_check(rows, category, check, severity, passed, observed, expected)


def main() -> None:
    out = results_dir(__file__)
    sessions = pd.read_csv(out / "unified_session_medians.csv")
    summary = pd.read_csv(out / "unified_configuration_summary.csv")
    leaders = pd.read_csv(out / "native_policy_leaders.csv")
    selected = pd.read_csv(out / "native_policy_session_medians.csv")
    pairwise = pd.read_csv(out / "pairwise_native_best_comparisons.csv")
    winners = pd.read_csv(out / "all_platform_metric_winners.csv")
    pareto = pd.read_csv(out / "all_configuration_pareto.csv")
    regret = pd.read_csv(out / "within_platform_exact_winner_regret.csv")
    checks = []

    record(checks, "coverage", "unified_session_rows", "FAIL", len(sessions) == 810, len(sessions), 810)
    record(checks, "coverage", "configuration_rows", "FAIL", len(summary) == 162, len(summary), 162)
    record(checks, "coverage", "leader_rows", "FAIL", len(leaders) == 180, len(leaders), 180)
    record(checks, "coverage", "selected_session_rows", "FAIL", len(selected) == 900, len(selected), 900)
    record(checks, "coverage", "pairwise_rows", "FAIL", len(pairwise) == 270, len(pairwise), 270)
    record(checks, "coverage", "winner_rows", "FAIL", len(winners) == 45, len(winners), 45)
    record(checks, "coverage", "exact_winner_regret_rows", "FAIL", len(regret) == 36, len(regret), 36)

    n = sessions.problem_size.to_numpy(float)
    bw_expected = (4.0 * n + 4.0) / sessions.runtime_s.to_numpy(float) / 1e9
    gbj_expected = (4.0 * n + 4.0) / sessions.energy_j.to_numpy(float) / 1e9
    edp_expected = sessions.runtime_s.to_numpy(float) * sessions.energy_j.to_numpy(float)
    throughput_expected = (n - 1.0) / sessions.runtime_s.to_numpy(float) / 1e9
    efficiency_expected = (n - 1.0) / sessions.energy_j.to_numpy(float) / 1e9
    identities = {
        "logical_bandwidth": np.max(np.abs(sessions.logical_bandwidth_gb_s - bw_expected)),
        "logical_gb_per_j": np.max(np.abs(sessions.logical_gb_per_j - gbj_expected)),
        "edp": np.max(np.abs(sessions.edp_j_s - edp_expected)),
        "throughput": np.max(np.abs(sessions.throughput_gflops - throughput_expected)),
        "efficiency": np.max(np.abs(sessions.efficiency_gflop_per_j - efficiency_expected)),
    }
    for name, error in identities.items():
        record(checks, "identity", name, "FAIL", float(error) < 1e-12, f"max_abs_error={error:.3e}", "<1e-12")

    # Inverse-view leader consistency.
    bad_runtime_bw = 0
    bad_energy_gbj = 0
    for (platform, size), group in leaders.groupby(["platform", "problem_size"]):
        r = group[group.policy == "runtime_opt"].iloc[0]
        b = group[group.policy == "bandwidth_opt"].iloc[0]
        e = group[group.policy == "energy_opt"].iloc[0]
        j = group[group.policy == "bytes_per_j_opt"].iloc[0]
        bad_runtime_bw += int(r.exact_configuration != b.exact_configuration)
        bad_energy_gbj += int(e.exact_configuration != j.exact_configuration)
    record(checks, "leaders", "runtime_equals_bandwidth_exact", "FAIL", bad_runtime_bw == 0, bad_runtime_bw, 0)
    record(checks, "leaders", "energy_equals_bytes_per_j_exact", "FAIL", bad_energy_gbj == 0, bad_energy_gbj, 0)

    # Pairwise ratio and classification identities.
    ratio_error = np.abs(pairwise.a_over_b_ratio - pairwise.a_median / pairwise.b_median)
    record(checks, "pairwise", "ratio_identity", "FAIL", float(ratio_error.max()) < 1e-12,
           f"max_abs_error={ratio_error.max():.3e}", "<1e-12")
    expected_classes = [
        classify_ratio(row.a_over_b_ratio, row.ratio_ci95_low, row.ratio_ci95_high,
                       row.platform_a, row.platform_b, bool(row.lower_is_better))
        for row in pairwise.itertuples()
    ]
    class_bad = int((pairwise.classification.astype(str).to_numpy() != np.asarray(expected_classes)).sum())
    record(checks, "pairwise", "classification_rule", "FAIL", class_bad == 0, class_bad, 0)

    # Independent Pareto recomputation.
    strict_bad = practical_bad = 0
    for size, group in pareto.groupby("problem_size"):
        expected_strict = strict_pareto(group)
        expected_practical = practical_pareto(group)
        observed_strict = group.strict_pareto.astype(str).str.lower().isin({"true", "1"})
        observed_practical = group.practical_pareto_2pct.astype(str).str.lower().isin({"true", "1"})
        strict_bad += int((observed_strict.to_numpy() != expected_strict.to_numpy()).sum())
        practical_bad += int((observed_practical.to_numpy() != expected_practical.to_numpy()).sum())
    record(checks, "pareto", "strict_recomputed", "FAIL", strict_bad == 0, strict_bad, 0)
    record(checks, "pareto", "practical_recomputed", "FAIL", practical_bad == 0, practical_bad, 0)

    energy_penalty_expected = 100.0 * (regret.runtime_opt_over_energy_opt_energy_ratio - 1.0)
    runtime_gain_expected = 100.0 * (1.0 - regret.runtime_opt_over_energy_opt_runtime_ratio)
    record(checks, "exact_winner_regret", "energy_penalty_identity", "FAIL",
           float(np.max(np.abs(regret.runtime_opt_energy_penalty_pct - energy_penalty_expected))) < 1e-12,
           f"max_abs_error={float(np.max(np.abs(regret.runtime_opt_energy_penalty_pct - energy_penalty_expected))):.3e}", "<1e-12")
    record(checks, "exact_winner_regret", "runtime_gain_identity", "FAIL",
           float(np.max(np.abs(regret.runtime_opt_runtime_gain_pct - runtime_gain_expected))) < 1e-12,
           f"max_abs_error={float(np.max(np.abs(regret.runtime_opt_runtime_gain_pct - runtime_gain_expected))):.3e}", "<1e-12")
    record(checks, "statistics", "exact_winner_regret_is_post_selection", "WARN", False,
           "same five sessions used for selection and interval", "descriptive only")

    record(checks, "semantics", "runtime_bandwidth_not_independent", "WARN", False,
           "inverse views", "do not count as independent evidence")
    record(checks, "semantics", "energy_gbj_not_independent", "WARN", False,
           "inverse views", "do not count as independent evidence")
    record(checks, "semantics", "logical_bandwidth_scope", "WARN", False,
           "(4*N+4)/runtime", "not measured physical traffic")
    record(checks, "semantics", "energy_domain_asymmetry", "WARN", False,
           "CPU package vs GPU board", "must remain explicit")
    record(checks, "semantics", "resident_gpu_scope", "WARN", False,
           "PCIe excluded", "must remain explicit")

    frame = pd.DataFrame(checks)
    frame.to_csv(out / "integrity_checks.csv", index=False)
    hard = frame[(frame.severity == "FAIL") & (frame.status == "FAIL")]
    warnings = frame[(frame.severity == "WARN") & (frame.status == "WARN")]
    verdict = "FAIL" if len(hard) else ("PASS WITH WARNINGS" if len(warnings) else "PASS")
    report = f"""# Independent integrity audit — REDUCTION

## Verdict

**{verdict}**

The audit recomputed central metric identities, inverse-view leader consistency,
pairwise classifications, and Pareto status from generated CSVs rather than prose.

## Hard failures

{markdown_table(hard)}

## Warnings

{markdown_table(warnings)}

## All checks

{markdown_table(frame, 300)}
"""
    (out / "INTEGRITY_AUDIT.md").write_text(report, encoding="utf-8")
    print(f"[ALL REDUCTION] integrity audit: {verdict}")
    if len(hard):
        raise SystemExit(2)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from all_strided_common import *

COMPARISON_METRICS = {
    "runtime_s": True,
    "energy_j": True,
    "edp_j_s": True,
    "power_w": True,
}


def classify_ratio(ratio: float, lo: float, hi: float) -> str:
    low = 1.0 / (1.0 + PRACTICAL_TOLERANCE)
    high = 1.0 + PRACTICAL_TOLERANCE
    if np.isfinite(hi) and hi < low:
        return "clear_strided_lower"
    if np.isfinite(lo) and lo > high:
        return "clear_strided_higher"
    if low <= ratio <= high:
        return "practically_equivalent_or_uncertain"
    return "uncertain_strided_lower" if ratio < 1 else "uncertain_strided_higher"


def parse_leader_set(value: object) -> frozenset[str]:
    """Normalize a comma-separated leader list to an order-independent set."""
    if pd.isna(value):
        return frozenset()
    return frozenset(
        item.strip()
        for item in str(value).split(",")
        if item.strip()
    )


def select_exact_best(summary: pd.DataFrame, platform: str, size: int, metric: str, lower: bool) -> pd.Series:
    group = summary[(summary.platform == platform) & (summary.problem_size == size)]
    col = f"{metric}_median"
    return group.loc[group[col].idxmin() if lower else group[col].idxmax()]


def compare_values(vs: np.ndarray, vd: np.ndarray, seed_parts: tuple[object, ...]) -> dict:
    strided_point = float(np.median(vs))
    dense_point = float(np.median(vd))
    ratio = strided_point / dense_point
    lo, hi = bootstrap_ratio_ci(vs, vd, seed_parts=seed_parts)
    return {
        "strided_median": strided_point,
        "dense_median": dense_point,
        "strided_over_dense_ratio": ratio,
        "ci95_low": lo,
        "ci95_high": hi,
        "classification": classify_ratio(ratio, lo, hi),
        "probability_strided_lower": probability_a_better(vs, vd, lower=True),
        "cliffs_delta_strided_minus_dense": cliffs_delta(vs, vd),
    }


def empty_outputs(out: Path, reason: str) -> None:
    for name in [
        "dense_vs_strided_configuration_matched.csv",
        "dense_vs_strided_native_best.csv",
        "layout_induced_placement_changes.csv",
        "layout_induced_configuration_changes.csv",
    ]:
        pd.DataFrame().to_csv(out / name, index=False)
    (out / "DENSE_VS_STRIDED_REPORT.md").write_text(
        "# Dense GEMM versus STRIDED_GEMM\n\n"
        f"Comparison skipped: {reason}\n",
        encoding="utf-8",
    )


def main() -> None:
    out = results_dir(__file__)
    strided_session_path = out / "unified_session_medians.csv"
    strided_summary_path = out / "unified_configuration_summary.csv"
    if not strided_session_path.is_file() or not strided_summary_path.is_file():
        raise SystemExit("Run 02_build_unified_stats.py first.")

    root = project_root(__file__)
    dense_out = root / "ALL AUDIT" / "GEMM" / "results"
    dense_session_path = dense_out / "unified_session_medians.csv"
    dense_summary_path = dense_out / "unified_configuration_summary.csv"
    if not dense_session_path.is_file() or not dense_summary_path.is_file():
        empty_outputs(out, f"dense GEMM unified outputs missing under {dense_out}")
        print(f"[ALL STRIDED_GEMM] dense comparison skipped: missing {dense_out}")
        return

    dense_sessions = pd.read_csv(dense_session_path)
    dense_summary = pd.read_csv(dense_summary_path)
    strided_sessions = pd.read_csv(strided_session_path)
    strided_summary = pd.read_csv(strided_summary_path)

    matched_rows: list[dict] = []
    for platform in PLATFORMS:
        dense_p = dense_sessions[dense_sessions.platform == platform]
        strided_p = strided_sessions[strided_sessions.platform == platform]
        common = sorted(set(dense_p.configuration.astype(str)) & set(strided_p.configuration.astype(str)))
        for size in SIZES:
            for configuration in common:
                d = dense_p[(dense_p.problem_size == size) & (dense_p.configuration.astype(str) == configuration)]
                s = strided_p[(strided_p.problem_size == size) & (strided_p.configuration.astype(str) == configuration)]
                if d.empty or s.empty:
                    continue
                for metric in COMPARISON_METRICS:
                    vd = pd.to_numeric(d[metric], errors="coerce").dropna().to_numpy(float)
                    vs = pd.to_numeric(s[metric], errors="coerce").dropna().to_numpy(float)
                    if not len(vd) or not len(vs):
                        continue
                    comp = compare_values(vs, vd, ("layout_matched", platform, size, configuration, metric))
                    matched_rows.append({
                        "platform": platform,
                        "platform_label": PLATFORM_LABELS[platform],
                        "problem_size": size,
                        "configuration": configuration,
                        "metric": metric,
                        "analysis_type": "configuration_matched_independent_sessions",
                        **comp,
                    })
    matched = pd.DataFrame(matched_rows)
    matched.to_csv(out / "dense_vs_strided_configuration_matched.csv", index=False)

    native_rows: list[dict] = []
    config_change_rows: list[dict] = []
    for platform in PLATFORMS:
        for size in SIZES:
            for metric, lower in COMPARISON_METRICS.items():
                dense_best = select_exact_best(dense_summary, platform, size, metric, lower)
                strided_best = select_exact_best(strided_summary, platform, size, metric, lower)
                d = dense_sessions[
                    (dense_sessions.platform == platform)
                    & (dense_sessions.problem_size == size)
                    & (dense_sessions.configuration.astype(str) == str(dense_best.configuration))
                ]
                s = strided_sessions[
                    (strided_sessions.platform == platform)
                    & (strided_sessions.problem_size == size)
                    & (strided_sessions.configuration.astype(str) == str(strided_best.configuration))
                ]
                vd = pd.to_numeric(d[metric], errors="coerce").dropna().to_numpy(float)
                vs = pd.to_numeric(s[metric], errors="coerce").dropna().to_numpy(float)
                comp = compare_values(vs, vd, ("layout_native", platform, size, metric, dense_best.configuration, strided_best.configuration))
                native_rows.append({
                    "platform": platform,
                    "platform_label": PLATFORM_LABELS[platform],
                    "problem_size": size,
                    "metric": metric,
                    "dense_best_configuration": dense_best.configuration,
                    "strided_best_configuration": strided_best.configuration,
                    "configuration_changed": str(dense_best.configuration) != str(strided_best.configuration),
                    "analysis_type": "descriptive_native_best_post_selection_independent_sessions",
                    **comp,
                })
                config_change_rows.append({
                    "platform": platform,
                    "problem_size": size,
                    "metric": metric,
                    "dense_best_configuration": dense_best.configuration,
                    "strided_best_configuration": strided_best.configuration,
                    "configuration_changed": str(dense_best.configuration) != str(strided_best.configuration),
                })
    native = pd.DataFrame(native_rows)
    native.to_csv(out / "dense_vs_strided_native_best.csv", index=False)
    pd.DataFrame(config_change_rows).to_csv(out / "layout_induced_configuration_changes.csv", index=False)

    dense_winners_path = dense_out / "all_platform_metric_winners.csv"
    strided_winners_path = out / "all_platform_metric_winners.csv"
    placement_rows = []
    if dense_winners_path.is_file() and strided_winners_path.is_file():
        dw = pd.read_csv(dense_winners_path)
        sw = pd.read_csv(strided_winners_path)
        for metric in ["runtime_s", "energy_j", "edp_j_s"]:
            for size in SIZES:
                d = dw[(dw.metric == metric) & (dw.problem_size == size)].iloc[0]
                s = sw[(sw.metric == metric) & (sw.problem_size == size)].iloc[0]
                point_estimate_winner_changed = (
                    str(d.exact_winner) != str(s.exact_winner)
                )
                leader_set_changed = (
                    parse_leader_set(d.leader_platforms)
                    != parse_leader_set(s.leader_platforms)
                )
                decisive_placement_changed = (
                    str(d.selection_status) == "clear_leader"
                    and str(s.selection_status) == "clear_leader"
                    and point_estimate_winner_changed
                )
                placement_rows.append({
                    "problem_size": size,
                    "metric": metric,
                    "dense_exact_winner": d.exact_winner,
                    "dense_leader_platforms": d.leader_platforms,
                    "dense_selection_status": d.selection_status,
                    "strided_exact_winner": s.exact_winner,
                    "strided_leader_platforms": s.leader_platforms,
                    "strided_selection_status": s.selection_status,
                    "point_estimate_winner_changed": point_estimate_winner_changed,
                    # Backward-compatible alias. New reports use the explicit name above.
                    "exact_winner_changed": point_estimate_winner_changed,
                    "leader_set_changed": leader_set_changed,
                    "decisive_placement_changed": decisive_placement_changed,
                })
    placement = pd.DataFrame(placement_rows)
    placement.to_csv(out / "layout_induced_placement_changes.csv", index=False)

    figdir = out / "figures"
    for metric, ylabel, filename in [
        ("runtime_s", "STRIDED / dense runtime", "dense_vs_strided_runtime_ratio.png"),
        ("energy_j", "STRIDED / dense measured energy", "dense_vs_strided_energy_ratio.png"),
        ("edp_j_s", "STRIDED / dense EDP", "dense_vs_strided_edp_ratio.png"),
    ]:
        subset = native[native.metric == metric]
        plt.figure(figsize=(8.5, 5.2))
        for platform, group in subset.groupby("platform"):
            group = group.sort_values("problem_size")
            plt.plot(group.problem_size, group.strided_over_dense_ratio, marker="o", label=PLATFORM_LABELS[platform])
        plt.axhline(1.0, linestyle="--")
        plt.xscale("log", base=2)
        plt.xlabel("Matrix size N")
        plt.ylabel(ylabel)
        plt.title(f"Dense versus STRIDED_GEMM: {metric}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(figdir / filename, dpi=180)
        plt.close()

    clear_penalties = native[native.classification == "clear_strided_higher"]
    clear_improvements = native[native.classification == "clear_strided_lower"]
    point_estimate_changes = (
        placement[placement.point_estimate_winner_changed]
        if not placement.empty else placement
    )
    leader_set_changes = (
        placement[placement.leader_set_changed]
        if not placement.empty else placement
    )
    decisive_changes = (
        placement[placement.decisive_placement_changed]
        if not placement.empty else placement
    )
    report = f"""# Dense GEMM versus STRIDED_GEMM

Ratios are `STRIDED_GEMM / dense GEMM`. Values above one indicate that the padded layout requires more of the reported metric. Dense and Strided campaigns were executed in separate sessions, therefore ratio intervals use independent bootstrap resampling rather than artificial session pairing.

## Statistical interpretation

- Configuration-matched comparisons hold platform, N, and thread configuration fixed.
- Native-best comparisons allow each workload to choose its own best configuration and are explicitly descriptive post-selection analyses.
- Runtime and energy are the two primitive decision axes. EDP is their composite.
- Throughput and GFLOP/J may be displayed elsewhere but are not counted as independent evidence.
- `logical_bytes=12N²` and allocated footprint `24N²` are semantic/layout quantities, not measured physical memory traffic.

## Summary counts

- Configuration-matched comparisons: {len(matched)}
- Native-best comparisons: {len(native)}
- Clear Strided penalties: {len(clear_penalties)}
- Clear Strided improvements: {len(clear_improvements)}
- Point-estimate winner changes: {len(point_estimate_changes)}
- Leader-set changes: {len(leader_set_changes)}
- Decisive cross-platform placement changes: {len(decisive_changes)}

A point-estimate winner change is not counted as a decisive placement change unless both Dense and Strided have `selection_status=clear_leader`. Leader sets are compared as unordered sets, so `3090,5060ti` and `5060ti,3090` are identical.

## Native-best layout ratios

{markdown_table(native, 160)}

## Point-estimate winner changes

{markdown_table(point_estimate_changes, 100)}

## Decisive cross-platform placement changes

{markdown_table(decisive_changes, 100)}

## All cross-platform placement rows

{markdown_table(placement, 100)}

## Interpretation guard

A `clear_strided_lower` result is not automatically an error. BLAS and cuBLAS may pack inputs, choose different kernels, or benefit from alignment changes. Unexpected directions should be audited against raw sessions and telemetry, not rejected by assumption.

A changed point-estimate winner inside the same `tie_or_uncertain` leader set is descriptive only. It is not evidence of a decisive cross-platform placement change.
"""
    (out / "DENSE_VS_STRIDED_REPORT.md").write_text(report, encoding="utf-8")
    print(f"[ALL STRIDED_GEMM] dense-vs-strided comparison written to {out}")


if __name__ == "__main__":
    main()

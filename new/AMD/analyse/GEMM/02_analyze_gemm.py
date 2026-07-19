#!/usr/bin/env python3
"""Produce session-aware scientific summaries and figures for one CPU."""
from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gemm_common import (
    PRACTICAL_TOLERANCE, SIZES, THREADS, add_derived, campaign_summary, context,
    load_campaign, markdown_table, metric_leaders, parse_args, practical_pareto_mask,
    robust_outlier_mask, session_medians, write_text,
)


def strict_pareto_mask(group: pd.DataFrame) -> pd.Series:
    energy = group["package_energy_per_op_j_median"].to_numpy(float)
    runtime = group["runtime_per_op_s_median"].to_numpy(float)
    keep = np.ones(len(group), dtype=bool)
    for i in range(len(group)):
        dominated = (
            (energy <= energy[i]) & (runtime <= runtime[i]) &
            ((energy < energy[i]) | (runtime < runtime[i]))
        )
        dominated[i] = False
        if dominated.any():
            keep[i] = False
    return pd.Series(keep, index=group.index)


def save_plots(summary: pd.DataFrame, sessions: pd.DataFrame, result_dir, platform: str) -> None:
    figures = result_dir / "figures"

    large = summary[summary["problem_size"] >= 1024]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for n, group in large.groupby("problem_size"):
        ax.plot(group["num_threads"], group["gflops_per_s_median"], marker="o", label=f"N={int(n)}")
    ax.set_xlabel("OpenBLAS threads")
    ax.set_ylabel("GFLOP/s (median of session medians)")
    ax.set_title(f"{platform} GEMM throughput scaling")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "throughput_scaling.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for n, group in large.groupby("problem_size"):
        ax.plot(group["num_threads"], group["package_energy_per_op_j_median"], marker="o", label=f"N={int(n)}")
    ax.set_xlabel("OpenBLAS threads")
    ax.set_ylabel("Package energy per GEMM [J]")
    ax.set_yscale("log")
    ax.set_title(f"{platform} GEMM package energy")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "package_energy_scaling.png", dpi=180)
    plt.close(fig)

    pivot = summary.pivot(index="problem_size", columns="num_threads", values="runtime_per_op_s_run_robust_cv")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    image = ax.imshow(100 * pivot.to_numpy(), aspect="auto", origin="lower")
    ax.set_xticks(range(len(pivot.columns)), [str(int(x)) for x in pivot.columns])
    ax.set_yticks(range(len(pivot.index)), [str(int(x)) for x in pivot.index])
    ax.set_xlabel("Threads")
    ax.set_ylabel("N")
    ax.set_title(f"{platform} run-level robust CV of runtime [%]")
    fig.colorbar(image, ax=ax, label="Robust CV [%]")
    fig.tight_layout()
    fig.savefig(figures / "runtime_stability_heatmap.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for session, group in sessions.groupby("session_number"):
        ordered = group.sort_values(["problem_size", "num_threads"])
        ax.plot(range(len(ordered)), ordered["temp_c"], label=f"session {int(session)}", alpha=0.8)
    ax.set_xlabel("Configuration order after sorting by N and threads")
    ax.set_ylabel("Median temperature [C]")
    ax.set_title(f"{platform} session-level thermal profile")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "session_temperature_profile.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args("Analyze the local CPU GEMM campaign")
    _, platform, run_dir, result_dir = context(__file__)
    campaign = load_campaign(run_dir, platform, args.campaign)
    df = add_derived(campaign.data)

    sessions = session_medians(df)
    sessions.to_csv(result_dir / "session_configuration_medians.csv", index=False)
    summary = campaign_summary(df)

    baseline = summary[summary["num_threads"] == 1][
        ["problem_size", "runtime_per_op_s_median", "package_energy_per_op_j_median"]
    ].rename(columns={
        "runtime_per_op_s_median": "runtime_1thread_s",
        "package_energy_per_op_j_median": "energy_1thread_j",
    })
    summary = summary.merge(baseline, on="problem_size", how="left")
    summary["speedup_vs_1thread"] = summary["runtime_1thread_s"] / summary["runtime_per_op_s_median"]
    summary["scaling_efficiency"] = summary["speedup_vs_1thread"] / summary["num_threads"]
    summary["energy_change_vs_1thread_pct"] = 100 * (
        summary["package_energy_per_op_j_median"] / summary["energy_1thread_j"] - 1
    )
    summary["stable_runtime"] = summary["runtime_per_op_s_run_robust_cv"] <= 0.05
    summary["stable_energy"] = summary["package_energy_per_op_j_run_robust_cv"] <= 0.08
    summary["pareto_optimal_strict"] = False
    summary["pareto_optimal_practical_2pct"] = False
    for _, group in summary.groupby("problem_size"):
        summary.loc[group.index, "pareto_optimal_strict"] = strict_pareto_mask(group).to_numpy()
        summary.loc[group.index, "pareto_optimal_practical_2pct"] = practical_pareto_mask(group).to_numpy()
    summary.to_csv(result_dir / "configuration_summary.csv", index=False)

    session_overview = df.groupby("session_number", as_index=False).agg(
        rows=("problem_size", "size"),
        median_runtime_s=("e2e_time_s", "median"),
        median_package_power_w=("package_avg_power_w", "median"),
        median_temp_c=("temp_c", "median"),
        max_temp_c=("temp_c", "max"),
        median_clock_before_mhz=("clock_before_mhz", "median"),
        median_clock_after_mhz=("clock_after_mhz", "median"),
        checksum_failures=("checksum_bool", lambda x: int((~x).sum())),
    )
    session_overview.to_csv(result_dir / "session_overview.csv", index=False)

    outlier_parts = []
    for (n, threads), group in df.groupby(["problem_size", "num_threads"]):
        mask_time = robust_outlier_mask(group["runtime_per_op_s"])
        mask_energy = robust_outlier_mask(group["package_energy_per_op_j"])
        chosen = group[mask_time | mask_energy].copy()
        if not chosen.empty:
            chosen["runtime_outlier"] = mask_time.loc[chosen.index].to_numpy()
            chosen["energy_outlier"] = mask_energy.loc[chosen.index].to_numpy()
            outlier_parts.append(chosen)
    outliers = pd.concat(outlier_parts, ignore_index=True) if outlier_parts else pd.DataFrame()
    keep_cols = [
        "source_file", "session_number", "sequence_index", "repetition",
        "problem_size", "num_threads", "runtime_per_op_s",
        "package_energy_per_op_j", "package_avg_power_w", "temp_c",
        "runtime_outlier", "energy_outlier",
    ]
    if not outliers.empty:
        outliers[keep_cols].to_csv(result_dir / "robust_outliers.csv", index=False)
    else:
        pd.DataFrame(columns=keep_cols).to_csv(result_dir / "robust_outliers.csv", index=False)

    best_energy, energy_candidates = metric_leaders(summary, "package_energy_per_op_j_median")
    best_runtime, runtime_candidates = metric_leaders(summary, "runtime_per_op_s_median")
    best_edp, edp_candidates = metric_leaders(summary, "edp_package_j_s_median")
    best_energy.to_csv(result_dir / "best_energy_by_size.csv", index=False)
    best_runtime.to_csv(result_dir / "best_runtime_by_size.csv", index=False)
    best_edp.to_csv(result_dir / "best_edp_by_size.csv", index=False)
    pd.concat([energy_candidates, runtime_candidates, edp_candidates], ignore_index=True).to_csv(
        result_dir / "near_optimal_candidates.csv", index=False
    )

    save_plots(summary, sessions, result_dir, platform)

    unstable_runtime = summary[~summary["stable_runtime"]]
    unstable_energy = summary[~summary["stable_energy"]]
    high_temp = float(df["temp_c"].max())
    outlier_share = len(outliers) / len(df) if len(df) else math.nan
    report = f"""# {platform} GEMM scientific analysis

## Campaign and statistical unit

- Campaign: `{campaign.stamp}`
- Raw measurements: {len(df)}
- Sessions: {len(campaign.files)}
- Repetitions per configuration/session: 10
- Primary inferential unit: **session median**, not each raw repetition
- Confidence intervals: non-parametric bootstrap of five session medians
- Primary energy metric: `device_energy_j / batches` = package energy per GEMM

The ten adjacent repetitions within one configuration are treated as repeated
measurements under one session state. They are useful for noise and outlier
analysis, but are not counted as 50 independent experimental sessions.

## Quality indicators

- Maximum temperature: {high_temp:.1f} C
- Robust outlier share: {100*outlier_share:.2f}%
- Runtime-unstable configurations (robust CV >5%): {len(unstable_runtime)} / {len(summary)}
- Energy-unstable configurations (robust CV >8%): {len(unstable_energy)} / {len(summary)}

## Package-energy leaders and tie status

{markdown_table(best_energy[["problem_size", "exact_min_threads", "leader_threads", "selection_status", "gap_to_second_pct", "package_energy_per_op_j_median", "runtime_per_op_s_median"]], 20)}

## Runtime leaders and tie status

{markdown_table(best_runtime[["problem_size", "exact_min_threads", "leader_threads", "selection_status", "gap_to_second_pct", "runtime_per_op_s_median", "package_energy_per_op_j_median"]], 20)}

## Interpretation constraints

1. `energy_per_op_j` from the raw CSV is not the cross-platform primary metric,
   because it is based on total RAPL energy and Intel may include a separate DRAM
   domain while AMD may not expose one. The analysis recomputes package-only energy.
2. Exact minima are not automatically called unique winners. A clear leader requires
   both a median advantage greater than {100*PRACTICAL_TOLERANCE:.0f}% and a 95%
   bootstrap interval separated from every competitor. Otherwise the result is
   labeled `tie_or_uncertain`, and all practical/CI-overlapping candidates are listed.
3. Strict and practical-2% Pareto frontiers are both retained.
4. AMD 32/64-thread results characterize native capability. Intel-vs-AMD fairness
   at fixed software parallelism is handled separately by `03_compare_gemm.py`.
5. The confidence intervals quantify between-session repeatability for this
   measurement campaign, not population-wide uncertainty across machines.
"""
    write_text(result_dir / "scientific_report.md", report)
    print(f"[{platform}] scientific analysis written to {result_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Session-aware scientific analysis for one CPU STRIDED_GEMM campaign."""
from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from strided_gemm_common import (
    PRACTICAL_TOLERANCE, add_derived, campaign_summary, context, load_campaign,
    markdown_table, metric_leaders, parse_args, practical_pareto_mask,
    robust_outlier_mask, session_medians, strict_pareto_mask, write_text,
)


def save_plots(summary: pd.DataFrame, sessions: pd.DataFrame, result_dir, platform: str) -> None:
    figures = result_dir / "figures"
    large = summary[summary["problem_size"] >= 1024]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for n, group in large.groupby("problem_size"):
        ax.plot(group["num_threads"], group["throughput_gflops_exact_median"], marker="o", label=f"N={int(n)}")
    ax.set_xlabel("OpenBLAS threads")
    ax.set_ylabel("GFLOP/s")
    ax.set_title(f"{platform} STRIDED_GEMM throughput scaling")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "throughput_scaling.png", dpi=180)
    plt.close(fig)

    for metric, ylabel, filename, title in [
        ("package_energy_per_op_j_median", "Package energy per GEMM [J]", "package_energy_scaling.png", "primary energy"),
        ("total_energy_per_op_j_median", "Package + DRAM energy per GEMM [J]", "total_energy_scaling.png", "optional DRAM sensitivity"),
    ]:
        fig, ax = plt.subplots(figsize=(9, 5.5))
        for n, group in large.groupby("problem_size"):
            ax.plot(group["num_threads"], group[metric], marker="o", label=f"N={int(n)}")
        ax.set_xlabel("OpenBLAS threads")
        ax.set_ylabel(ylabel)
        ax.set_yscale("log")
        ax.set_title(f"{platform} STRIDED_GEMM {title}")
        ax.grid(True, alpha=0.3)
        ax.legend(ncol=2, fontsize=8)
        fig.tight_layout()
        fig.savefig(figures / filename, dpi=180)
        plt.close(fig)

    for metric, filename, label in [
        ("runtime_per_op_s_run_robust_cv", "runtime_stability_heatmap.png", "Runtime robust CV [%]"),
        ("package_energy_per_op_j_run_robust_cv", "energy_stability_heatmap.png", "Package-energy robust CV [%]"),
    ]:
        pivot = summary.pivot(index="problem_size", columns="num_threads", values=metric)
        fig, ax = plt.subplots(figsize=(9, 5.5))
        image = ax.imshow(100 * pivot.to_numpy(), aspect="auto", origin="lower")
        ax.set_xticks(range(len(pivot.columns)), [str(int(x)) for x in pivot.columns])
        ax.set_yticks(range(len(pivot.index)), [str(int(x)) for x in pivot.index])
        ax.set_xlabel("Threads")
        ax.set_ylabel("N")
        ax.set_title(f"{platform} STRIDED_GEMM {label}")
        fig.colorbar(image, ax=ax, label=label)
        fig.tight_layout()
        fig.savefig(figures / filename, dpi=180)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for session, group in sessions.groupby("session_number"):
        ordered = group.sort_values(["problem_size", "num_threads"])
        ax.plot(range(len(ordered)), ordered["temp_c"], label=f"session {int(session)}", alpha=0.8)
    ax.set_xlabel("Configuration order after sorting by N and threads")
    ax.set_ylabel("Median temperature [C]")
    ax.set_title(f"{platform} STRIDED_GEMM session thermal profile")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "session_temperature_profile.png", dpi=180)
    plt.close(fig)

    for n, group in summary.groupby("problem_size"):
        fig, ax = plt.subplots(figsize=(6.5, 5))
        ax.scatter(group["runtime_per_op_s_median"], group["package_energy_per_op_j_median"])
        for _, row in group.iterrows():
            ax.annotate(str(int(row["num_threads"])),
                        (row["runtime_per_op_s_median"], row["package_energy_per_op_j_median"]),
                        xytext=(4, 4), textcoords="offset points", fontsize=8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Runtime per GEMM [s]")
        ax.set_ylabel("Package energy per GEMM [J]")
        ax.set_title(f"{platform} STRIDED_GEMM Pareto map, N={int(n)}")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(figures / f"pareto_N{int(n)}.png", dpi=180)
        plt.close(fig)


def main() -> None:
    args = parse_args("Analyze a CPU STRIDED_GEMM campaign")
    _, platform, run_dir, result_dir = context(__file__)
    checks_path = result_dir / "validation_checks.csv"
    if not checks_path.is_file():
        raise SystemExit("Run 01_validate_strided_gemm.py first.")
    checks = pd.read_csv(checks_path)
    if ((checks["severity"] == "FAIL") & (checks["status"] == "FAIL")).any():
        raise SystemExit("Validation has hard failures; analysis aborted.")

    campaign = load_campaign(run_dir, platform, args.campaign)
    df = add_derived(campaign.data)
    sessions = session_medians(df)
    sessions.to_csv(result_dir / "session_configuration_medians.csv", index=False)
    summary = campaign_summary(df)

    baseline = summary[summary["num_threads"] == 1][[
        "problem_size", "runtime_per_op_s_median", "total_energy_per_op_j_median",
        "package_energy_per_op_j_median",
    ]].rename(columns={
        "runtime_per_op_s_median": "runtime_1thread_s",
        "total_energy_per_op_j_median": "total_energy_1thread_j",
        "package_energy_per_op_j_median": "package_energy_1thread_j",
    })
    summary = summary.merge(baseline, on="problem_size", how="left")
    summary["speedup_vs_1thread"] = summary["runtime_1thread_s"] / summary["runtime_per_op_s_median"]
    summary["scaling_efficiency"] = summary["speedup_vs_1thread"] / summary["num_threads"]
    summary["total_energy_change_vs_1thread_pct"] = 100 * (
        summary["total_energy_per_op_j_median"] / summary["total_energy_1thread_j"] - 1
    )
    summary["package_energy_change_vs_1thread_pct"] = 100 * (
        summary["package_energy_per_op_j_median"] / summary["package_energy_1thread_j"] - 1
    )
    summary["stable_runtime_5pct"] = summary["runtime_per_op_s_between_session_cv"] <= 0.05
    summary["stable_total_energy_5pct"] = summary["total_energy_per_op_j_between_session_cv"] <= 0.05
    summary["stable_package_energy_5pct"] = summary["package_energy_per_op_j_between_session_cv"] <= 0.05
    summary["pareto_optimal_strict_total"] = False
    summary["pareto_optimal_practical_2pct_total"] = False
    summary["pareto_optimal_strict_package"] = False
    summary["pareto_optimal_practical_2pct_package"] = False
    for _, group in summary.groupby("problem_size"):
        summary.loc[group.index, "pareto_optimal_strict_total"] = strict_pareto_mask(group).to_numpy()
        summary.loc[group.index, "pareto_optimal_practical_2pct_total"] = practical_pareto_mask(group).to_numpy()
        summary.loc[group.index, "pareto_optimal_strict_package"] = strict_pareto_mask(
            group, energy_col="package_energy_per_op_j_median").to_numpy()
        summary.loc[group.index, "pareto_optimal_practical_2pct_package"] = practical_pareto_mask(
            group, energy_col="package_energy_per_op_j_median").to_numpy()
    summary.to_csv(result_dir / "configuration_summary.csv", index=False)

    session_overview = df.groupby("session_number", as_index=False).agg(
        rows=("problem_size", "size"),
        median_runtime_window_s=("e2e_time_s", "median"),
        median_total_power_w=("total_avg_power_w", "median"),
        median_package_power_w=("package_avg_power_w", "median"),
        median_dram_power_w=("dram_avg_power_w", "median"),
        median_temp_c=("temp_c", "median"),
        max_temp_c=("temp_c", "max"),
        median_clock_before_mhz=("clock_before_mhz", "median"),
        median_clock_after_mhz=("clock_after_mhz", "median"),
        checksum_failures=("checksum_bool", lambda x: int((~x).sum())),
    )
    session_overview.to_csv(result_dir / "session_overview.csv", index=False)

    outlier_parts = []
    for (session, n, threads), group in df.groupby(["session_number", "problem_size", "num_threads"]):
        masks = {
            "runtime_outlier": robust_outlier_mask(group["runtime_per_op_s"]),
            "total_energy_outlier": robust_outlier_mask(group["total_energy_per_op_j"]),
            "package_energy_outlier": robust_outlier_mask(group["package_energy_per_op_j"]),
        }
        any_mask = masks["runtime_outlier"] | masks["total_energy_outlier"] | masks["package_energy_outlier"]
        if any_mask.any():
            part = group.loc[any_mask].copy()
            for name, mask in masks.items():
                part[name] = mask.loc[part.index].to_numpy()
            outlier_parts.append(part)
    outliers = pd.concat(outlier_parts, ignore_index=True) if outlier_parts else pd.DataFrame()
    out_cols = [
        "source_file", "session_number", "sequence_index", "repetition", "problem_size",
        "num_threads", "runtime_per_op_s", "total_energy_per_op_j",
        "package_energy_per_op_j", "dram_energy_per_op_j", "total_avg_power_w", "temp_c",
        "runtime_outlier", "total_energy_outlier", "package_energy_outlier",
    ]
    if outliers.empty:
        pd.DataFrame(columns=out_cols).to_csv(result_dir / "robust_outliers.csv", index=False)
    else:
        outliers[out_cols].to_csv(result_dir / "robust_outliers.csv", index=False)

    leader_specs = [
        ("total_energy_per_op_j_median", "best_total_energy_by_size.csv"),
        ("package_energy_per_op_j_median", "best_package_energy_by_size.csv"),
        ("runtime_per_op_s_median", "best_runtime_by_size.csv"),
        ("edp_total_j_s_median", "best_total_edp_by_size.csv"),
        ("edp_package_j_s_median", "best_package_edp_by_size.csv"),
        ("throughput_gflops_exact_median", "best_throughput_by_size.csv", "max"),
        ("total_efficiency_gflop_per_j_median", "best_total_efficiency_by_size.csv", "max"),
    ]
    leaders = {}
    candidates = []
    for spec in leader_specs:
        metric, filename = spec[0], spec[1]
        objective = spec[2] if len(spec) == 3 else "min"
        leader, cand = metric_leaders(summary, metric, objective=objective)
        leader.to_csv(result_dir / filename, index=False)
        leaders[metric] = leader
        candidates.append(cand)
    pd.concat(candidates, ignore_index=True).to_csv(result_dir / "near_optimal_candidates.csv", index=False)

    energy = leaders["package_energy_per_op_j_median"]
    runtime = leaders["runtime_per_op_s_median"]
    total = leaders["total_energy_per_op_j_median"]
    trade_rows = []
    for n in sorted(summary["problem_size"].unique()):
        e = energy[energy["problem_size"] == n].iloc[0]
        r = runtime[runtime["problem_size"] == n].iloc[0]
        total_row_leader = total[total["problem_size"] == n].iloc[0]
        e_set = set(str(e["leader_threads"]).split(","))
        r_set = set(str(r["leader_threads"]).split(","))
        overlap = sorted(e_set & r_set, key=int)
        if e["selection_status"] == "clear_leader" and r["selection_status"] == "clear_leader" and not overlap:
            cls = "clear_configuration_tradeoff"
        elif overlap:
            cls = "shared_near_optimal_configuration"
        else:
            cls = "uncertain_configuration_tradeoff"
        e_thread, r_thread = int(e["exact_best_threads"]), int(r["exact_best_threads"])
        e_row = summary[(summary["problem_size"] == n) & (summary["num_threads"] == e_thread)].iloc[0]
        r_row = summary[(summary["problem_size"] == n) & (summary["num_threads"] == r_thread)].iloc[0]
        trade_rows.append({
            "problem_size": int(n), "tradeoff_class": cls,
            "package_energy_exact_best_threads": e_thread,
            "package_energy_leader_threads": e["leader_threads"],
            "package_energy_selection_status": e["selection_status"],
            "runtime_exact_best_threads": r_thread,
            "runtime_leader_threads": r["leader_threads"],
            "runtime_selection_status": r["selection_status"],
            "shared_leader_threads": ",".join(overlap) if overlap else "none",
            "energy_penalty_using_runtime_best_pct": 100 * (
                r_row["package_energy_per_op_j_median"] / e_row["package_energy_per_op_j_median"] - 1),
            "runtime_penalty_using_energy_best_pct": 100 * (
                e_row["runtime_per_op_s_median"] / r_row["runtime_per_op_s_median"] - 1),
            "total_energy_sensitivity_same_exact_thread": e_thread == int(total_row_leader["exact_best_threads"]),
            "total_energy_sensitivity_exact_best_threads": int(total_row_leader["exact_best_threads"]),
            "total_energy_sensitivity_leader_threads": total_row_leader["leader_threads"],
        })
    tradeoffs = pd.DataFrame(trade_rows)
    tradeoffs.to_csv(result_dir / "within_platform_energy_runtime_tradeoffs.csv", index=False)

    save_plots(summary, sessions, result_dir, platform)

    unstable_runtime = summary[~summary["stable_runtime_5pct"]]
    unstable_energy = summary[~summary["stable_package_energy_5pct"]]
    clear_tradeoffs = tradeoffs[tradeoffs["tradeoff_class"] == "clear_configuration_tradeoff"]
    outlier_share = len(outliers) / len(df) if len(df) else math.nan
    report = f"""# {platform} STRIDED_GEMM scientific analysis

## Campaign and statistical unit

- Campaign: `{campaign.stamp}`
- Raw measurements: {len(df)}
- Sessions: {len(campaign.files)}
- Repetitions per configuration/session: 10
- Primary statistical unit: session median
- Primary energy: CPU package RAPL (`device_energy_j / batches`)
- Optional sensitivity: package + DRAM (`total_energy_j / batches`) where DRAM RAPL exists
- Layout: N×N logical matrices with `ld=2N`; allocated footprint is 2× dense GEMM

The ten repetitions inside one session are technical repetitions. They characterize local
noise but are not treated as 50 independent experimental sessions.

## Quality indicators

- Maximum temperature: {df['temp_c'].max():.1f} C
- Diagnostic robust-outlier share: {100*outlier_share:.2f}%
- Runtime-unstable configurations, between-session CV >5%: {len(unstable_runtime)} / {len(summary)}
- Package-energy-unstable configurations, between-session CV >5%: {len(unstable_energy)} / {len(summary)}
- Clear energy-runtime thread trade-offs: {len(clear_tradeoffs)} / {summary['problem_size'].nunique()}

## Primary package-energy leaders

{markdown_table(energy[['problem_size','exact_best_threads','leader_threads','selection_status','gap_to_second_pct','package_energy_per_op_j_median','runtime_per_op_s_median']], 20)}

## Runtime leaders

{markdown_table(runtime[['problem_size','exact_best_threads','leader_threads','selection_status','gap_to_second_pct','runtime_per_op_s_median','package_energy_per_op_j_median']], 20)}

## Energy-runtime configuration classification

{markdown_table(tradeoffs, 20)}

## Interpretation contract

1. Runtime and throughput are inverse views of the same fixed-work axis; energy and GFLOP/J
   are inverse views of the same fixed-work axis. They are not counted as independent votes.
2. EDP is a composite of runtime and energy, not a third independent physical objective.
3. A clear leader requires a >{100*PRACTICAL_TOLERANCE:.0f}% median gap and separated 95%
   bootstrap intervals across five session medians.
4. Exact minima are preserved, but unresolved cases are labeled `tie_or_uncertain`.
5. `logical_bytes_per_op=12N²` describes logical operands. The allocated footprint is 24N²
   because all three matrices use N×2N storage. This is not equivalent to measured DRAM traffic.
6. Package-only is primary across CPUs because AMD DRAM RAPL is unavailable. Package+DRAM
   is retained only as an optional within-platform sensitivity where DRAM measurements exist.
7. Confidence intervals describe repeatability on this machine, not a processor population.
"""
    write_text(result_dir / "scientific_report.md", report)
    print(f"[{platform}] STRIDED_GEMM scientific analysis written to {result_dir}")


if __name__ == "__main__":
    main()

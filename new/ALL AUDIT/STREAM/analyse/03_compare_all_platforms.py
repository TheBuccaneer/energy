#!/usr/bin/env python3
from __future__ import annotations

import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from all_stream_common import *

METRIC_POLICY = {
    "runtime_s": "runtime_opt",
    "energy_j": "energy_opt",
    "edp_j_s": "edp_opt",
    "logical_bandwidth_gb_s": "bandwidth_opt",
    "logical_gb_per_j": "bytes_per_j_opt",
}


def pairwise(selected: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric, policy in METRIC_POLICY.items():
        lower = LOWER_IS_BETTER[metric]
        subset = selected[selected.policy == policy]
        for size in SIZES:
            for a, b in itertools.combinations(PLATFORMS, 2):
                va = subset[(subset.platform == a) & (subset.problem_size == size)][metric].to_numpy(float)
                vb = subset[(subset.platform == b) & (subset.problem_size == size)][metric].to_numpy(float)
                ma, mb = float(np.median(va)), float(np.median(vb))
                ratio = ma / mb
                lo, hi = bootstrap_ratio_ci(va, vb, (metric, size, a, b))
                rows.append({
                    "metric": metric, "policy": policy, "lower_is_better": lower, "problem_size": size,
                    "platform_a": a, "platform_a_label": LABELS[a], "platform_b": b, "platform_b_label": LABELS[b],
                    "a_median": ma, "b_median": mb, "a_over_b_ratio": ratio,
                    "ratio_ci95_low": lo, "ratio_ci95_high": hi,
                    "probability_a_better": probability_a_better(va, vb, lower),
                    "cliffs_delta_a_minus_b": cliffs_delta(va, vb),
                    "classification": classify_ratio(ratio, lo, hi, a, b, lower),
                    "analysis_type": "descriptive_native_best_post_selection_independent_sessions",
                })
    return pd.DataFrame(rows)


def metric_winners(leaders: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric, policy in METRIC_POLICY.items():
        lower = LOWER_IS_BETTER[metric]
        for size in SIZES:
            group = leaders[(leaders.policy == policy) & (leaders.problem_size == size)].copy().reset_index(drop=True)
            # Build a compact summary table with the selected metric values and intervals.
            temp = pd.DataFrame({
                "platform": group.platform,
                f"{metric}_median": group[f"selected_{metric}_median"],
                f"{metric}_ci95_low": group[f"selected_{metric}_ci95_low"],
                f"{metric}_ci95_high": group[f"selected_{metric}_ci95_high"],
            })
            leader = select_leaders(temp, metric, lower, "platform")
            best = temp.iloc[leader.pop("exact_index")]
            exact_winner = leader.pop("exact_item")
            leader_platforms = leader.pop("leader_items")
            rows.append({
                "metric": metric, "policy": policy, "lower_is_better": lower, "problem_size": size,
                **leader, "exact_winner": exact_winner, "leader_platforms": leader_platforms,
                "exact_winner_label": LABELS[str(best.platform)],
                "exact_winner_value": float(best[f"{metric}_median"]),
            })
    return pd.DataFrame(rows)


def plot_native(leaders: pd.DataFrame, metric: str, policy: str, ylabel: str, path: Path, logy=False):
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    subset = leaders[leaders.policy == policy]
    for platform, group in subset.groupby("platform"):
        group = group.sort_values("problem_size")
        ax.plot(group.problem_size, group[f"selected_{metric}_median"], marker="o", label=LABELS[platform])
    ax.set_xscale("log", base=2)
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("Elemente N")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    out = results_dir(__file__)
    sessions = pd.read_csv(out / "unified_session_medians.csv")
    summary = pd.read_csv(out / "unified_configuration_summary.csv")
    leaders = pd.read_csv(out / "native_policy_leaders.csv")
    selected = pd.read_csv(out / "native_policy_session_medians.csv")

    pairs = pairwise(selected)
    pairs.to_csv(out / "pairwise_native_best_comparisons.csv", index=False)
    winners = metric_winners(leaders)
    winners.to_csv(out / "all_platform_metric_winners.csv", index=False)

    best_group_rows = []
    for metric, policy in METRIC_POLICY.items():
        lower = LOWER_IS_BETTER[metric]
        subset = selected[selected.policy == policy]
        for size in SIZES:
            cpu_platform = leaders[(leaders.policy == policy) & (leaders.problem_size == size) & (leaders.device_kind == "CPU")].sort_values(f"selected_{metric}_median", ascending=lower).iloc[0].platform
            gpu_platform = leaders[(leaders.policy == policy) & (leaders.problem_size == size) & (leaders.device_kind == "GPU")].sort_values(f"selected_{metric}_median", ascending=lower).iloc[0].platform
            va = subset[(subset.platform == cpu_platform) & (subset.problem_size == size)][metric].to_numpy(float)
            vb = subset[(subset.platform == gpu_platform) & (subset.problem_size == size)][metric].to_numpy(float)
            ratio = float(np.median(va) / np.median(vb))
            lo, hi = bootstrap_ratio_ci(va, vb, ("best_cpu_gpu", metric, size, cpu_platform, gpu_platform))
            best_group_rows.append({
                "metric": metric, "policy": policy, "problem_size": size,
                "best_cpu": cpu_platform, "best_cpu_label": LABELS[cpu_platform],
                "best_gpu": gpu_platform, "best_gpu_label": LABELS[gpu_platform],
                "cpu_over_gpu_ratio": ratio, "ratio_ci95_low": lo, "ratio_ci95_high": hi,
                "classification": classify_ratio(ratio, lo, hi, cpu_platform, gpu_platform, lower),
            })
    pd.DataFrame(best_group_rows).to_csv(out / "best_cpu_vs_best_gpu.csv", index=False)

    placement_rows = []
    for size in SIZES:
        row = {"problem_size": size, "working_set_gib": 12.0 * size / (1024 ** 3)}
        for metric, policy in METRIC_POLICY.items():
            win = winners[(winners.metric == metric) & (winners.problem_size == size)].iloc[0]
            row[f"{metric}_exact_winner"] = win.exact_winner
            row[f"{metric}_leader_platforms"] = win.leader_platforms
            row[f"{metric}_selection_status"] = win.selection_status
        row["fastest_equals_greenest_point"] = row["runtime_s_exact_winner"] == row["energy_j_exact_winner"]
        runtime_set = {x for x in str(row["runtime_s_leader_platforms"]).split(",") if x}
        energy_set = {x for x in str(row["energy_j_leader_platforms"]).split(",") if x}
        row["runtime_energy_leader_overlap"] = ",".join(sorted(runtime_set & energy_set))
        row["clear_device_tradeoff"] = (
            not bool(runtime_set & energy_set)
            and row["runtime_s_selection_status"] == "clear_leader"
            and row["energy_j_selection_status"] == "clear_leader"
        )
        placement_rows.append(row)
    pd.DataFrame(placement_rows).to_csv(out / "placement_by_size.csv", index=False)

    pareto_parts = []
    for size, group in summary.groupby("problem_size"):
        part = group.copy()
        part["strict_pareto"] = strict_pareto(part)
        part["practical_pareto_2pct"] = practical_pareto(part)
        pareto_parts.append(part)
    pd.concat(pareto_parts, ignore_index=True).to_csv(out / "all_configuration_pareto.csv", index=False)

    figdir = out / "figures"
    plot_native(leaders, "runtime_s", "runtime_opt", "Laufzeit pro Triad [s]", figdir / "all_platform_runtime_native_best.png", True)
    plot_native(leaders, "energy_j", "energy_opt", "Primärenergie pro Triad [J]", figdir / "all_platform_energy_native_best.png", True)
    plot_native(leaders, "logical_bandwidth_gb_s", "bandwidth_opt", "Logische Bandbreite [GB/s]", figdir / "all_platform_logical_bandwidth_native_best.png")
    plot_native(leaders, "logical_gb_per_j", "bytes_per_j_opt", "Logische GB/J", figdir / "all_platform_logical_gb_per_j_native_best.png")
    plot_native(leaders, "edp_j_s", "edp_opt", "EDP [J·s]", figdir / "all_platform_edp_native_best.png", True)
    print(f"[ALL STREAM] comparisons: {out}")


if __name__ == "__main__":
    main()

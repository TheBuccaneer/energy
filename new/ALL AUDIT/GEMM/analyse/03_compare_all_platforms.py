#!/usr/bin/env python3
from __future__ import annotations

import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from all_gemm_common import *

METRIC_POLICY = {
    "runtime_s": "runtime_opt",
    "energy_j": "energy_opt",
    "edp_j_s": "edp_opt",
    "throughput_gflops": "throughput_opt",
    "efficiency_gflop_per_j": "efficiency_opt",
}


def pairwise_comparisons(selected: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for metric, policy in METRIC_POLICY.items():
        lower = LOWER_IS_BETTER[metric]
        subset = selected[selected["policy"] == policy]
        for size in SIZES:
            for a, b in itertools.combinations(PLATFORMS, 2):
                ga = subset[(subset.platform == a) & (subset.problem_size == size)]
                gb = subset[(subset.platform == b) & (subset.problem_size == size)]
                va = ga[metric].to_numpy(float)
                vb = gb[metric].to_numpy(float)
                point_a = float(np.median(va))
                point_b = float(np.median(vb))
                ratio = point_a / point_b
                ci_lo, ci_hi = bootstrap_ratio_ci(
                    va, vb, seed_parts=(metric, policy, size, a, b)
                )
                rows.append({
                    "metric": metric,
                    "policy": policy,
                    "lower_is_better": lower,
                    "problem_size": size,
                    "platform_a": a,
                    "platform_a_label": PLATFORM_LABELS[a],
                    "platform_b": b,
                    "platform_b_label": PLATFORM_LABELS[b],
                    "a_median": point_a,
                    "b_median": point_b,
                    "a_over_b_ratio": ratio,
                    "ratio_ci95_low": ci_lo,
                    "ratio_ci95_high": ci_hi,
                    "classification": classify_pairwise_ratio(
                        ratio, ci_lo, ci_hi, a, b, lower
                    ),
                    "probability_a_better": probability_a_better(va, vb, lower),
                    "cliffs_delta_a_minus_b": cliffs_delta(va, vb),
                    "a_configuration": ga["configuration"].iloc[0],
                    "b_configuration": gb["configuration"].iloc[0],
                    "a_selection_status": ga["selection_status"].iloc[0],
                    "b_selection_status": gb["selection_status"].iloc[0],
                    "energy_domain_a": ga["energy_domain"].iloc[0],
                    "energy_domain_b": gb["energy_domain"].iloc[0],
                    "inference_scope": "descriptive_native_best_post_selection",
                })
    frame = pd.DataFrame(rows)
    return frame.sort_values(["metric", "problem_size", "platform_a", "platform_b"])


def metric_winners(leaders: pd.DataFrame, pairwise: pd.DataFrame) -> pd.DataFrame:
    """Select all-platform leaders using the pairwise ratio-CI rule.

    The point winner is the best platform median. A competitor remains in the
    leader set unless the exact winner is classified as clearly better than it
    by the same practical-ratio CI rule used in the pairwise tables. This avoids
    contradictory labels between the winner table and pairwise comparisons.
    """
    rows: list[dict] = []
    for metric, policy in METRIC_POLICY.items():
        lower = LOWER_IS_BETTER[metric]
        for size in SIZES:
            raw = leaders[(leaders.policy == policy) & (leaders.problem_size == size)].copy()
            value_col = f"selected_{metric}_median"
            exact_idx = raw[value_col].idxmin() if lower else raw[value_col].idxmax()
            exact_row = raw.loc[exact_idx]
            exact_platform = str(exact_row.platform)
            exact_value = float(exact_row[value_col])

            point_values = raw[value_col].astype(float).sort_values(ascending=lower)
            second = float(point_values.iloc[1])
            gap_to_second = 100.0 * practical_gap(second, exact_value, lower)

            leader_platforms = [exact_platform]
            ratio_clear_against_all = True
            for other in PLATFORMS:
                if other == exact_platform:
                    continue
                if PLATFORMS.index(exact_platform) < PLATFORMS.index(other):
                    row = pairwise[
                        (pairwise.metric == metric)
                        & (pairwise.problem_size == size)
                        & (pairwise.platform_a == exact_platform)
                        & (pairwise.platform_b == other)
                    ].iloc[0]
                else:
                    row = pairwise[
                        (pairwise.metric == metric)
                        & (pairwise.problem_size == size)
                        & (pairwise.platform_a == other)
                        & (pairwise.platform_b == exact_platform)
                    ].iloc[0]
                expected = f"clear_{exact_platform}"
                if str(row.classification) != expected:
                    ratio_clear_against_all = False
                    leader_platforms.append(other)

            # Keep the old marginal-CI diagnostic as a separate field.
            marginal_group = raw.rename(columns={
                value_col: f"{metric}_median",
                f"selected_{metric}_ci95_low": f"{metric}_ci95_low",
                f"selected_{metric}_ci95_high": f"{metric}_ci95_high",
                "platform": "configuration",
            })
            marginal = select_leaders(marginal_group, metric, lower)

            clear = (
                ratio_clear_against_all
                and gap_to_second > 100.0 * PRACTICAL_TOLERANCE
                and len(leader_platforms) == 1
            )
            rows.append({
                "metric": metric,
                "policy": policy,
                "lower_is_better": lower,
                "problem_size": size,
                "exact_winner": exact_platform,
                "exact_winner_label": PLATFORM_LABELS[exact_platform],
                "exact_winner_value": exact_value,
                "leader_platforms": ",".join(leader_platforms),
                "leader_count": len(leader_platforms),
                "selection_status": "clear_leader" if clear else "tie_or_uncertain",
                "gap_to_second_pct": gap_to_second,
                "practical_gap_gt_tolerance": gap_to_second > 100.0 * PRACTICAL_TOLERANCE,
                "ci_separated_from_all": ratio_clear_against_all,
                "ratio_ci_clear_against_all": ratio_clear_against_all,
                "marginal_ci_separated_from_all": marginal["ci_separated_from_all"],
                "practical_tolerance_pct": 100 * PRACTICAL_TOLERANCE,
                "inference_scope": "descriptive_native_best_post_selection",
            })
    return pd.DataFrame(rows).sort_values(["metric", "problem_size"])



def best_group_comparison(selected: pd.DataFrame, leaders: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for metric, policy in METRIC_POLICY.items():
        lower = LOWER_IS_BETTER[metric]
        leader_policy = leaders[leaders.policy == policy]
        selected_policy = selected[selected.policy == policy]
        for size in SIZES:
            current = leader_policy[leader_policy.problem_size == size]
            cpu = current[current.device_kind == "CPU"]
            gpu = current[current.device_kind == "GPU"]
            value_col = f"selected_{metric}_median"
            cpu_row = cpu.loc[cpu[value_col].idxmin() if lower else cpu[value_col].idxmax()]
            gpu_row = gpu.loc[gpu[value_col].idxmin() if lower else gpu[value_col].idxmax()]
            cpu_platform = str(cpu_row.platform)
            gpu_platform = str(gpu_row.platform)
            cpu_sessions = selected_policy[(selected_policy.problem_size == size) & (selected_policy.platform == cpu_platform)]
            gpu_sessions = selected_policy[(selected_policy.problem_size == size) & (selected_policy.platform == gpu_platform)]
            vc = cpu_sessions[metric].to_numpy(float)
            vg = gpu_sessions[metric].to_numpy(float)
            cpu_point = float(np.median(vc))
            gpu_point = float(np.median(vg))
            ratio = cpu_point / gpu_point
            lo, hi = bootstrap_ratio_ci(vc, vg, seed_parts=("best_cpu_gpu", metric, size, cpu_platform, gpu_platform))
            rows.append({
                "metric": metric,
                "policy": policy,
                "lower_is_better": lower,
                "problem_size": size,
                "best_cpu": cpu_platform,
                "best_cpu_label": PLATFORM_LABELS[cpu_platform],
                "best_cpu_configuration": cpu_row.exact_configuration,
                "best_cpu_selection_status": cpu_row.selection_status,
                "best_cpu_value": cpu_point,
                "best_gpu": gpu_platform,
                "best_gpu_label": PLATFORM_LABELS[gpu_platform],
                "best_gpu_configuration": gpu_row.exact_configuration,
                "best_gpu_selection_status": gpu_row.selection_status,
                "best_gpu_value": gpu_point,
                "cpu_over_gpu_ratio": ratio,
                "ratio_ci95_low": lo,
                "ratio_ci95_high": hi,
                "classification": classify_pairwise_ratio(ratio, lo, hi, "CPU", "GPU", lower),
                "probability_cpu_better": probability_a_better(vc, vg, lower),
                "cliffs_delta_cpu_minus_gpu": cliffs_delta(vc, vg),
                "cpu_energy_domain": cpu_row.energy_domain,
                "gpu_energy_domain": gpu_row.energy_domain,
                "inference_scope": "descriptive_native_best_post_selection",
            })
    return pd.DataFrame(rows).sort_values(["metric", "problem_size"])


def placement_table(winners: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for size in SIZES:
        subset = winners[winners.problem_size == size].set_index("metric")
        energy = subset.loc["energy_j"]
        runtime = subset.loc["runtime_s"]
        edp = subset.loc["edp_j_s"]
        energy_set = {x for x in str(energy.leader_platforms).split(",") if x}
        runtime_set = {x for x in str(runtime.leader_platforms).split(",") if x}
        overlap = sorted(energy_set & runtime_set)
        if overlap:
            placement_class = "shared_energy_runtime_leader"
        elif energy.selection_status == "clear_leader" and runtime.selection_status == "clear_leader":
            placement_class = "clear_device_tradeoff"
        else:
            placement_class = "uncertain_device_tradeoff"
        rows.append({
            "problem_size": size,
            "energy_exact_winner": energy.exact_winner,
            "energy_leaders": energy.leader_platforms,
            "energy_selection_status": energy.selection_status,
            "runtime_exact_winner": runtime.exact_winner,
            "runtime_leaders": runtime.leader_platforms,
            "runtime_selection_status": runtime.selection_status,
            "edp_exact_winner": edp.exact_winner,
            "edp_leaders": edp.leader_platforms,
            "edp_selection_status": edp.selection_status,
            "energy_runtime_leader_overlap": ",".join(overlap),
            "placement_class": placement_class,
        })
    return pd.DataFrame(rows)


def pareto_and_tradeoffs(summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pareto_parts: list[pd.DataFrame] = []
    trade_rows: list[dict] = []
    count_rows: list[dict] = []

    for size, group in summary.groupby("problem_size"):
        data = group.copy().reset_index(drop=True)
        strict = []
        practical = []
        for i, row in data.iterrows():
            e = float(row.energy_j_median)
            t = float(row.runtime_s_median)
            strict_dominated = False
            practical_dominated = False
            for j, other in data.iterrows():
                if i == j:
                    continue
                oe = float(other.energy_j_median)
                ot = float(other.runtime_s_median)
                if oe <= e and ot <= t and (oe < e or ot < t):
                    strict_dominated = True
                if (
                    oe <= e * (1 + PRACTICAL_TOLERANCE)
                    and ot <= t * (1 + PRACTICAL_TOLERANCE)
                    and (
                        oe < e / (1 + PRACTICAL_TOLERANCE)
                        or ot < t / (1 + PRACTICAL_TOLERANCE)
                    )
                ):
                    practical_dominated = True
            strict.append(not strict_dominated)
            practical.append(not practical_dominated)
        data["pareto_strict"] = strict
        data["pareto_practical_2pct"] = practical
        pareto_parts.append(data)

        best_e = float(data.energy_j_median.min())
        best_t = float(data.runtime_s_median.min())
        for _, row in data.iterrows():
            e_pen = row.energy_j_median / best_e - 1.0
            t_pen = row.runtime_s_median / best_t - 1.0
            if e_pen <= PRACTICAL_TOLERANCE and t_pen <= PRACTICAL_TOLERANCE:
                category = "dominant_or_practically_equivalent"
            elif e_pen <= PRACTICAL_TOLERANCE and t_pen > PRACTICAL_TOLERANCE:
                category = "energy_efficient_compromise"
            elif t_pen <= PRACTICAL_TOLERANCE and e_pen > PRACTICAL_TOLERANCE:
                category = "runtime_efficient_compromise"
            elif bool(row.pareto_practical_2pct):
                category = "balanced_pareto_tradeoff"
            else:
                category = "dominated"
            trade_rows.append({
                "problem_size": int(size),
                "platform": row.platform,
                "platform_label": row.platform_label,
                "configuration": row.configuration,
                "num_threads": int(row.num_threads),
                "energy_domain": row.energy_domain,
                "runtime_s_median": row.runtime_s_median,
                "energy_j_median": row.energy_j_median,
                "throughput_gflops_median": row.throughput_gflops_median,
                "efficiency_gflop_per_j_median": row.efficiency_gflop_per_j_median,
                "edp_j_s_median": row.edp_j_s_median,
                "runtime_penalty_vs_global_best_pct": 100 * t_pen,
                "energy_penalty_vs_global_best_pct": 100 * e_pen,
                "pareto_strict": bool(row.pareto_strict),
                "pareto_practical_2pct": bool(row.pareto_practical_2pct),
                "tradeoff_class": category,
            })

        counts = pd.Series([r["tradeoff_class"] for r in trade_rows if r["problem_size"] == size]).value_counts()
        for category, count in counts.items():
            count_rows.append({
                "problem_size": int(size),
                "tradeoff_class": category,
                "configurations": int(count),
            })

    return (
        pd.concat(pareto_parts, ignore_index=True),
        pd.DataFrame(trade_rows),
        pd.DataFrame(count_rows),
    )


def crossover_summary(pairwise: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for keys, group in pairwise.groupby(["metric", "platform_a", "platform_b"]):
        metric, a, b = keys
        ordered = group.sort_values("problem_size")
        previous = None
        for _, row in ordered.iterrows():
            classification = str(row.classification)
            if classification.startswith("clear_"):
                state = classification.removeprefix("clear_")
            elif classification.startswith("uncertain_"):
                state = "uncertain"
            else:
                state = "equivalent_or_uncertain"
            if previous is not None and state != previous[1]:
                rows.append({
                    "metric": metric,
                    "platform_a": a,
                    "platform_b": b,
                    "from_problem_size": previous[0],
                    "to_problem_size": int(row.problem_size),
                    "from_state": previous[1],
                    "to_state": state,
                })
            previous = (int(row.problem_size), state)
    return pd.DataFrame(rows)


def plot_native_metric(leaders: pd.DataFrame, metric: str, policy: str, ylabel: str, filename: str, figdir, logy: bool = True) -> None:
    plt.figure(figsize=(8.5, 5.5))
    subset = leaders[leaders.policy == policy]
    for platform in PLATFORMS:
        group = subset[subset.platform == platform].sort_values("problem_size")
        x = group.problem_size.to_numpy(float)
        y = group[f"selected_{metric}_median"].to_numpy(float)
        lo = group[f"selected_{metric}_ci95_low"].to_numpy(float)
        hi = group[f"selected_{metric}_ci95_high"].to_numpy(float)
        plt.errorbar(x, y, yerr=[y - lo, hi - y], marker="o", capsize=3, label=PLATFORM_LABELS[platform])
    plt.xscale("log", base=2)
    if logy:
        plt.yscale("log")
    plt.xlabel("Matrix size N")
    plt.ylabel(ylabel)
    plt.title(f"GEMM native-best comparison: {ylabel}")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figdir / filename, dpi=180)
    plt.close()


def plot_pair_ratio(pairwise: pd.DataFrame, metric: str, a: str, b: str, filename: str, figdir) -> None:
    group = pairwise[(pairwise.metric == metric) & (pairwise.platform_a == a) & (pairwise.platform_b == b)].sort_values("problem_size")
    x = group.problem_size.to_numpy(float)
    ratio = group.a_over_b_ratio.to_numpy(float)
    lo = group.ratio_ci95_low.to_numpy(float)
    hi = group.ratio_ci95_high.to_numpy(float)
    plt.figure(figsize=(8.5, 5.5))
    plt.errorbar(x, ratio, yerr=[ratio - lo, hi - ratio], marker="o", capsize=3)
    plt.axhline(1.0, linestyle="--")
    plt.axhline(1.0 + PRACTICAL_TOLERANCE, linestyle=":")
    plt.axhline(1.0 / (1.0 + PRACTICAL_TOLERANCE), linestyle=":")
    plt.xscale("log", base=2)
    plt.xlabel("Matrix size N")
    plt.ylabel(f"{PLATFORM_LABELS[a]} / {PLATFORM_LABELS[b]} ratio")
    plt.title(f"{metric}: {PLATFORM_LABELS[a]} vs {PLATFORM_LABELS[b]}")
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(figdir / filename, dpi=180)
    plt.close()


def plot_pareto_by_size(pareto: pd.DataFrame, figdir) -> None:
    for size in SIZES:
        group = pareto[pareto.problem_size == size]
        plt.figure(figsize=(7.5, 5.5))
        for platform in PLATFORMS:
            p = group[group.platform == platform]
            if p.empty:
                continue
            plt.scatter(p.runtime_s_median, p.energy_j_median, label=PLATFORM_LABELS[platform], alpha=0.75)
        front = group[group.pareto_practical_2pct].sort_values("runtime_s_median")
        if len(front) >= 2:
            plt.plot(front.runtime_s_median, front.energy_j_median, linestyle="--", alpha=0.7)
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Runtime per GEMM (s)")
        plt.ylabel("Measured device-domain energy per GEMM (J)")
        plt.title(f"GEMM energy-runtime configurations at N={size}")
        plt.grid(True, which="both", alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(figdir / f"pareto_N{size}.png", dpi=180)
        plt.close()


def main() -> None:
    out = results_dir(__file__)
    for name in ["native_policy_leaders.csv", "native_policy_session_medians.csv", "unified_configuration_summary.csv"]:
        if not (out / name).is_file():
            raise SystemExit("Run 02_build_unified_stats.py first.")

    leaders = pd.read_csv(out / "native_policy_leaders.csv")
    selected = pd.read_csv(out / "native_policy_session_medians.csv")
    summary = pd.read_csv(out / "unified_configuration_summary.csv")

    pairwise = pairwise_comparisons(selected)
    pairwise.to_csv(out / "pairwise_native_best_comparisons.csv", index=False)

    winners = metric_winners(leaders, pairwise)
    winners.to_csv(out / "all_platform_metric_winners.csv", index=False)

    best_cpu_gpu = best_group_comparison(selected, leaders)
    best_cpu_gpu.to_csv(out / "best_cpu_vs_best_gpu.csv", index=False)

    placement = placement_table(winners)
    placement.to_csv(out / "placement_by_size.csv", index=False)

    pareto, tradeoffs, counts = pareto_and_tradeoffs(summary)
    pareto.to_csv(out / "all_configuration_pareto.csv", index=False)
    tradeoffs.to_csv(out / "configuration_tradeoff_map.csv", index=False)
    counts.to_csv(out / "configuration_tradeoff_counts.csv", index=False)

    crossover = crossover_summary(pairwise)
    crossover.to_csv(out / "crossover_summary.csv", index=False)

    # Preserve the already-audited CPU-only fair views in the central output directory.
    cpu_source = platform_files(__file__, "AMD").results
    for source_name, target_name in [
        ("cross_common_thread_comparison.csv", "cpu_common_thread_comparison.csv"),
        ("cross_native_best_comparison.csv", "cpu_native_best_comparison.csv"),
        ("cross_tradeoff_counts.csv", "cpu_tradeoff_counts.csv"),
    ]:
        source = cpu_source / source_name
        if source.is_file():
            pd.read_csv(source).to_csv(out / target_name, index=False)

    figdir = out / "figures"
    plot_native_metric(leaders, "runtime_s", "runtime_opt", "Runtime per GEMM (s)", "all_platform_runtime_native_best.png", figdir)
    plot_native_metric(leaders, "energy_j", "energy_opt", "Measured device-domain energy per GEMM (J)", "all_platform_energy_native_best.png", figdir)
    plot_native_metric(leaders, "edp_j_s", "edp_opt", "Device-domain EDP (J·s)", "all_platform_edp_native_best.png", figdir)
    plot_native_metric(leaders, "throughput_gflops", "throughput_opt", "Throughput (GFLOP/s)", "all_platform_throughput_native_best.png", figdir, False)
    plot_native_metric(leaders, "efficiency_gflop_per_j", "efficiency_opt", "Measured GFLOP/J", "all_platform_efficiency_native_best.png", figdir, False)

    for metric in ["runtime_s", "energy_j", "edp_j_s", "throughput_gflops", "efficiency_gflop_per_j"]:
        plot_pair_ratio(pairwise, metric, "3090", "5060ti", f"gpu_3090_over_5060ti_{metric}.png", figdir)

    plot_pareto_by_size(pareto, figdir)

    print(f"[ALL GEMM] all-platform comparisons written to {out}")


if __name__ == "__main__":
    main()

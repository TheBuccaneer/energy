#!/usr/bin/env python3
"""Compare Intel and AMD using common-thread and native-best views."""
from __future__ import annotations

import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gemm_common import (
    THREADS, add_derived, campaign_summary, context, load_campaign,
    markdown_table, metric_leaders, write_text,
)

TOLERANCE = 0.02


def classify(row: pd.Series) -> str:
    e_ratio = row["intel_energy_j"] / row["amd_energy_j"]
    t_ratio = row["intel_runtime_s"] / row["amd_runtime_s"]
    intel_e = e_ratio < 1 - TOLERANCE
    amd_e = e_ratio > 1 + TOLERANCE
    intel_t = t_ratio < 1 - TOLERANCE
    amd_t = t_ratio > 1 + TOLERANCE
    if intel_e and intel_t:
        return "Intel dominant"
    if amd_e and amd_t:
        return "AMD dominant"
    if intel_e and amd_t:
        return "Intel energy-efficient / AMD runtime-efficient"
    if amd_e and intel_t:
        return "AMD energy-efficient / Intel runtime-efficient"
    return "practically equivalent or mixed within 2%"


def load_summary(root, platform: str):
    run_dir = root / platform / "runs" / "GEMM"
    campaign = load_campaign(run_dir, platform)
    summary = campaign_summary(add_derived(campaign.data))
    return campaign, summary


def native_best(summary: pd.DataFrame, prefix: str) -> pd.DataFrame:
    energy, _ = metric_leaders(summary, "package_energy_per_op_j_median")
    runtime, _ = metric_leaders(summary, "runtime_per_op_s_median")
    edp, _ = metric_leaders(summary, "edp_package_j_s_median")

    rows = []
    for n in sorted(summary["problem_size"].unique()):
        e = energy[energy["problem_size"] == n].iloc[0]
        t = runtime[runtime["problem_size"] == n].iloc[0]
        p = edp[edp["problem_size"] == n].iloc[0]
        rows.append({
            "problem_size": int(n),
            f"{prefix}_energy_best_threads": int(e["exact_min_threads"]),
            f"{prefix}_energy_leader_threads": e["leader_threads"],
            f"{prefix}_energy_selection_status": e["selection_status"],
            f"{prefix}_energy_best_j": e["package_energy_per_op_j_median"],
            f"{prefix}_energy_best_runtime_s": e["runtime_per_op_s_median"],
            f"{prefix}_runtime_best_threads": int(t["exact_min_threads"]),
            f"{prefix}_runtime_leader_threads": t["leader_threads"],
            f"{prefix}_runtime_selection_status": t["selection_status"],
            f"{prefix}_runtime_best_s": t["runtime_per_op_s_median"],
            f"{prefix}_runtime_best_energy_j": t["package_energy_per_op_j_median"],
            f"{prefix}_edp_best_threads": int(p["exact_min_threads"]),
            f"{prefix}_edp_leader_threads": p["leader_threads"],
            f"{prefix}_edp_selection_status": p["selection_status"],
            f"{prefix}_edp_best_j_s": p["edp_package_j_s_median"],
        })
    return pd.DataFrame(rows)


def write_outputs(result_dir, common, native, report, source_figure_dir):
    common.to_csv(result_dir / "cross_common_thread_comparison.csv", index=False)
    native.to_csv(result_dir / "cross_native_best_comparison.csv", index=False)
    common.groupby("tradeoff_class", as_index=False).size().rename(columns={"size": "count"}).to_csv(
        result_dir / "cross_tradeoff_counts.csv", index=False
    )
    write_text(result_dir / "cross_platform_report.md", report)
    target = result_dir / "figures"
    target.mkdir(exist_ok=True)
    for path in source_figure_dir.glob("cross_*.png"):
        destination = target / path.name
        if path.resolve() != destination.resolve():
            shutil.copy2(path, destination)


def main() -> None:
    root, _, _, _ = context(__file__)
    intel_campaign, intel = load_summary(root, "INTEL")
    amd_campaign, amd = load_summary(root, "AMD")

    common_threads = sorted(set(THREADS["INTEL"]) & set(THREADS["AMD"]))
    cols = [
        "problem_size", "num_threads", "runtime_per_op_s_median",
        "package_energy_per_op_j_median", "package_avg_power_w_median",
        "gflops_per_s_median", "edp_package_j_s_median",
        "runtime_per_op_s_between_session_robust_cv",
        "package_energy_per_op_j_between_session_robust_cv",
    ]
    i = intel[intel["num_threads"].isin(common_threads)][cols].copy()
    a = amd[amd["num_threads"].isin(common_threads)][cols].copy()
    i = i.rename(columns={
        "runtime_per_op_s_median": "intel_runtime_s",
        "package_energy_per_op_j_median": "intel_energy_j",
        "package_avg_power_w_median": "intel_power_w",
        "gflops_per_s_median": "intel_gflops",
        "edp_package_j_s_median": "intel_edp_j_s",
        "runtime_per_op_s_between_session_robust_cv": "intel_runtime_session_cv",
        "package_energy_per_op_j_between_session_robust_cv": "intel_energy_session_cv",
    })
    a = a.rename(columns={
        "runtime_per_op_s_median": "amd_runtime_s",
        "package_energy_per_op_j_median": "amd_energy_j",
        "package_avg_power_w_median": "amd_power_w",
        "gflops_per_s_median": "amd_gflops",
        "edp_package_j_s_median": "amd_edp_j_s",
        "runtime_per_op_s_between_session_robust_cv": "amd_runtime_session_cv",
        "package_energy_per_op_j_between_session_robust_cv": "amd_energy_session_cv",
    })
    common = i.merge(a, on=["problem_size", "num_threads"], validate="one_to_one")
    common["intel_over_amd_runtime_ratio"] = common["intel_runtime_s"] / common["amd_runtime_s"]
    common["intel_over_amd_energy_ratio"] = common["intel_energy_j"] / common["amd_energy_j"]
    common["intel_over_amd_power_ratio"] = common["intel_power_w"] / common["amd_power_w"]
    common["intel_over_amd_edp_ratio"] = common["intel_edp_j_s"] / common["amd_edp_j_s"]
    common["tradeoff_class"] = common.apply(classify, axis=1)

    native = native_best(intel, "intel").merge(native_best(amd, "amd"), on="problem_size")
    native["intel_over_amd_energy_best_ratio"] = native["intel_energy_best_j"] / native["amd_energy_best_j"]
    native["intel_over_amd_runtime_best_ratio"] = native["intel_runtime_best_s"] / native["amd_runtime_best_s"]
    native["intel_over_amd_edp_best_ratio"] = native["intel_edp_best_j_s"] / native["amd_edp_best_j_s"]

    temp_figures = root / "INTEL" / "results" / "GEMM" / "figures"
    temp_figures.mkdir(parents=True, exist_ok=True)
    for metric, label, filename in [
        ("intel_over_amd_runtime_ratio", "Intel / AMD runtime ratio", "cross_runtime_ratio_heatmap.png"),
        ("intel_over_amd_energy_ratio", "Intel / AMD package-energy ratio", "cross_energy_ratio_heatmap.png"),
    ]:
        pivot = common.pivot(index="problem_size", columns="num_threads", values=metric)
        fig, ax = plt.subplots(figsize=(9, 5.5))
        image = ax.imshow(pivot.to_numpy(), aspect="auto", origin="lower", vmin=0.5, vmax=1.5)
        ax.set_xticks(range(len(pivot.columns)), [str(int(x)) for x in pivot.columns])
        ax.set_yticks(range(len(pivot.index)), [str(int(x)) for x in pivot.index])
        ax.set_xlabel("Common thread count")
        ax.set_ylabel("N")
        ax.set_title(label + " (<1 favors Intel, >1 favors AMD)")
        fig.colorbar(image, ax=ax, label="ratio")
        fig.tight_layout()
        fig.savefig(temp_figures / filename, dpi=180)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(native["problem_size"], native["intel_energy_best_j"], marker="o", label="Intel native best")
    ax.plot(native["problem_size"], native["amd_energy_best_j"], marker="o", label="AMD native best")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("N")
    ax.set_ylabel("Minimum package energy per GEMM [J]")
    ax.set_title("Native-best package energy")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(temp_figures / "cross_native_best_energy.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(native["problem_size"], native["intel_runtime_best_s"], marker="o", label="Intel native best")
    ax.plot(native["problem_size"], native["amd_runtime_best_s"], marker="o", label="AMD native best")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("N")
    ax.set_ylabel("Minimum runtime per GEMM [s]")
    ax.set_title("Native-best runtime")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(temp_figures / "cross_native_best_runtime.png", dpi=180)
    plt.close(fig)

    counts = common["tradeoff_class"].value_counts().rename_axis("class").reset_index(name="count")
    report = f"""# Intel–AMD GEMM comparison

- Intel campaign: `{intel_campaign.stamp}`
- AMD campaign: `{amd_campaign.stamp}`
- Common-thread comparison: threads {common_threads}
- Native-best comparison: Intel may use up to 20 threads; AMD may use 32/64
- Equivalence tolerance for qualitative classification: ±{100*TOLERANCE:.0f}%
- Primary energy metric: package-only `device_energy_j / batches`

## Why there are two comparisons

The **common-thread** view holds software parallelism constant and is the cleaner
hardware comparison at matched thread counts. The **native-best** view asks what
each processor can achieve when allowed to use its available thread grid. These
answer different research questions and must not be merged into one winner count.

## Common-thread trade-off counts

{markdown_table(counts, 20)}

## Native-best comparison by size

{markdown_table(native[["problem_size", "intel_energy_best_threads", "intel_energy_leader_threads", "intel_energy_selection_status", "amd_energy_best_threads", "amd_energy_leader_threads", "amd_energy_selection_status", "intel_over_amd_energy_best_ratio", "intel_runtime_best_threads", "intel_runtime_leader_threads", "intel_runtime_selection_status", "amd_runtime_best_threads", "amd_runtime_leader_threads", "amd_runtime_selection_status", "intel_over_amd_runtime_best_ratio"]], 20)}

## Interpretation

- A ratio below 1 favors Intel; above 1 favors AMD.
- Dominance requires both lower package energy and lower runtime by more than 2%.
- Native-best thread fields retain the exact observed minimum, but the accompanying
  leader sets/status prevent a sub-2% or CI-overlapping difference from being
  presented as a unique thread-count winner.
- A trade-off class is not a failure: it identifies whether energy or runtime is
  being exchanged and is more informative than a single winner count.
- Package energy domains are not physically identical across CPU vendors and do
  not include the same external memory/system components. Results support
  package-level placement decisions, not whole-system energy claims.
- Five session medians provide repeatability evidence for these machines and
  settings; they do not establish generality across processor samples or systems.
"""

    for platform in ["INTEL", "AMD"]:
        result_dir = root / platform / "results" / "GEMM"
        result_dir.mkdir(parents=True, exist_ok=True)
        write_outputs(result_dir, common, native, report, temp_figures)
    print("Cross-platform comparison written to both results/GEMM directories")


if __name__ == "__main__":
    main()

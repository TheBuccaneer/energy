#!/usr/bin/env python3
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from all_strided_common import *

POLICIES = {
    "energy_opt": ("energy_j", True),
    "runtime_opt": ("runtime_s", True),
    "edp_opt": ("edp_j_s", True),
    "throughput_opt": ("throughput_gflops", False),
    "efficiency_opt": ("efficiency_gflop_per_j", False),
}


def main() -> None:
    out = results_dir(__file__)
    preflight = out / "preflight_checks.csv"
    if not preflight.is_file():
        raise SystemExit("Run 01_preflight_all_strided.py first.")
    checks = pd.read_csv(preflight)
    if ((checks["severity"] == "FAIL") & (checks["status"] == "FAIL")).any():
        raise SystemExit("Preflight contains hard failures; unified analysis aborted.")

    sessions = pd.concat(
        [normalize_session_medians(__file__, platform) for platform in PLATFORMS],
        ignore_index=True,
    )
    sessions.to_csv(out / "unified_session_medians.csv", index=False)

    summary = summarize_configurations(sessions)
    summary.to_csv(out / "unified_configuration_summary.csv", index=False)

    leader_rows: list[dict] = []
    selected_session_parts: list[pd.DataFrame] = []

    for (platform, size), group in summary.groupby(["platform", "problem_size"]):
        for policy, (metric, lower) in POLICIES.items():
            leader = select_leaders(group, metric, lower)
            exact_index = leader.pop("exact_index")
            if exact_index is None:
                continue
            exact = summary.loc[exact_index]
            row = {
                "platform": platform,
                "platform_label": PLATFORM_LABELS[platform],
                "device_kind": DEVICE_KIND[platform],
                "energy_domain": ENERGY_DOMAIN[platform],
                "problem_size": int(size),
                "policy": policy,
                "optimized_metric": metric,
                "lower_is_better": lower,
                **leader,
                "exact_num_threads": int(exact["num_threads"]),
            }
            for name in [
                "runtime_s", "energy_j", "power_w", "throughput_gflops",
                "efficiency_gflop_per_j", "edp_j_s", "temperature_c", "clock_mhz",
            ]:
                row[f"selected_{name}_median"] = exact[f"{name}_median"]
                row[f"selected_{name}_ci95_low"] = exact[f"{name}_ci95_low"]
                row[f"selected_{name}_ci95_high"] = exact[f"{name}_ci95_high"]
                row[f"selected_{name}_session_cv_pct"] = exact[f"{name}_session_cv_pct"]
            leader_rows.append(row)

            selected = sessions[
                (sessions["platform"] == platform)
                & (sessions["problem_size"] == size)
                & (sessions["configuration"] == exact["configuration"])
            ].copy()
            selected["policy"] = policy
            selected["optimized_metric"] = metric
            selected["selection_status"] = row["selection_status"]
            selected["leader_configurations"] = row["leader_configurations"]
            selected_session_parts.append(selected)

    leaders = pd.DataFrame(leader_rows).sort_values(["problem_size", "policy", "platform"])
    leaders.to_csv(out / "native_policy_leaders.csv", index=False)

    selected_sessions = pd.concat(selected_session_parts, ignore_index=True)
    selected_sessions.to_csv(out / "native_policy_session_medians.csv", index=False)

    # Compact one-row-per-platform/size table with the three central operating policies.
    compact_rows: list[dict] = []
    for (platform, size), group in leaders.groupby(["platform", "problem_size"]):
        row = {
            "platform": platform,
            "platform_label": PLATFORM_LABELS[platform],
            "device_kind": DEVICE_KIND[platform],
            "energy_domain": ENERGY_DOMAIN[platform],
            "problem_size": int(size),
        }
        for policy in ["energy_opt", "runtime_opt", "edp_opt", "throughput_opt", "efficiency_opt"]:
            p = group[group["policy"] == policy].iloc[0]
            row[f"{policy}_configuration"] = p["exact_configuration"]
            row[f"{policy}_selection_status"] = p["selection_status"]
            row[f"{policy}_leader_configurations"] = p["leader_configurations"]
            row[f"{policy}_runtime_s"] = p["selected_runtime_s_median"]
            row[f"{policy}_energy_j"] = p["selected_energy_j_median"]
            row[f"{policy}_edp_j_s"] = p["selected_edp_j_s_median"]
            row[f"{policy}_throughput_gflops"] = p["selected_throughput_gflops_median"]
            row[f"{policy}_efficiency_gflop_per_j"] = p["selected_efficiency_gflop_per_j_median"]
            row[f"{policy}_power_w"] = p["selected_power_w_median"]
        compact_rows.append(row)
    compact = pd.DataFrame(compact_rows).sort_values(["problem_size", "platform"])
    compact.to_csv(out / "native_best_by_platform_size.csv", index=False)

    # Within-platform energy/runtime trade-off. This remains meaningful mainly on CPUs;
    # GPUs have one resident configuration and therefore zero configuration trade-off.
    tradeoff_rows: list[dict] = []
    for platform in PLATFORMS:
        for size in SIZES:
            e = leaders[(leaders.platform == platform) & (leaders.problem_size == size) & (leaders.policy == "energy_opt")].iloc[0]
            r = leaders[(leaders.platform == platform) & (leaders.problem_size == size) & (leaders.policy == "runtime_opt")].iloc[0]
            e_set = {x for x in str(e.leader_configurations).split(",") if x}
            r_set = {x for x in str(r.leader_configurations).split(",") if x}
            overlap = sorted(e_set & r_set)
            energy_penalty_runtime_opt = 100.0 * (
                r.selected_energy_j_median / e.selected_energy_j_median - 1.0
            )
            runtime_gain_runtime_opt = 100.0 * (
                1.0 - r.selected_runtime_s_median / e.selected_runtime_s_median
            )
            tradeoff_rows.append({
                "platform": platform,
                "platform_label": PLATFORM_LABELS[platform],
                "problem_size": size,
                "energy_opt_configuration": e.exact_configuration,
                "runtime_opt_configuration": r.exact_configuration,
                "energy_leaders": e.leader_configurations,
                "runtime_leaders": r.leader_configurations,
                "leader_set_overlap": ",".join(overlap),
                "leader_sets_disjoint": not bool(overlap),
                "energy_selection_status": e.selection_status,
                "runtime_selection_status": r.selection_status,
                "runtime_opt_energy_penalty_pct": energy_penalty_runtime_opt,
                "runtime_opt_runtime_gain_pct": runtime_gain_runtime_opt,
                "interpretation": (
                    "clear_configuration_tradeoff"
                    if not overlap and e.selection_status == "clear_leader" and r.selection_status == "clear_leader"
                    else "disjoint_but_at_least_one_uncertain"
                    if not overlap
                    else "no_tie_aware_conflict"
                ),
            })
    tradeoffs = pd.DataFrame(tradeoff_rows)
    tradeoffs.to_csv(out / "within_platform_energy_runtime_tradeoffs.csv", index=False)

    # Stability table across all configurations and device types.
    stability = summary[[
        "platform", "platform_label", "problem_size", "configuration", "num_threads",
        "runtime_s_session_cv_pct", "energy_j_session_cv_pct",
        "throughput_gflops_session_cv_pct", "power_w_session_cv_pct",
        "temperature_c_session_cv_pct",
    ]].copy()
    stability["runtime_stable_5pct"] = stability["runtime_s_session_cv_pct"] <= 5.0
    stability["energy_stable_5pct"] = stability["energy_j_session_cv_pct"] <= 5.0
    stability["throughput_stable_5pct"] = stability["throughput_gflops_session_cv_pct"] <= 5.0
    stability.to_csv(out / "all_platform_stability.csv", index=False)

    print(f"[ALL STRIDED_GEMM] unified statistics written to {out}")


if __name__ == "__main__":
    main()

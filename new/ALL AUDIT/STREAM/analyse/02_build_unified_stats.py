#!/usr/bin/env python3
from __future__ import annotations

import numpy as np
import pandas as pd

from all_stream_common import *

POLICIES = {
    "runtime_opt": ("runtime_s", True),
    "energy_opt": ("energy_j", True),
    "edp_opt": ("edp_j_s", True),
    "bandwidth_opt": ("logical_bandwidth_gb_s", False),
    "bytes_per_j_opt": ("logical_gb_per_j", False),
}


def main() -> None:
    out = results_dir(__file__)
    checks = pd.read_csv(out / "preflight_checks.csv")
    if ((checks.severity == "FAIL") & (checks.status == "FAIL")).any():
        raise SystemExit("Preflight contains hard failures.")

    sessions = pd.concat([load_sessions(__file__, p) for p in PLATFORMS], ignore_index=True)
    sessions.to_csv(out / "unified_session_medians.csv", index=False)
    summary = summarize_configurations(sessions)
    summary.to_csv(out / "unified_configuration_summary.csv", index=False)

    leader_rows = []
    selected_parts = []
    for (platform, size), group in summary.groupby(["platform", "problem_size"]):
        for policy, (metric, lower) in POLICIES.items():
            leader = select_leaders(group, metric, lower, "configuration")
            best = group.reset_index(drop=True).iloc[leader.pop("exact_index")]
            exact_configuration = leader.pop("exact_item")
            leader_configurations = leader.pop("leader_items")
            row = {
                "platform": platform, "platform_label": LABELS[platform], "device_kind": KINDS[platform],
                "energy_domain": ENERGY_DOMAINS[platform], "problem_size": int(size), "policy": policy,
                "optimized_metric": metric, "lower_is_better": lower, **leader,
                "exact_configuration": exact_configuration, "leader_configurations": leader_configurations,
                "exact_num_threads": int(best.num_threads),
            }
            for name in ["runtime_s", "energy_j", "total_energy_j", "power_w", "edp_j_s", "throughput_gflops", "efficiency_gflop_per_j", "logical_bandwidth_gb_s", "logical_gb_per_j", "temperature_c", "clock_mhz"]:
                for suffix in ["median", "ci95_low", "ci95_high", "session_cv_pct"]:
                    row[f"selected_{name}_{suffix}"] = best[f"{name}_{suffix}"]
            leader_rows.append(row)
            selected = sessions[(sessions.platform == platform) & (sessions.problem_size == size) & (sessions.configuration == best.configuration)].copy()
            selected["policy"] = policy
            selected["optimized_metric"] = metric
            selected["selection_status"] = row["selection_status"]
            selected_parts.append(selected)

    leaders = pd.DataFrame(leader_rows).sort_values(["problem_size", "policy", "platform"])
    leaders.to_csv(out / "native_policy_leaders.csv", index=False)
    selected = pd.concat(selected_parts, ignore_index=True)
    selected.to_csv(out / "native_policy_session_medians.csv", index=False)

    trade_rows = []
    for platform in PLATFORMS:
        for size in SIZES:
            e = leaders[(leaders.platform == platform) & (leaders.problem_size == size) & (leaders.policy == "energy_opt")].iloc[0]
            r = leaders[(leaders.platform == platform) & (leaders.problem_size == size) & (leaders.policy == "runtime_opt")].iloc[0]
            e_set = {x for x in str(e.leader_configurations).split(",") if x}
            r_set = {x for x in str(r.leader_configurations).split(",") if x}
            overlap = sorted(e_set & r_set)
            trade_rows.append({
                "platform": platform, "platform_label": LABELS[platform], "problem_size": size,
                "energy_opt_configuration": e.exact_configuration, "runtime_opt_configuration": r.exact_configuration,
                "energy_leaders": e.leader_configurations, "runtime_leaders": r.leader_configurations,
                "leader_set_overlap": ",".join(overlap), "leader_sets_disjoint": not bool(overlap),
                "energy_selection_status": e.selection_status, "runtime_selection_status": r.selection_status,
                "runtime_opt_energy_penalty_pct": 100.0 * (r.selected_energy_j_median / e.selected_energy_j_median - 1.0),
                "runtime_opt_runtime_gain_pct": 100.0 * (1.0 - r.selected_runtime_s_median / e.selected_runtime_s_median),
                "interpretation": (
                    "clear_configuration_tradeoff" if not overlap and e.selection_status == "clear_leader" and r.selection_status == "clear_leader"
                    else "disjoint_but_at_least_one_uncertain" if not overlap else "no_tie_aware_conflict"
                ),
            })
    pd.DataFrame(trade_rows).to_csv(out / "within_platform_energy_runtime_tradeoffs.csv", index=False)

    stability = summary[[
        "platform", "platform_label", "problem_size", "configuration", "num_threads",
        "runtime_s_session_cv_pct", "energy_j_session_cv_pct", "logical_bandwidth_gb_s_session_cv_pct",
        "power_w_session_cv_pct", "temperature_c_session_cv_pct",
    ]].copy()
    stability["runtime_stable_5pct"] = stability.runtime_s_session_cv_pct <= 5.0
    stability["energy_stable_5pct"] = stability.energy_j_session_cv_pct <= 5.0
    stability["bandwidth_stable_5pct"] = stability.logical_bandwidth_gb_s_session_cv_pct <= 5.0
    stability.to_csv(out / "all_platform_stability.csv", index=False)
    print(f"[ALL STREAM] unified statistics: {out}")


if __name__ == "__main__":
    main()

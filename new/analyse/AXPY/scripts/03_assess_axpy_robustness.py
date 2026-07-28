#!/usr/bin/env python3
"""Ergänzt fixed-config-, oracle-, CI- und In-Range-Robustheit für AXPY."""
from __future__ import annotations

import argparse
import math
from collections import Counter, defaultdict
from pathlib import Path

from axpy_analysis_common import PLATFORMS, SIZES, median, read_csv, write_csv, write_json

METRICS = {
    "runtime": ("time_e2e_op_s", "median_time_e2e_op_s"),
    "energy": ("device_energy_op_j", "median_device_energy_op_j"),
    "edp": ("edp_device_j_s", "median_edp_device_j_s"),
}


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AXPY Robustheits- und Sensitivitätsanalyse")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--normalized", type=Path, required=True)
    parser.add_argument("--cross", type=Path, required=True)
    parser.add_argument("--pareto", type=Path, required=True)
    parser.add_argument("--comparison-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tie-percent", type=float, default=2.0)
    parser.add_argument("--required-sessions", type=int, default=5)
    parser.add_argument("--required-repetitions", type=int, default=10)
    return parser.parse_args()


def f(row: dict, key: str) -> float:
    return float(row[key])


def ii(row: dict, key: str) -> int:
    return int(float(row[key]))


def best_per_platform(rows: list[dict], metric: str) -> dict[str, dict]:
    result = {}
    for platform in {row["platform"] for row in rows}:
        points = [row for row in rows if row["platform"] == platform]
        result[platform] = min(points, key=lambda row: (f(row, metric), ii(row, "threads")))
    return result


def winner_set(points: list[dict], key: str, tie: float) -> set[str]:
    if not points:
        return set()
    minimum = min(float(row[key]) for row in points)
    return {
        row["platform"]
        for row in points
        if float(row[key]) <= minimum * (1.0 + tie)
    }


def ci_fields(base: str) -> tuple[str, str]:
    return f"ci95_low_{base}", f"ci95_high_{base}"


def selected_fixed_configs(config_rows: list[dict], point_key: str, n: int) -> dict[str, dict]:
    return best_per_platform(
        [row for row in config_rows if ii(row, "problem_size") == n],
        point_key,
    )


def session_winner_support(
    session_rows: list[dict],
    metric_base: str,
    n: int,
    tie: float,
    *,
    selected: dict[str, dict] | None,
) -> tuple[Counter, dict[int, set[str]], int]:
    """selected=None => oracle envelope; otherwise fixed globally selected configs."""
    supports = Counter()
    session_sets: dict[int, set[str]] = {}
    session_numbers = sorted(
        {
            ii(row, "session_number")
            for row in session_rows
            if ii(row, "problem_size") == n
        }
    )
    value_key = f"median_{metric_base}"

    for session_number in session_numbers:
        rows = [
            row for row in session_rows
            if ii(row, "problem_size") == n
            and ii(row, "session_number") == session_number
        ]
        env = []
        if selected is None:
            for platform in {row["platform"] for row in rows}:
                points = [row for row in rows if row["platform"] == platform]
                best = min(points, key=lambda row: (f(row, value_key), ii(row, "threads")))
                env.append({"platform": platform, "value": best[value_key]})
        else:
            for platform, selected_row in selected.items():
                selected_threads = ii(selected_row, "threads")
                matches = [
                    row for row in rows
                    if row["platform"] == platform
                    and ii(row, "threads") == selected_threads
                ]
                if matches:
                    env.append({"platform": platform, "value": matches[0][value_key]})
        if env:
            wins = winner_set(env, "value", tie)
            supports.update(wins)
            session_sets[session_number] = wins
    return supports, session_sets, len(session_numbers)


def display_set(value: str) -> str:
    return value.replace("|", ", ") if value else "–"


def in_range_estimates(
    normalized: list[dict], metric_base: str, expected_repetitions: int
) -> list[dict]:
    grouped = defaultdict(lambda: defaultdict(list))
    for row in normalized:
        if row["runtime_status"] != "in_range":
            continue
        key = (row["platform"], ii(row, "problem_size"), ii(row, "threads"))
        grouped[key][ii(row, "session_number")].append(float(row[metric_base]))

    configs = []
    for (platform, n, threads), sessions in grouped.items():
        ordered = sorted(sessions.items())
        session_values = [median(values) for _, values in ordered if values]
        counts = [len(values) for _, values in ordered]
        if session_values:
            configs.append(
                {
                    "platform": platform,
                    "problem_size": n,
                    "threads": threads,
                    "value": median(session_values),
                    "sessions_covered": len(session_values),
                    "complete_sessions": sum(
                        count == expected_repetitions for count in counts
                    ),
                    "min_in_range_repetitions_per_session": min(counts),
                    "total_in_range_repetitions": sum(counts),
                }
            )
    return configs

def in_range_envelope(
    configs: list[dict],
    n: int,
    required_sessions: int | None,
) -> list[dict]:
    points = [row for row in configs if row["problem_size"] == n]
    if required_sessions is not None:
        points = [
            row for row in points
            if row["sessions_covered"] == required_sessions
            and row["complete_sessions"] == required_sessions
        ]
    env = []
    for platform in {row["platform"] for row in points}:
        platform_points = [row for row in points if row["platform"] == platform]
        env.append(min(platform_points, key=lambda row: (row["value"], row["threads"])))
    return env


def main() -> int:
    a = args()
    tie = a.tie_percent / 100.0
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=True)

    config = read_csv(a.config)
    sessions = read_csv(a.session)
    normalized = read_csv(a.normalized)
    cross = read_csv(a.cross)
    pareto = read_csv(a.pareto)
    observed_sessions = len({ii(row, "session_number") for row in sessions})
    expected_sessions = a.required_sessions

    robust_rows = []
    sensitivity = []
    inrange = {
        name: in_range_estimates(normalized, base, a.required_repetitions)
        for name, (base, _) in METRICS.items()
    }
    partial_config_keys: set[tuple[str, int, int]] = set()
    directional_gap_total = 0

    for n in SIZES:
        points = [row for row in config if ii(row, "problem_size") == n]
        if not points:
            continue
        record = {"problem_size": n}

        for name, (base, point_key) in METRICS.items():
            selected = selected_fixed_configs(config, point_key, n)
            env_rows = list(selected.values())
            point_wins = winner_set(
                [
                    {"platform": platform, point_key: row[point_key]}
                    for platform, row in selected.items()
                ],
                point_key,
                tie,
            )

            fixed_support, _, fixed_sessions = session_winner_support(
                sessions, base, n, tie, selected=selected
            )
            oracle_support, _, oracle_sessions = session_winner_support(
                sessions, base, n, tie, selected=None
            )

            ci_separated = set()
            magnitude_stable = set()
            fixed_support_text = []
            oracle_support_text = []
            selected_config_text = []

            for platform in sorted(point_wins):
                row = selected[platform]
                low_key, high_key = ci_fields(base)
                winner_high = f(row, high_key)
                others_low = [
                    f(other, low_key)
                    for other_platform, other in selected.items()
                    if other_platform != platform
                ]
                if others_low and all(
                    winner_high <= value * (1.0 + tie) for value in others_low
                ):
                    ci_separated.add(platform)
                if f(row, f"cv_all_rows_{base}") <= 0.10:
                    magnitude_stable.add(platform)
                fixed_support_text.append(
                    f"{platform}:{fixed_support[platform]}/{fixed_sessions}"
                )
                oracle_support_text.append(
                    f"{platform}:{oracle_support[platform]}/{oracle_sessions}"
                )
                selected_config_text.append(
                    f"{platform}:T{ii(row, 'threads')}"
                )

            fixed_all_session = {
                platform for platform in point_wins
                if fixed_support[platform] == expected_sessions
            }
            oracle_all_session = {
                platform for platform in point_wins
                if oracle_support[platform] == expected_sessions
            }
            fixed_majority = {
                platform for platform in point_wins
                if fixed_support[platform] >= max(1, math.ceil(0.8 * expected_sessions))
            }
            directionally_robust = (
                point_wins & fixed_all_session & ci_separated
            )
            oracle_consistent = point_wins & oracle_all_session
            fully_robust = (
                directionally_robust & oracle_consistent & magnitude_stable
            )
            if point_wins != directionally_robust:
                directional_gap_total += 1

            ir_all = [row for row in inrange[name] if row["problem_size"] == n]
            ir_complete_env = in_range_envelope(
                inrange[name], n, expected_sessions
            )
            ir_any_env = in_range_envelope(inrange[name], n, None)
            ir_complete_wins = winner_set(ir_complete_env, "value", tie)
            ir_any_wins = winner_set(ir_any_env, "value", tie)
            partial_configs = [
                row for row in ir_all if row["sessions_covered"] != expected_sessions or row["complete_sessions"] != expected_sessions
            ]
            partial_config_keys.update(
                (row["platform"], row["problem_size"], row["threads"])
                for row in partial_configs
            )
            complete_platforms = {row["platform"] for row in ir_complete_env}
            partial_platforms = {
                row["platform"] for row in partial_configs
            }
            primary_agrees = point_wins == ir_complete_wins

            sensitivity.append(
                {
                    "problem_size": n,
                    "metric": name,
                    "point_winners": "|".join(sorted(point_wins)),
                    "in_range_complete_5x10_winners": "|".join(sorted(ir_complete_wins)),
                    "in_range_any_coverage_winners": "|".join(sorted(ir_any_wins)),
                    "primary_same_winner_set": "yes" if primary_agrees else "no",
                    "complete_platforms": "|".join(sorted(complete_platforms)),
                    "partial_platforms": "|".join(sorted(partial_platforms)),
                    "partial_configurations": len(partial_configs),
                    "required_sessions": expected_sessions,
                    "required_repetitions_per_session": a.required_repetitions,
                    "minimum_in_range_repetitions_in_partial_configs": min(
                        [row["min_in_range_repetitions_per_session"] for row in partial_configs],
                        default=a.required_repetitions,
                    ),
                }
            )

            record[f"{name}_selected_configs"] = "|".join(selected_config_text)
            record[f"{name}_point_winners"] = "|".join(sorted(point_wins))
            record[f"{name}_fixed_config_session_support"] = "|".join(fixed_support_text)
            record[f"{name}_oracle_envelope_session_support"] = "|".join(oracle_support_text)
            record[f"{name}_fixed_all_session_winners"] = "|".join(sorted(fixed_all_session))
            record[f"{name}_oracle_all_session_winners"] = "|".join(sorted(oracle_all_session))
            record[f"{name}_oracle_consistent_winners"] = "|".join(sorted(oracle_consistent))
            record[f"{name}_fixed_majority_winners"] = "|".join(sorted(fixed_majority))
            record[f"{name}_ci_separated_winners"] = "|".join(sorted(ci_separated))
            record[f"{name}_magnitude_stable_winners"] = "|".join(sorted(magnitude_stable))
            record[f"{name}_directionally_robust_winners"] = "|".join(sorted(directionally_robust))
            record[f"{name}_fully_robust_winners"] = "|".join(sorted(fully_robust))
            # Kompatibilitätsalias, jetzt eindeutig als strengste Klasse definiert.
            record[f"{name}_robust_winners"] = "|".join(sorted(fully_robust))
            record[f"{name}_in_range_complete_5x10_winners"] = "|".join(sorted(ir_complete_wins))
            record[f"{name}_in_range_any_coverage_winners"] = "|".join(sorted(ir_any_wins))
            record[f"{name}_in_range_primary_agrees"] = "yes" if primary_agrees else "no"
            record[f"{name}_in_range_partial_configs"] = len(partial_configs)

        robust_rows.append(record)

    robust_by_n = {row["problem_size"]: row for row in robust_rows}
    merged = []
    for row in cross:
        n = ii(row, "problem_size")
        item = dict(row)
        item.update(
            {key: value for key, value in robust_by_n[n].items() if key != "problem_size"}
        )
        merged.append(item)
    write_csv(a.cross, merged, list(merged[0].keys()))
    write_csv(
        out / "axpy_winner_robustness.csv",
        robust_rows,
        list(robust_rows[0].keys()),
    )
    write_csv(
        out / "axpy_in_range_sensitivity.csv",
        sensitivity,
        list(sensitivity[0].keys()),
    )

    # Konservative Unsicherheits-Pareto-Klassifikation.
    by_n = defaultdict(list)
    for row in pareto:
        by_n[ii(row, "problem_size")].append(row)
    pareto_new = []
    for n, points in by_n.items():
        for candidate in points:
            dominators = []
            for other in points:
                if other is candidate:
                    continue
                time_sep = (
                    f(other, "ci95_high_time_e2e_op_s")
                    <= f(candidate, "ci95_low_time_e2e_op_s")
                )
                energy_sep = (
                    f(other, "ci95_high_device_energy_op_j")
                    <= f(candidate, "ci95_low_device_energy_op_j")
                )
                strict = (
                    f(other, "ci95_high_time_e2e_op_s")
                    < f(candidate, "ci95_low_time_e2e_op_s")
                    or f(other, "ci95_high_device_energy_op_j")
                    < f(candidate, "ci95_low_device_energy_op_j")
                )
                if time_sep and energy_sep and strict:
                    dominators.append(other)
            item = dict(candidate)
            item["robustly_dominated_by_count"] = len(dominators)
            item["uncertainty_pareto"] = "yes" if not dominators else "no"
            pareto_new.append(item)
    write_csv(a.pareto, pareto_new, list(pareto_new[0].keys()))

    throttle = Counter(
        (row["platform"], row["throttle_reasons_hex"], row["throttle_labels"])
        for row in normalized
    )
    throttle_rows = [
        {
            "platform": platform,
            "throttle_reasons_hex": mask,
            "throttle_labels": labels,
            "rows": count,
        }
        for (platform, mask, labels), count in sorted(throttle.items())
    ]
    write_csv(
        out / "axpy_throttle_summary.csv",
        throttle_rows,
        list(throttle_rows[0].keys()),
    )

    report = [
        "# AXPY – Robustheit und Sensitivität",
        "",
        "Die Analyse trennt nun zwei unterschiedliche Session-Fragen: "
        "`fixed_config_session_support` hält die global ausgewählte Threadkonfiguration "
        "fest; `oracle_envelope_session_support` darf die Threadzahl in jeder Session "
        "neu wählen. Die primäre robuste Winner-Aussage verwendet die feste Konfiguration.",
        "",
        "`directionally_robust` bedeutet: Punktgewinner, 5/5 Sessions mit fester "
        "Konfiguration und konservativ getrennte Bootstrap-Intervalle. "
        "`fully_robust` verlangt zusätzlich 5/5 Konsistenz des per-Session-Oracle-"
        "Envelopes und CV ≤ 10 %. Damit bleiben ein einzelner alternativer CPU-"
        "Betriebspunkt oder eine instabile Effektgröße sichtbar.",
        "",
        "| N | Laufzeit: Punkt → Richtung → voll | Energie: Punkt → Richtung → voll | EDP: Punkt → Richtung → voll | In-Range 5×10 |",
        "|---:|---|---|---|---|",
    ]
    for row in robust_rows:
        agree = all(row[f"{metric}_in_range_primary_agrees"] == "yes" for metric in METRICS)
        report.append(
            f"| {row['problem_size']} | "
            f"{display_set(row['runtime_point_winners'])} → "
            f"{display_set(row['runtime_directionally_robust_winners'])} → "
            f"{display_set(row['runtime_fully_robust_winners'])} | "
            f"{display_set(row['energy_point_winners'])} → "
            f"{display_set(row['energy_directionally_robust_winners'])} → "
            f"{display_set(row['energy_fully_robust_winners'])} | "
            f"{display_set(row['edp_point_winners'])} → "
            f"{display_set(row['edp_directionally_robust_winners'])} → "
            f"{display_set(row['edp_fully_robust_winners'])} | "
            f"{'gleich' if agree else 'abweichend'} |"
        )

    report += [
        "",
        "## GPU-Throttle-Masken",
        "",
        "| Plattform | Maske | Dekodierung | Zeilen |",
        "|---|---|---|---:|",
    ]
    for row in throttle_rows:
        if row["platform"] in {"3090", "5060ti"}:
            report.append(
                f"| {row['platform']} | `{row['throttle_reasons_hex']}` | "
                f"{row['throttle_labels']} | {row['rows']} |"
            )

    primary_disagreements = [
        row for row in sensitivity if row["primary_same_winner_set"] == "no"
    ]
    report += [
        "",
        "## Einordnung",
        "",
        f"- In-Range-5×10-Sensitivitätsabweichungen: **{len(primary_disagreements)}** "
        f"von {len(sensitivity)} Metrik×Größe-Fällen.",
        f"- Eindeutige unvollständige In-Range-Konfigurationen: **{len(partial_config_keys)}**; "
        "sie sind aus der primären 5×10-Analyse ausgeschlossen und nur in der "
        "Any-Coverage-Sensitivität sichtbar.",
        "- `directionally_robust=yes`, aber `fully_robust=no` bedeutet: Richtung des "
        "Gewinners stabil, genaue Effektgröße jedoch variabel.",
        "- Bootstrap-Intervalle beruhen auf fünf Session-Zusammenfassungen und sind "
        "Unsicherheitsintervalle, kein alleiniger Signifikanznachweis.",
    ]
    (out / "axpy_robustness_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )

    comparison = a.comparison_report.read_text(encoding="utf-8")
    marker = "\n## Robustheitsstatus der Gewinner\n"
    if marker in comparison:
        comparison = comparison.split(marker, 1)[0].rstrip() + "\n"
    comparison += (
        marker
        + "\nWinner Counts oberhalb sind Punktschätzer. Die primäre Session-Aussage "
        "hält die ausgewählte Threadkonfiguration fest. Richtungssicherheit und "
        "Stabilität der Effektgröße stehen getrennt in `axpy_robustness_report.md`.\n"
    )
    a.comparison_report.write_text(comparison, encoding="utf-8")

    robustness_warnings = len(primary_disagreements) + len(partial_config_keys) + directional_gap_total
    status = "PASS_WITH_WARNINGS" if robustness_warnings else "PASS"
    write_json(
        out / "axpy_robustness_complete.json",
        {
            "status": status,
            "sizes": len(robust_rows),
            "observed_sessions": observed_sessions,
            "required_sessions": expected_sessions,
            "required_repetitions": a.required_repetitions,
            "in_range_primary_disagreements": len(primary_disagreements),
            "partial_in_range_configurations": len(partial_config_keys),
            "directional_gap_cases": directional_gap_total,
            "warnings": robustness_warnings,
        },
    )
    print(
        f"AXPY ROBUSTHEIT: {status}; Größen={len(robust_rows)}; "
        f"In-Range-Abweichungen={len(primary_disagreements)}; "
        f"partial={len(partial_config_keys)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

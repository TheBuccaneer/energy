#!/usr/bin/env python3
"""
Vergleicht alle AXPY-Plattformen auf Basis der session-bewussten Summary.

Erzeugt:
- globale Pareto-Klassifikation aller CPU-Threadpunkte und GPU-Punkte
- Plattform-Envelopes pro Problemgröße
- Laufzeit-, Energie- und EDP-Penalties relativ zum jeweiligen Besten
- Markdown-Bericht
- optionale PNG-Abbildungen
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path

from axpy_analysis_common import (
    PLATFORMS,
    SIZES,
    format_float,
    read_csv,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AXPY: Vergleich, Pareto- und Crossover-Auswertung aller Plattformen."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("new/analyse/AXPY/all_platforms/axpy_config_summary.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("new/analyse/AXPY/all_platforms"),
    )
    parser.add_argument(
        "--tie-percent",
        type=float,
        default=2.0,
        help="Totzone für Gewinner-/Klassifikationsaussagen (Standard 2%%).",
    )
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def f(row: dict, key: str) -> float:
    return float(row[key])


def i(row: dict, key: str) -> int:
    return int(float(row[key]))


def dominates(a: dict, b: dict) -> bool:
    ta = f(a, "median_time_e2e_op_s")
    ea = f(a, "median_device_energy_op_j")
    tb = f(b, "median_time_e2e_op_s")
    eb = f(b, "median_device_energy_op_j")
    return ta <= tb and ea <= eb and (ta < tb or ea < eb)


def classify_point(
    row: dict,
    *,
    min_time: float,
    min_energy: float,
    pareto: bool,
    tie_fraction: float,
) -> str:
    near_time = f(row, "median_time_e2e_op_s") <= min_time * (1.0 + tie_fraction)
    near_energy = (
        f(row, "median_device_energy_op_j")
        <= min_energy * (1.0 + tie_fraction)
    )
    if near_time and near_energy:
        return "joint_winner"
    if near_time:
        return "runtime_winner"
    if near_energy:
        return "energy_winner"
    if pareto:
        return "pareto_tradeoff"
    return "dominated"


def best_row(rows: list[dict], metric: str) -> dict:
    return min(
        rows,
        key=lambda row: (
            f(row, metric),
            f(row, "median_device_energy_op_j"),
            f(row, "median_time_e2e_op_s"),
            i(row, "threads"),
        ),
    )


def size_label(n: int) -> str:
    return f"{n // 1_000_000}M"


def plot_lines(
    output: Path,
    envelopes: list[dict],
    value_key: str,
    ci_low_key: str,
    ci_high_key: str,
    ylabel: str,
    filename: str,
) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    fig, ax = plt.subplots()
    for spec in PLATFORMS:
        rows = sorted(
            [row for row in envelopes if row["platform"] == spec.key],
            key=lambda row: int(row["problem_size"]),
        )
        if not rows:
            continue
        x = [int(row["problem_size"]) for row in rows]
        y = [float(row[value_key]) for row in rows]
        low = [float(row[ci_low_key]) for row in rows]
        high = [float(row[ci_high_key]) for row in rows]
        yerr = [
            [max(0.0, value - lower) for value, lower in zip(y, low)],
            [max(0.0, upper - value) for value, upper in zip(y, high)],
        ]
        ax.errorbar(x, y, yerr=yerr, marker="o", label=spec.label, capsize=2)

    ax.set_xscale("log", base=2)
    ax.set_xticks(list(SIZES))
    ax.set_xticklabels([size_label(n) for n in SIZES])
    ax.set_yscale("log")
    ax.set_xlabel("Problemgröße N")
    ax.set_ylabel(ylabel)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / filename, dpi=180)
    plt.close(fig)
    return True


def plot_pareto_by_size(output: Path, pareto_rows: list[dict]) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    created = False
    for n in SIZES:
        rows = [row for row in pareto_rows if int(row["problem_size"]) == n]
        if not rows:
            continue
        fig, ax = plt.subplots()
        for spec in PLATFORMS:
            points = [row for row in rows if row["platform"] == spec.key]
            if not points:
                continue
            x = [float(row["median_time_e2e_op_s"]) * 1000.0 for row in points]
            y = [float(row["median_device_energy_op_j"]) for row in points]
            ax.scatter(x, y, label=spec.label)
            for row, x_value, y_value in zip(points, x, y):
                if row["pareto"] == "yes":
                    label = (
                        f"T={row['threads']}" if int(row["threads"]) >= 0 else "GPU"
                    )
                    ax.annotate(label, (x_value, y_value))

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("E2E-Zeit pro Operation [ms]")
        ax.set_ylabel("Device-Domain-Energie pro Operation [J]")
        ax.set_title(f"AXPY Pareto-Punkte, N={size_label(n)}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / f"axpy_pareto_N{n}.png", dpi=180)
        plt.close(fig)
        created = True
    return created


def main() -> int:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    if not input_path.is_file():
        raise SystemExit(f"Input fehlt: {input_path}")

    rows = read_csv(input_path)
    required = {
        "platform", "platform_label", "kind", "campaign_id",
        "problem_size", "threads", "median_time_e2e_op_s",
        "median_device_energy_op_j", "median_edp_device_j_s",
        "median_logical_bandwidth_gb_s", "median_gflops_per_s",
        "cv_all_rows_time_e2e_op_s", "cv_all_rows_device_energy_op_j",
        "ci95_low_time_e2e_op_s", "ci95_high_time_e2e_op_s",
        "ci95_low_device_energy_op_j", "ci95_high_device_energy_op_j",
        "ci95_low_edp_device_j_s", "ci95_high_edp_device_j_s",
    }
    missing = required - set(rows[0] if rows else {})
    if missing:
        raise SystemExit(f"Input-Spalten fehlen: {sorted(missing)}")

    tie_fraction = args.tie_percent / 100.0

    grouped_by_n: dict[int, list[dict]] = defaultdict(list)
    grouped_platform_n: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rows:
        n = i(row, "problem_size")
        grouped_by_n[n].append(row)
        grouped_platform_n[(row["platform"], n)].append(row)

    pareto_output = []
    for n, points in sorted(grouped_by_n.items()):
        min_time = min(f(row, "median_time_e2e_op_s") for row in points)
        min_energy = min(f(row, "median_device_energy_op_j") for row in points)
        for row in points:
            dominators = [candidate for candidate in points if dominates(candidate, row)]
            pareto = not dominators
            item = dict(row)
            item["pareto"] = "yes" if pareto else "no"
            item["dominated_by_count"] = len(dominators)
            item["runtime_ratio_to_global_best"] = (
                f(row, "median_time_e2e_op_s") / min_time
            )
            item["energy_ratio_to_global_best"] = (
                f(row, "median_device_energy_op_j") / min_energy
            )
            item["classification"] = classify_point(
                row,
                min_time=min_time,
                min_energy=min_energy,
                pareto=pareto,
                tie_fraction=tie_fraction,
            )
            pareto_output.append(item)

    envelope_rows = []
    for (platform, n), points in sorted(grouped_platform_n.items()):
        runtime_best = best_row(points, "median_time_e2e_op_s")
        energy_best = best_row(points, "median_device_energy_op_j")
        edp_best = best_row(points, "median_edp_device_j_s")
        envelope_rows.append(
            {
                "platform": platform,
                "platform_label": runtime_best["platform_label"],
                "kind": runtime_best["kind"],
                "problem_size": n,
                "runtime_best_threads": runtime_best["threads"],
                "runtime_best_time_s": runtime_best["median_time_e2e_op_s"],
                "runtime_best_time_ci95_low_s": runtime_best["ci95_low_time_e2e_op_s"],
                "runtime_best_time_ci95_high_s": runtime_best["ci95_high_time_e2e_op_s"],
                "runtime_best_energy_j": runtime_best["median_device_energy_op_j"],
                "runtime_best_edp_j_s": runtime_best["median_edp_device_j_s"],
                "energy_best_threads": energy_best["threads"],
                "energy_best_time_s": energy_best["median_time_e2e_op_s"],
                "energy_best_energy_j": energy_best["median_device_energy_op_j"],
                "energy_best_energy_ci95_low_j": energy_best["ci95_low_device_energy_op_j"],
                "energy_best_energy_ci95_high_j": energy_best["ci95_high_device_energy_op_j"],
                "energy_best_edp_j_s": energy_best["median_edp_device_j_s"],
                "edp_best_threads": edp_best["threads"],
                "edp_best_time_s": edp_best["median_time_e2e_op_s"],
                "edp_best_energy_j": edp_best["median_device_energy_op_j"],
                "edp_best_edp_j_s": edp_best["median_edp_device_j_s"],
                "edp_best_edp_ci95_low_j_s": edp_best["ci95_low_edp_device_j_s"],
                "edp_best_edp_ci95_high_j_s": edp_best["ci95_high_edp_device_j_s"],
                "edp_best_bandwidth_gb_s": edp_best[
                    "median_logical_bandwidth_gb_s"
                ],
                "edp_best_gflops_per_s": edp_best["median_gflops_per_s"],
            }
        )

    cross_rows = []
    penalty_rows = []
    for n in SIZES:
        points = [row for row in envelope_rows if int(row["problem_size"]) == n]
        if not points:
            continue

        min_runtime = min(float(row["runtime_best_time_s"]) for row in points)
        min_energy = min(float(row["energy_best_energy_j"]) for row in points)
        min_edp = min(float(row["edp_best_edp_j_s"]) for row in points)

        runtime_winners = [
            row["platform"] for row in points
            if float(row["runtime_best_time_s"]) <= min_runtime * (1 + tie_fraction)
        ]
        energy_winners = [
            row["platform"] for row in points
            if float(row["energy_best_energy_j"]) <= min_energy * (1 + tie_fraction)
        ]
        edp_winners = [
            row["platform"] for row in points
            if float(row["edp_best_edp_j_s"]) <= min_edp * (1 + tie_fraction)
        ]

        global_pareto = [
            row for row in pareto_output
            if int(row["problem_size"]) == n and row["pareto"] == "yes"
        ]
        cross_rows.append(
            {
                "problem_size": n,
                "runtime_winners": "|".join(runtime_winners),
                "energy_winners": "|".join(energy_winners),
                "edp_winners": "|".join(edp_winners),
                "best_runtime_s": min_runtime,
                "best_device_energy_j": min_energy,
                "best_edp_j_s": min_edp,
                "global_pareto_points": len(global_pareto),
                "global_pareto_platforms": "|".join(
                    sorted({row["platform"] for row in global_pareto})
                ),
            }
        )

        for row in points:
            penalty_rows.append(
                {
                    "problem_size": n,
                    "platform": row["platform"],
                    "platform_label": row["platform_label"],
                    "runtime_best_threads": row["runtime_best_threads"],
                    "runtime_best_time_s": row["runtime_best_time_s"],
                    "runtime_penalty_vs_best": (
                        float(row["runtime_best_time_s"]) / min_runtime
                    ),
                    "energy_best_threads": row["energy_best_threads"],
                    "energy_best_energy_j": row["energy_best_energy_j"],
                    "energy_penalty_vs_best": (
                        float(row["energy_best_energy_j"]) / min_energy
                    ),
                    "edp_best_threads": row["edp_best_threads"],
                    "edp_best_edp_j_s": row["edp_best_edp_j_s"],
                    "edp_penalty_vs_best": (
                        float(row["edp_best_edp_j_s"]) / min_edp
                    ),
                }
            )

    write_csv(
        output / "axpy_global_pareto.csv",
        pareto_output,
        list(pareto_output[0].keys()) if pareto_output else [],
    )
    write_csv(
        output / "axpy_platform_envelopes.csv",
        envelope_rows,
        list(envelope_rows[0].keys()) if envelope_rows else [],
    )
    write_csv(
        output / "axpy_cross_platform_by_size.csv",
        cross_rows,
        list(cross_rows[0].keys()) if cross_rows else [],
    )
    write_csv(
        output / "axpy_platform_penalties.csv",
        penalty_rows,
        list(penalty_rows[0].keys()) if penalty_rows else [],
    )

    runtime_win_counts = Counter()
    energy_win_counts = Counter()
    edp_win_counts = Counter()
    for row in cross_rows:
        runtime_win_counts.update(filter(None, row["runtime_winners"].split("|")))
        energy_win_counts.update(filter(None, row["energy_winners"].split("|")))
        edp_win_counts.update(filter(None, row["edp_winners"].split("|")))

    pareto_counts = Counter(
        row["platform"] for row in pareto_output if row["pareto"] == "yes"
    )
    classification_counts = Counter(row["classification"] for row in pareto_output)

    unstable = sorted(
        rows,
        key=lambda row: max(
            f(row, "cv_all_rows_time_e2e_op_s"),
            f(row, "cv_all_rows_device_energy_op_j"),
        ),
        reverse=True,
    )[:12]

    report = [
        "# AXPY – Vergleich aller Plattformen",
        "",
        "## Methodik",
        "",
        "Die Vergleichspunkte stammen aus dem Median der Session-Mediane. "
        "Für CPUs bleibt jede Threadkonfiguration zunächst ein eigener Punkt; "
        "die GPUs besitzen je Größe einen Punkt. Die globale Pareto-Analyse "
        "verwendet E2E-Zeit und Device-Domain-Energie pro Operation.",
        "",
        f"Gewinnergleichstände verwenden eine Totzone von ±{args.tie_percent:.2f} %. "
        "Die Plattform-Envelopes wählen pro Größe separat die schnellste, "
        "energieärmste und EDP-minimale Konfiguration einer Plattform. "
        "Die Linienabbildungen zeigen Bootstrap-Unsicherheitsbalken und beschriften "
        "die tatsächlich gemessenen Dezimalgrößen 1M bis 256M.",
        "",
        "## Punktgewinner über die neun Problemgrößen",
        "",
        "| Kriterium | Intel | AMD | RTX 3090 | RTX 5060 Ti |",
        "|---|---:|---:|---:|---:|",
        f"| Laufzeit-Envelope | {runtime_win_counts['intel']} | "
        f"{runtime_win_counts['amd']} | {runtime_win_counts['3090']} | "
        f"{runtime_win_counts['5060ti']} |",
        f"| Energie-Envelope | {energy_win_counts['intel']} | "
        f"{energy_win_counts['amd']} | {energy_win_counts['3090']} | "
        f"{energy_win_counts['5060ti']} |",
        f"| EDP-Envelope | {edp_win_counts['intel']} | "
        f"{edp_win_counts['amd']} | {edp_win_counts['3090']} | "
        f"{edp_win_counts['5060ti']} |",
        "",
        "Mehrfachzählungen sind bei Gleichständen innerhalb der Totzone möglich. Diese Tabelle verwendet nur Punktschätzer; die nachgeschaltete Robustheitsanalyse prüft Session-Konsistenz, Streuung, Intervalle und In-Range-Sensitivität.",
        "",
        "## Globale Pareto-Struktur",
        "",
        "| Plattform | Zahl globaler Pareto-Punkte |",
        "|---|---:|",
    ]
    labels = {spec.key: spec.label for spec in PLATFORMS}
    for spec in PLATFORMS:
        report.append(f"| {spec.label} | {pareto_counts[spec.key]} |")

    report += [
        "",
        "Klassifikationen aller Plattform-/Thread-/Größenpunkte:",
        "",
    ]
    for name in (
        "joint_winner", "runtime_winner", "energy_winner",
        "pareto_tradeoff", "dominated",
    ):
        report.append(f"- `{name}`: {classification_counts[name]}")

    report += [
        "",
        "## Auffälligste Streuungen",
        "",
        "| Plattform | N | Threads | CV Zeit | CV Energie |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in unstable:
        report.append(
            f"| {row['platform_label']} | {i(row, 'problem_size')} | "
            f"{i(row, 'threads')} | "
            f"{100*f(row, 'cv_all_rows_time_e2e_op_s'):.2f} % | "
            f"{100*f(row, 'cv_all_rows_device_energy_op_j'):.2f} % |"
        )

    report += [
        "",
        "## Wissenschaftliche Interpretation",
        "",
        "- `device_energy` bedeutet auf CPUs RAPL-Package und auf GPUs "
        "NVML-Board. Der Vergleich ist deshalb kein vollständiger "
        "Systemenergievergleich. Bei AXPY begünstigt die fehlende externe "
        "CPU-DRAM-Energie tendenziell die CPU-Seite.",
        "- Runtime-, Energie- und EDP-Envelopes dürfen nicht als derselbe "
        "Betriebspunkt interpretiert werden: Eine Plattform kann für jedes "
        "Kriterium eine andere Threadzahl wählen.",
        "- Die globale Pareto-Tabelle ist deshalb die zentrale Datei für "
        "Trade-off- und Crossover-Aussagen.",
        "- Winner Counts sind deskriptive Orientierung. Die spätere "
        "Paper-Analyse sollte Session-Unsicherheit, Effektgrößen und "
        "Crossover-Stabilität explizit modellieren.",
        "",
        "## Ergebnisdateien",
        "",
        "- `axpy_global_pareto.csv`: alle Konfigurationen mit Pareto-Status",
        "- `axpy_platform_envelopes.csv`: beste Laufzeit/Energie/EDP je Plattform und N",
        "- `axpy_cross_platform_by_size.csv`: Gewinner je Problemgröße",
        "- `axpy_platform_penalties.csv`: Penalty-Faktoren relativ zum Besten",
    ]

    plots_created = False
    if not args.no_plots:
        plots_created |= plot_lines(
            output, envelope_rows, "runtime_best_time_s",
            "runtime_best_time_ci95_low_s", "runtime_best_time_ci95_high_s",
            "Beste E2E-Zeit pro Operation [s]",
            "axpy_best_runtime_vs_size.png",
        )
        plots_created |= plot_lines(
            output, envelope_rows, "energy_best_energy_j",
            "energy_best_energy_ci95_low_j", "energy_best_energy_ci95_high_j",
            "Beste Device-Domain-Energie pro Operation [J]",
            "axpy_best_energy_vs_size.png",
        )
        plots_created |= plot_lines(
            output, envelope_rows, "edp_best_edp_j_s",
            "edp_best_edp_ci95_low_j_s", "edp_best_edp_ci95_high_j_s",
            "Bestes EDP [J·s]",
            "axpy_best_edp_vs_size.png",
        )
        plots_created |= plot_pareto_by_size(output, pareto_output)

    report += [
        "",
        "Abbildungen: " + (
            "erstellt" if plots_created
            else "nicht erstellt (matplotlib fehlt oder --no-plots gesetzt)"
        ),
    ]

    (output / "axpy_comparison_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )

    print("=" * 76)
    print("AXPY VERGLEICH ABGESCHLOSSEN")
    print(f"Konfigurationspunkte: {len(rows)}")
    print(f"Pareto-Punkte:        {sum(pareto_counts.values())}")
    print(f"Ausgabe:              {output}")
    print("=" * 76)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

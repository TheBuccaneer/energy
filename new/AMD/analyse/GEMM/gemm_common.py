#!/usr/bin/env python3
"""Shared helpers for the CPU GEMM audit and analysis pipeline."""
from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

SIZES = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
THREADS = {
    "INTEL": [1, 2, 4, 8, 10, 16, 20],
    "AMD": [1, 2, 4, 8, 10, 16, 20, 32, 64],
}
SLUG = {"INTEL": "intel", "AMD": "amd"}
EXPECTED_SESSIONS = 5
EXPECTED_REPETITIONS = 10
TARGET_LOW_S = 0.75
TARGET_HIGH_S = 1.25
PRACTICAL_TOLERANCE = 0.02

EXPECTED_COLUMNS = [
    "schema_version", "timestamp", "session_id", "sequence_index",
    "run_id_global", "repetition", "workload", "implementation",
    "execution_mode", "device_name", "num_threads", "problem_size",
    "problem_spec", "batches", "e2e_time_s", "kernel_time_s",
    "wall_time_s", "device_energy_j", "total_energy_j", "dram_energy_j",
    "energy_per_op_j", "energy_per_second_j", "energy_per_flop_j",
    "time_per_op_ms_kernel", "time_per_op_ms_e2e", "flops_total",
    "gflops_per_s", "logical_bytes_per_op", "avg_power_w",
    "runtime_status", "pcie_gen", "pcie_width", "sm_clock_mhz",
    "clock_before_mhz", "clock_after_mhz", "mem_clock_mhz", "temp_c",
    "temp_before_c", "temp_after_c", "throttle_reasons", "cpu_cycles",
    "cpu_instructions", "cpu_ipc", "cpu_cache_misses", "checksum_ok",
]

NUMERIC_COLUMNS = [
    "sequence_index", "run_id_global", "repetition", "num_threads",
    "problem_size", "batches", "e2e_time_s", "kernel_time_s",
    "wall_time_s", "device_energy_j", "total_energy_j", "dram_energy_j",
    "energy_per_op_j", "energy_per_second_j", "energy_per_flop_j",
    "time_per_op_ms_kernel", "time_per_op_ms_e2e", "flops_total",
    "gflops_per_s", "logical_bytes_per_op", "avg_power_w", "sm_clock_mhz",
    "clock_before_mhz", "clock_after_mhz", "temp_c", "temp_before_c",
    "temp_after_c",
]

@dataclass
class Campaign:
    platform: str
    stamp: str
    files: list[Path]
    sessions: list[int]
    data: pd.DataFrame


def context(script_file: str | Path) -> tuple[Path, str, Path, Path]:
    """Return project root, platform, run directory, and result directory."""
    script = Path(script_file).resolve()
    platform = script.parents[2].name.upper()
    if platform not in THREADS:
        raise RuntimeError(f"Script must live under AMD/analyse/GEMM or INTEL/analyse/GEMM: {script}")
    root = script.parents[3]
    run_dir = root / platform / "runs" / "GEMM"
    result_dir = root / platform / "results" / "GEMM"
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "figures").mkdir(exist_ok=True)
    return root, platform, run_dir, result_dir


def parse_args(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--campaign",
        help="Timestamp der Kampagne, z. B. 20260719_085511. Standard: neueste vollständige Kampagne.",
    )
    return parser.parse_args()


def discover_campaigns(run_dir: Path, platform: str) -> dict[str, list[tuple[int, Path]]]:
    pattern = re.compile(
        rf"^gemm_{SLUG[platform]}_(\d{{8}}_\d{{6}})_session(\d+)\.csv$",
        re.IGNORECASE,
    )
    groups: dict[str, list[tuple[int, Path]]] = {}
    for path in sorted(run_dir.glob("*.csv")):
        match = pattern.match(path.name)
        if match:
            groups.setdefault(match.group(1), []).append((int(match.group(2)), path))
    return groups


def select_campaign(run_dir: Path, platform: str, requested: str | None) -> tuple[str, list[tuple[int, Path]]]:
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run-Verzeichnis fehlt: {run_dir}")
    groups = discover_campaigns(run_dir, platform)
    if not groups:
        raise FileNotFoundError(f"Keine GEMM-CSV für {platform} in {run_dir}")
    if requested:
        if requested not in groups:
            raise ValueError(f"Kampagne {requested} fehlt. Verfügbar: {', '.join(sorted(groups))}")
        return requested, sorted(groups[requested])

    complete = {
        stamp: entries for stamp, entries in groups.items()
        if sorted(s for s, _ in entries) == list(range(1, EXPECTED_SESSIONS + 1))
    }
    pool = complete if complete else groups
    stamp = max(pool)
    return stamp, sorted(pool[stamp])


def normalize_bool(series: pd.Series) -> pd.Series:
    truth = {"1", "true", "t", "yes", "y"}
    return series.astype(str).str.strip().str.lower().isin(truth)


def load_campaign(run_dir: Path, platform: str, requested: str | None = None) -> Campaign:
    stamp, entries = select_campaign(run_dir, platform, requested)
    frames: list[pd.DataFrame] = []
    sessions: list[int] = []
    files: list[Path] = []
    for session, path in entries:
        frame = pd.read_csv(path)
        frame["source_file"] = path.name
        frame["session_number"] = session
        frames.append(frame)
        sessions.append(session)
        files.append(path)
    data = pd.concat(frames, ignore_index=True, sort=False)
    for column in NUMERIC_COLUMNS:
        if column in data:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    if "checksum_ok" in data:
        data["checksum_bool"] = normalize_bool(data["checksum_ok"])
    else:
        data["checksum_bool"] = False
    return Campaign(platform, stamp, files, sessions, data)


def add_derived(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()
    safe_batches = df["batches"].replace(0, np.nan)
    safe_time = df["e2e_time_s"].replace(0, np.nan)
    df["package_energy_per_op_j"] = df["device_energy_j"] / safe_batches
    df["package_avg_power_w"] = df["device_energy_j"] / safe_time
    df["runtime_per_op_s"] = df["e2e_time_s"] / safe_batches
    df["edp_package_j_s"] = df["package_energy_per_op_j"] * df["runtime_per_op_s"]
    df["temperature_rise_c"] = df["temp_after_c"] - df["temp_before_c"]
    df["clock_change_pct"] = 100.0 * (
        df["clock_after_mhz"] - df["clock_before_mhz"]
    ) / df["clock_before_mhz"].replace(0, np.nan)
    return df


def relative_close(actual: pd.Series, expected: pd.Series, rtol: float, atol: float = 1e-12) -> pd.Series:
    a = pd.to_numeric(actual, errors="coerce").to_numpy(dtype=float)
    e = pd.to_numeric(expected, errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(a) & np.isfinite(e) & (np.abs(a - e) <= atol + rtol * np.abs(e))
    return pd.Series(ok, index=actual.index)


def robust_cv(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return math.nan
    med = float(np.median(arr))
    if med == 0:
        return math.nan
    mad = float(np.median(np.abs(arr - med)))
    return 1.4826 * mad / abs(med)


def standard_cv(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2 or np.mean(arr) == 0:
        return math.nan
    return float(np.std(arr, ddof=1) / abs(np.mean(arr)))


def robust_outlier_mask(values: pd.Series, threshold: float = 3.5) -> pd.Series:
    x = pd.to_numeric(values, errors="coerce")
    med = x.median()
    mad = (x - med).abs().median()
    if not np.isfinite(mad) or mad == 0:
        return pd.Series(False, index=values.index)
    score = (x - med).abs() / (1.4826 * mad)
    return score > threshold


def session_medians(data: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "runtime_per_op_s", "time_per_op_ms_e2e", "package_energy_per_op_j",
        "package_avg_power_w", "gflops_per_s", "edp_package_j_s", "temp_c",
        "sm_clock_mhz", "batches",
    ]
    available = [m for m in metrics if m in data]
    return (
        data.groupby(["session_number", "problem_size", "num_threads"], as_index=False)[available]
        .median(numeric_only=True)
    )


def bootstrap_median_ci(values: Iterable[float], seed: int = 20260719, draws: int = 10000) -> tuple[float, float, float]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return math.nan, math.nan, math.nan
    point = float(np.median(arr))
    if arr.size == 1:
        return point, point, point
    if arr.size <= 5:
        # Exact non-parametric bootstrap: all n^n ordered resamples.
        grids = np.indices((arr.size,) * arr.size).reshape(arr.size, -1).T
        medians = np.median(arr[grids], axis=1)
    else:
        rng = np.random.default_rng(seed)
        sampled = rng.choice(arr, size=(draws, arr.size), replace=True)
        medians = np.median(sampled, axis=1)
    low, high = np.quantile(medians, [0.025, 0.975])
    return point, float(low), float(high)


def campaign_summary(data: pd.DataFrame) -> pd.DataFrame:
    sessions = session_medians(data)
    rows: list[dict] = []
    metrics = [
        "runtime_per_op_s", "time_per_op_ms_e2e", "package_energy_per_op_j",
        "package_avg_power_w", "gflops_per_s", "edp_package_j_s", "temp_c",
    ]
    for (n, threads), group in sessions.groupby(["problem_size", "num_threads"]):
        row = {"problem_size": int(n), "num_threads": int(threads), "sessions": int(len(group))}
        raw = data[(data["problem_size"] == n) & (data["num_threads"] == threads)]
        row["runs"] = int(len(raw))
        for metric in metrics:
            if metric not in group:
                continue
            point, low, high = bootstrap_median_ci(group[metric], seed=int(n) + int(threads))
            row[f"{metric}_median"] = point
            row[f"{metric}_ci95_low"] = low
            row[f"{metric}_ci95_high"] = high
            row[f"{metric}_between_session_robust_cv"] = robust_cv(group[metric])
            row[f"{metric}_run_robust_cv"] = robust_cv(raw[metric])
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["problem_size", "num_threads"]).reset_index(drop=True)



def practical_pareto_mask(group: pd.DataFrame, tolerance: float = PRACTICAL_TOLERANCE) -> pd.Series:
    """Return a practical Pareto frontier using a relative indifference band."""
    energy = group["package_energy_per_op_j_median"].to_numpy(float)
    runtime = group["runtime_per_op_s_median"].to_numpy(float)
    keep = np.ones(len(group), dtype=bool)
    for i in range(len(group)):
        no_worse = (energy <= energy[i] * (1 + tolerance)) & (runtime <= runtime[i] * (1 + tolerance))
        materially_better = (energy < energy[i] * (1 - tolerance)) | (runtime < runtime[i] * (1 - tolerance))
        dominated = no_worse & materially_better
        dominated[i] = False
        keep[i] = not dominated.any()
    return pd.Series(keep, index=group.index)


def metric_leaders(summary: pd.DataFrame, metric: str,
                   tolerance: float = PRACTICAL_TOLERANCE) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Annotate exact minima without overstating practically/statistically unresolved ties.

    A unique clear leader requires both a >tolerance median gap and non-overlapping
    95% bootstrap intervals against every competitor. Candidate sets include any
    configuration within the practical tolerance or whose interval overlaps the
    exact minimum's interval.
    """
    base_metric = metric[:-7] if metric.endswith("_median") else metric
    low_col = f"{base_metric}_ci95_low"
    high_col = f"{base_metric}_ci95_high"
    leader_rows: list[dict] = []
    candidate_rows: list[dict] = []

    for n, group in summary.groupby("problem_size", sort=True):
        ordered = group.sort_values([metric, "num_threads"]).copy()
        best = ordered.iloc[0]
        best_value = float(best[metric])
        best_low = float(best[low_col])
        best_high = float(best[high_col])
        others = ordered.iloc[1:]

        practical_separation = bool((others[metric] > best_value * (1 + tolerance)).all())
        ci_separation = bool((best_high < others[low_col]).all())
        clear = practical_separation and ci_separation

        overlap = (ordered[low_col] <= best_high) & (ordered[high_col] >= best_low)
        within = ordered[metric] <= best_value * (1 + tolerance)
        candidates = ordered[within | overlap].copy()
        leader_threads = ",".join(str(int(x)) for x in candidates["num_threads"])

        record = best.to_dict()
        record.update({
            "exact_min_threads": int(best["num_threads"]),
            "selection_status": "clear_leader" if clear else "tie_or_uncertain",
            "leader_threads": leader_threads,
            "leader_count": int(len(candidates)),
            "practical_gap_gt_tolerance": practical_separation,
            "ci_separated_from_all": ci_separation,
            "practical_tolerance_pct": 100 * tolerance,
        })
        if len(others):
            second = float(others.iloc[0][metric])
            record["gap_to_second_pct"] = 100 * (second / best_value - 1)
        else:
            record["gap_to_second_pct"] = math.nan
        leader_rows.append(record)

        for _, row in ordered.iterrows():
            candidate_rows.append({
                "problem_size": int(n),
                "metric": metric,
                "num_threads": int(row["num_threads"]),
                "value": float(row[metric]),
                "ci95_low": float(row[low_col]),
                "ci95_high": float(row[high_col]),
                "relative_to_min_pct": 100 * (float(row[metric]) / best_value - 1),
                "within_practical_tolerance": bool(row[metric] <= best_value * (1 + tolerance)),
                "ci_overlaps_exact_min": bool(row[low_col] <= best_high and row[high_col] >= best_low),
                "in_leader_set": bool(row[metric] <= best_value * (1 + tolerance) or
                                      (row[low_col] <= best_high and row[high_col] >= best_low)),
                "exact_minimum": int(row["num_threads"]) == int(best["num_threads"]),
                "selection_status": "clear_leader" if clear else "tie_or_uncertain",
            })

    return pd.DataFrame(leader_rows), pd.DataFrame(candidate_rows)

def markdown_table(frame: pd.DataFrame, max_rows: int = 30) -> str:
    if frame.empty:
        return "_Keine Einträge._"
    return frame.head(max_rows).to_markdown(index=False)


def write_text(path: Path, text: str) -> None:
    path.write_text(text.rstrip() + "\n", encoding="utf-8")

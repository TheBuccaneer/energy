#!/usr/bin/env python3
"""Shared helpers for the CPU STRIDED_GEMM audit and analysis pipeline."""
from __future__ import annotations

import argparse
import itertools
import math
from functools import lru_cache
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

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
MAX_BATCHES = 10_000_000
PRACTICAL_TOLERANCE = 0.02

EXPECTED_COLUMNS = [
    "schema_version", "timestamp", "session_id", "sequence_index",
    "run_id_global", "repetition", "workload", "implementation",
    "execution_mode", "device_name", "num_threads", "problem_size",
    "problem_spec", "batches", "e2e_time_s", "kernel_time_s",
    "wall_time_s", "total_energy_j", "device_energy_j", "dram_energy_j",
    "energy_per_op_j", "energy_per_second_j", "energy_per_flop_j",
    "time_per_op_ms_kernel", "time_per_op_ms_e2e", "flops_total",
    "gflops_per_s", "logical_bytes_per_op", "avg_power_w",
    "runtime_status", "pcie_gen", "pcie_width", "sm_clock_mhz",
    "clock_before_mhz", "clock_after_mhz", "mem_clock_mhz", "temp_c",
    "temp_before_c", "temp_after_c", "throttle_reasons", "cpu_cycles",
    "cpu_instructions", "cpu_ipc", "cpu_cache_misses", "checksum_ok",
]
# Some GPU-derived CPU writers historically used device_energy_j before total_energy_j.
ALTERNATE_COLUMNS = EXPECTED_COLUMNS.copy()
i, j = ALTERNATE_COLUMNS.index("total_energy_j"), ALTERNATE_COLUMNS.index("device_energy_j")
ALTERNATE_COLUMNS[i], ALTERNATE_COLUMNS[j] = ALTERNATE_COLUMNS[j], ALTERNATE_COLUMNS[i]

NUMERIC_COLUMNS = [
    "sequence_index", "run_id_global", "repetition", "num_threads",
    "problem_size", "batches", "e2e_time_s", "kernel_time_s",
    "wall_time_s", "total_energy_j", "device_energy_j", "dram_energy_j",
    "energy_per_op_j", "energy_per_second_j", "energy_per_flop_j",
    "time_per_op_ms_kernel", "time_per_op_ms_e2e", "flops_total",
    "gflops_per_s", "logical_bytes_per_op", "avg_power_w", "pcie_gen",
    "pcie_width", "sm_clock_mhz", "clock_before_mhz", "clock_after_mhz",
    "mem_clock_mhz", "temp_c", "temp_before_c", "temp_after_c",
    "cpu_cycles", "cpu_instructions", "cpu_ipc", "cpu_cache_misses",
]

PRIMARY_ENERGY_METRIC = "package_energy_per_op_j"  # CPU package RAPL; cross-platform primary
PACKAGE_ENERGY_METRIC = "package_energy_per_op_j"

@dataclass
class Campaign:
    platform: str
    workload: str
    stamp: str
    files: list[Path]
    sessions: list[int]
    data: pd.DataFrame


def context(script_file: str | Path) -> tuple[Path, str, Path, Path]:
    script = Path(script_file).resolve()
    platform = script.parents[2].name.upper()
    if platform not in THREADS:
        raise RuntimeError(
            f"Script must live under AMD/analyse/STRIDED_GEMM or "
            f"INTEL/analyse/STRIDED_GEMM: {script}"
        )
    root = script.parents[3]
    run_dir = root / platform / "runs" / "STRIDED_GEMM"
    result_dir = root / platform / "results" / "STRIDED_GEMM"
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "figures").mkdir(exist_ok=True)
    return root, platform, run_dir, result_dir


def parse_args(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--campaign",
        help="Campaign timestamp, e.g. 20260719_175453. Default: newest complete campaign.",
    )
    return parser.parse_args()


def campaign_pattern(platform: str, workload: str = "STRIDED_GEMM") -> re.Pattern[str]:
    if workload == "STRIDED_GEMM":
        prefix = f"strided_gemm_{SLUG[platform]}"
    elif workload == "GEMM":
        prefix = f"gemm_{SLUG[platform]}"
    else:
        raise ValueError(workload)
    return re.compile(rf"^{prefix}_(\d{{8}}_\d{{6}})_session(\d+)\.csv$", re.I)


def discover_campaigns(run_dir: Path, platform: str,
                       workload: str = "STRIDED_GEMM") -> dict[str, list[tuple[int, Path]]]:
    pattern = campaign_pattern(platform, workload)
    groups: dict[str, list[tuple[int, Path]]] = {}
    for path in sorted(run_dir.glob("*.csv")):
        match = pattern.match(path.name)
        if match:
            groups.setdefault(match.group(1), []).append((int(match.group(2)), path))
    return groups


def select_campaign(run_dir: Path, platform: str, requested: str | None,
                    workload: str = "STRIDED_GEMM") -> tuple[str, list[tuple[int, Path]]]:
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory missing: {run_dir}")
    groups = discover_campaigns(run_dir, platform, workload)
    if not groups:
        raise FileNotFoundError(f"No official {workload} CSV campaign for {platform} in {run_dir}")
    if requested:
        if requested not in groups:
            raise ValueError(f"Campaign {requested} missing. Available: {', '.join(sorted(groups))}")
        return requested, sorted(groups[requested])
    complete = {
        stamp: entries for stamp, entries in groups.items()
        if sorted(s for s, _ in entries) == list(range(1, EXPECTED_SESSIONS + 1))
    }
    if not complete:
        newest = max(groups)
        return newest, sorted(groups[newest])
    stamp = max(complete)
    return stamp, sorted(complete[stamp])


def normalize_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"1", "true", "t", "yes", "y"})


def load_campaign(run_dir: Path, platform: str, requested: str | None = None,
                  workload: str = "STRIDED_GEMM") -> Campaign:
    stamp, entries = select_campaign(run_dir, platform, requested, workload)
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
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    data["checksum_bool"] = normalize_bool(data["checksum_ok"]) if "checksum_ok" in data else False
    return Campaign(platform, workload, stamp, files, sessions, data)


def load_project_campaign(root: Path, platform: str, workload: str,
                          requested: str | None = None) -> Campaign:
    return load_campaign(root / platform / "runs" / workload, platform, requested, workload)


def add_derived(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()
    batches = df["batches"].replace(0, np.nan)
    seconds = df["e2e_time_s"].replace(0, np.nan)
    n = df["problem_size"].astype(float)
    flops_per_op = 2.0 * n ** 3

    df["runtime_per_op_s"] = df["e2e_time_s"] / batches
    df["kernel_runtime_per_op_s"] = df["kernel_time_s"] / batches
    df["package_energy_per_op_j"] = df["device_energy_j"] / batches
    df["dram_energy_per_op_j"] = np.where(df["dram_energy_j"] >= 0, df["dram_energy_j"] / batches, np.nan)
    df["total_energy_per_op_j"] = df["total_energy_j"] / batches
    df["package_avg_power_w"] = df["device_energy_j"] / seconds
    df["dram_avg_power_w"] = np.where(df["dram_energy_j"] >= 0, df["dram_energy_j"] / seconds, np.nan)
    df["total_avg_power_w"] = df["total_energy_j"] / seconds
    df["edp_total_j_s"] = df["total_energy_per_op_j"] * df["runtime_per_op_s"]
    df["edp_package_j_s"] = df["package_energy_per_op_j"] * df["runtime_per_op_s"]
    df["throughput_gflops_exact"] = flops_per_op / df["runtime_per_op_s"] / 1e9
    df["total_efficiency_gflop_per_j"] = flops_per_op / df["total_energy_per_op_j"] / 1e9
    df["package_efficiency_gflop_per_j"] = flops_per_op / df["package_energy_per_op_j"] / 1e9
    df["temperature_rise_c"] = df["temp_after_c"] - df["temp_before_c"]
    df["clock_change_pct"] = 100.0 * (
        df["clock_after_mhz"] - df["clock_before_mhz"]
    ) / df["clock_before_mhz"].replace(0, np.nan)
    df["logical_bytes_expected"] = 12.0 * n ** 2
    df["allocated_footprint_bytes"] = 24.0 * n ** 2
    df["padding_footprint_factor"] = 2.0
    return df


def relative_close(actual: pd.Series, expected: pd.Series | np.ndarray,
                   rtol: float, atol: float = 1e-12) -> pd.Series:
    a = pd.to_numeric(actual, errors="coerce").to_numpy(dtype=float)
    e = pd.to_numeric(pd.Series(expected, index=actual.index), errors="coerce").to_numpy(dtype=float)
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
    score = 0.67448975 * (x - med) / mad
    return score.abs() > threshold


def session_medians(data: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "runtime_per_op_s", "kernel_runtime_per_op_s", "total_energy_per_op_j",
        "package_energy_per_op_j", "dram_energy_per_op_j", "total_avg_power_w",
        "package_avg_power_w", "dram_avg_power_w", "edp_total_j_s",
        "edp_package_j_s", "throughput_gflops_exact", "total_efficiency_gflop_per_j",
        "package_efficiency_gflop_per_j", "temp_c", "clock_before_mhz",
        "clock_after_mhz", "clock_change_pct", "batches",
    ]
    available = [m for m in metrics if m in data.columns]
    return (
        data.groupby(["session_number", "problem_size", "num_threads"], as_index=False)[available]
        .median(numeric_only=True)
    )


@lru_cache(maxsize=8)
def _bootstrap_index_grid(n: int) -> np.ndarray:
    return np.asarray(list(itertools.product(range(n), repeat=n)), dtype=np.int16)

def exact_bootstrap_medians(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.asarray([], dtype=float)
    if arr.size <= 7:
        return np.median(arr[_bootstrap_index_grid(int(arr.size))], axis=1)
    rng = np.random.default_rng(20260720)
    return np.median(rng.choice(arr, size=(20_000, arr.size), replace=True), axis=1)


def bootstrap_median_ci(values: Iterable[float]) -> tuple[float, float, float]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return math.nan, math.nan, math.nan
    point = float(np.median(arr))
    boot = exact_bootstrap_medians(arr)
    low, high = np.quantile(boot, [0.025, 0.975])
    return point, float(low), float(high)


def independent_ratio_ci(a_values: Iterable[float], b_values: Iterable[float],
                         seed: int = 20260720, draws: int = 5_000) -> tuple[float, float, float]:
    """Median(A)/Median(B), with independent non-parametric bootstrap samples."""
    a = np.asarray(list(a_values), dtype=float)
    b = np.asarray(list(b_values), dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if not len(a) or not len(b):
        return math.nan, math.nan, math.nan
    point = float(np.median(a) / np.median(b))
    ba, bb = exact_bootstrap_medians(a), exact_bootstrap_medians(b)
    if len(ba) * len(bb) <= 2_000_000:
        ratios = (ba[:, None] / bb[None, :]).ravel()
    else:
        rng = np.random.default_rng(seed)
        ratios = ba[rng.integers(0, len(ba), size=draws)] / bb[rng.integers(0, len(bb), size=draws)]
    lo, hi = np.quantile(ratios, [0.025, 0.975])
    return point, float(lo), float(hi)


def probability_superiority(a_values: Iterable[float], b_values: Iterable[float],
                            lower_is_better: bool = True) -> float:
    a = np.asarray(list(a_values), dtype=float)
    b = np.asarray(list(b_values), dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if not len(a) or not len(b):
        return math.nan
    wins = 0.0
    for av in a:
        for bv in b:
            if av == bv:
                wins += 0.5
            elif (av < bv) if lower_is_better else (av > bv):
                wins += 1.0
    return wins / (len(a) * len(b))


def cliffs_delta(a_values: Iterable[float], b_values: Iterable[float]) -> float:
    """Positive means A tends to be numerically larger than B."""
    a = np.asarray(list(a_values), dtype=float)
    b = np.asarray(list(b_values), dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if not len(a) or not len(b):
        return math.nan
    greater = sum(av > bv for av in a for bv in b)
    lower = sum(av < bv for av in a for bv in b)
    return (greater - lower) / (len(a) * len(b))


def campaign_summary(data: pd.DataFrame) -> pd.DataFrame:
    sessions = session_medians(data)
    metrics = [
        "runtime_per_op_s", "total_energy_per_op_j", "package_energy_per_op_j",
        "dram_energy_per_op_j", "total_avg_power_w", "package_avg_power_w",
        "edp_total_j_s", "edp_package_j_s", "throughput_gflops_exact",
        "total_efficiency_gflop_per_j", "package_efficiency_gflop_per_j", "temp_c",
    ]
    rows: list[dict] = []
    for (n, threads), group in sessions.groupby(["problem_size", "num_threads"]):
        row: dict = {
            "problem_size": int(n), "num_threads": int(threads),
            "sessions": int(group["session_number"].nunique()),
        }
        raw = data[(data["problem_size"] == n) & (data["num_threads"] == threads)]
        row["runs"] = int(len(raw))
        for metric in metrics:
            if metric not in group.columns or group[metric].dropna().empty:
                continue
            point, low, high = bootstrap_median_ci(group[metric])
            row[f"{metric}_median"] = point
            row[f"{metric}_ci95_low"] = low
            row[f"{metric}_ci95_high"] = high
            row[f"{metric}_between_session_robust_cv"] = robust_cv(group[metric])
            row[f"{metric}_between_session_cv"] = standard_cv(group[metric])
            row[f"{metric}_run_robust_cv"] = robust_cv(raw[metric])
            row[f"{metric}_run_cv"] = standard_cv(raw[metric])
        # Define fixed-work presentation metrics from the primitive median axes.
        # This prevents median-of-ratios artifacts and guarantees that throughput is
        # exactly the inverse runtime view and GFLOP/J exactly the inverse energy view.
        flops_per_op = 2.0 * float(n) ** 3
        if "runtime_per_op_s_median" in row:
            row["throughput_gflops_exact_median"] = flops_per_op / row["runtime_per_op_s_median"] / 1e9
            row["throughput_gflops_exact_ci95_low"] = flops_per_op / row["runtime_per_op_s_ci95_high"] / 1e9
            row["throughput_gflops_exact_ci95_high"] = flops_per_op / row["runtime_per_op_s_ci95_low"] / 1e9
        if "total_energy_per_op_j_median" in row:
            row["total_efficiency_gflop_per_j_median"] = flops_per_op / row["total_energy_per_op_j_median"] / 1e9
            row["total_efficiency_gflop_per_j_ci95_low"] = flops_per_op / row["total_energy_per_op_j_ci95_high"] / 1e9
            row["total_efficiency_gflop_per_j_ci95_high"] = flops_per_op / row["total_energy_per_op_j_ci95_low"] / 1e9
        if "package_energy_per_op_j_median" in row:
            row["package_efficiency_gflop_per_j_median"] = flops_per_op / row["package_energy_per_op_j_median"] / 1e9
            row["package_efficiency_gflop_per_j_ci95_low"] = flops_per_op / row["package_energy_per_op_j_ci95_high"] / 1e9
            row["package_efficiency_gflop_per_j_ci95_high"] = flops_per_op / row["package_energy_per_op_j_ci95_low"] / 1e9
        if "total_energy_per_op_j_median" in row and "runtime_per_op_s_median" in row:
            row["edp_total_j_s_median"] = row["total_energy_per_op_j_median"] * row["runtime_per_op_s_median"]
            row["edp_total_j_s_ci95_low"] = row["total_energy_per_op_j_ci95_low"] * row["runtime_per_op_s_ci95_low"]
            row["edp_total_j_s_ci95_high"] = row["total_energy_per_op_j_ci95_high"] * row["runtime_per_op_s_ci95_high"]
        if "package_energy_per_op_j_median" in row and "runtime_per_op_s_median" in row:
            row["edp_package_j_s_median"] = row["package_energy_per_op_j_median"] * row["runtime_per_op_s_median"]
            row["edp_package_j_s_ci95_low"] = row["package_energy_per_op_j_ci95_low"] * row["runtime_per_op_s_ci95_low"]
            row["edp_package_j_s_ci95_high"] = row["package_energy_per_op_j_ci95_high"] * row["runtime_per_op_s_ci95_high"]
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["problem_size", "num_threads"]).reset_index(drop=True)


def strict_pareto_mask(group: pd.DataFrame,
                       energy_col: str = "total_energy_per_op_j_median",
                       runtime_col: str = "runtime_per_op_s_median") -> pd.Series:
    energy = group[energy_col].to_numpy(float)
    runtime = group[runtime_col].to_numpy(float)
    keep = np.ones(len(group), dtype=bool)
    for i in range(len(group)):
        dominated = ((energy <= energy[i]) & (runtime <= runtime[i]) &
                     ((energy < energy[i]) | (runtime < runtime[i])))
        dominated[i] = False
        keep[i] = not dominated.any()
    return pd.Series(keep, index=group.index)


def practical_pareto_mask(group: pd.DataFrame, tolerance: float = PRACTICAL_TOLERANCE,
                          energy_col: str = "total_energy_per_op_j_median",
                          runtime_col: str = "runtime_per_op_s_median") -> pd.Series:
    energy = group[energy_col].to_numpy(float)
    runtime = group[runtime_col].to_numpy(float)
    keep = np.ones(len(group), dtype=bool)
    for i in range(len(group)):
        no_worse = (energy <= energy[i] * (1 + tolerance)) & (runtime <= runtime[i] * (1 + tolerance))
        materially_better = (energy < energy[i] * (1 - tolerance)) | (runtime < runtime[i] * (1 - tolerance))
        dominated = no_worse & materially_better
        dominated[i] = False
        keep[i] = not dominated.any()
    return pd.Series(keep, index=group.index)


def metric_leaders(summary: pd.DataFrame, metric: str,
                   objective: Literal["min", "max"] = "min",
                   tolerance: float = PRACTICAL_TOLERANCE) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = metric[:-7] if metric.endswith("_median") else metric
    low_col, high_col = f"{base}_ci95_low", f"{base}_ci95_high"
    leader_rows, candidate_rows = [], []
    for n, group in summary.groupby("problem_size", sort=True):
        ordered = group.sort_values([metric, "num_threads"], ascending=[objective == "min", True]).copy()
        best = ordered.iloc[0]
        best_value, best_low, best_high = float(best[metric]), float(best[low_col]), float(best[high_col])
        others = ordered.iloc[1:]
        if objective == "min":
            practical_separation = bool((others[metric] > best_value * (1 + tolerance)).all())
            ci_separation = bool((best_high < others[low_col]).all())
            within = ordered[metric] <= best_value * (1 + tolerance)
        else:
            practical_separation = bool((others[metric] < best_value / (1 + tolerance)).all())
            ci_separation = bool((best_low > others[high_col]).all())
            within = ordered[metric] >= best_value / (1 + tolerance)
        clear = practical_separation and ci_separation
        overlap = (ordered[low_col] <= best_high) & (ordered[high_col] >= best_low)
        candidates = ordered[within | overlap].copy()
        record = best.to_dict()
        record.update({
            "exact_best_threads": int(best["num_threads"]),
            "selection_status": "clear_leader" if clear else "tie_or_uncertain",
            "leader_threads": ",".join(str(int(x)) for x in candidates["num_threads"]),
            "leader_count": int(len(candidates)),
            "practical_gap_gt_tolerance": practical_separation,
            "ci_separated_from_all": ci_separation,
            "objective": objective,
            "practical_tolerance_pct": 100 * tolerance,
        })
        if len(others):
            second = float(others.iloc[0][metric])
            gap = (second / best_value - 1) if objective == "min" else (best_value / second - 1)
            record["gap_to_second_pct"] = 100 * gap
        else:
            record["gap_to_second_pct"] = math.nan
        leader_rows.append(record)
        for _, row in ordered.iterrows():
            rel = ((float(row[metric]) / best_value - 1) if objective == "min"
                   else (best_value / float(row[metric]) - 1))
            candidate_rows.append({
                "problem_size": int(n), "metric": metric, "objective": objective,
                "num_threads": int(row["num_threads"]), "value": float(row[metric]),
                "ci95_low": float(row[low_col]), "ci95_high": float(row[high_col]),
                "relative_loss_vs_best_pct": 100 * rel,
                "within_practical_tolerance": bool(within.loc[row.name]),
                "ci_overlaps_exact_best": bool(overlap.loc[row.name]),
                "in_leader_set": bool(within.loc[row.name] or overlap.loc[row.name]),
                "exact_best": int(row["num_threads"]) == int(best["num_threads"]),
                "selection_status": "clear_leader" if clear else "tie_or_uncertain",
            })
    return pd.DataFrame(leader_rows), pd.DataFrame(candidate_rows)


def markdown_table(frame: pd.DataFrame, max_rows: int = 40) -> str:
    return "_Keine Einträge._" if frame.empty else frame.head(max_rows).to_markdown(index=False)


def write_text(path: Path, text: str) -> None:
    path.write_text(text.rstrip() + "\n", encoding="utf-8")

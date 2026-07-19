#!/usr/bin/env python3
"""Audit and summarize CPU/GPU energy benchmark CSV campaigns.

Designed for the cpu-gpu-v2 45-column schema used by the six workloads:
GEMM, STRIDED_GEMM, STREAM, AXPY, REDUCTION, and CONV2D.

Primary cross-platform energy metric:
    package_energy_per_op_j = device_energy_j / batches

This is intentionally package-only. On Intel, total_energy_j may additionally
contain DRAM energy; on AMD, dram_energy_j is unavailable (-1).
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

try:
    import numpy as np
    import pandas as pd
except ImportError as exc:  # pragma: no cover - user-facing dependency check
    raise SystemExit(
        "Missing dependency. Install with:\n"
        "  python3 -m pip install --user pandas numpy\n"
    ) from exc


EXPECTED_COLUMNS = [
    "schema_version",
    "timestamp",
    "session_id",
    "sequence_index",
    "run_id_global",
    "repetition",
    "workload",
    "implementation",
    "execution_mode",
    "device_name",
    "num_threads",
    "problem_size",
    "problem_spec",
    "batches",
    "e2e_time_s",
    "kernel_time_s",
    "wall_time_s",
    "device_energy_j",
    "total_energy_j",
    "dram_energy_j",
    "energy_per_op_j",
    "energy_per_second_j",
    "energy_per_flop_j",
    "time_per_op_ms_kernel",
    "time_per_op_ms_e2e",
    "flops_total",
    "gflops_per_s",
    "logical_bytes_per_op",
    "avg_power_w",
    "runtime_status",
    "pcie_gen",
    "pcie_width",
    "sm_clock_mhz",
    "clock_before_mhz",
    "clock_after_mhz",
    "mem_clock_mhz",
    "temp_c",
    "temp_before_c",
    "temp_after_c",
    "throttle_reasons",
    "cpu_cycles",
    "cpu_instructions",
    "cpu_ipc",
    "cpu_cache_misses",
    "checksum_ok",
]

REQUIRED_COLUMNS = {
    "schema_version",
    "session_id",
    "repetition",
    "workload",
    "device_name",
    "num_threads",
    "problem_size",
    "problem_spec",
    "batches",
    "e2e_time_s",
    "device_energy_j",
    "total_energy_j",
    "dram_energy_j",
    "time_per_op_ms_e2e",
    "gflops_per_s",
    "avg_power_w",
    "runtime_status",
    "checksum_ok",
}

NUMERIC_COLUMNS = [
    "sequence_index",
    "run_id_global",
    "repetition",
    "num_threads",
    "problem_size",
    "batches",
    "e2e_time_s",
    "kernel_time_s",
    "wall_time_s",
    "device_energy_j",
    "total_energy_j",
    "dram_energy_j",
    "energy_per_op_j",
    "energy_per_second_j",
    "energy_per_flop_j",
    "time_per_op_ms_kernel",
    "time_per_op_ms_e2e",
    "flops_total",
    "gflops_per_s",
    "logical_bytes_per_op",
    "avg_power_w",
    "pcie_gen",
    "pcie_width",
    "sm_clock_mhz",
    "clock_before_mhz",
    "clock_after_mhz",
    "mem_clock_mhz",
    "temp_c",
    "temp_before_c",
    "temp_after_c",
    "cpu_cycles",
    "cpu_instructions",
    "cpu_ipc",
    "cpu_cache_misses",
]

EXPECTED_PROBLEMS = {
    "GEMM": 9,
    "STRIDED_GEMM": 9,
    "STREAM": 9,
    "AXPY": 9,
    "REDUCTION": 9,
    "CONV2D": 6,
}

TRUE_VALUES = {"t", "true", "1", "yes", "y"}
FALSE_VALUES = {"f", "false", "0", "no", "n"}


@dataclass
class FileAudit:
    path: str
    rows: int = 0
    columns: int = 0
    header_exact: bool = False
    missing_required: str = ""
    schema_versions: str = ""
    workloads: str = ""
    sessions: str = ""
    checksum_failures: int = 0
    invalid_numeric_rows: int = 0
    duplicate_keys: int = 0
    expected_rows: int | None = None
    row_count_ok: bool | None = None
    status: str = "ok"
    error: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit and summarize benchmark run CSV files recursively."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Root directory containing run CSVs, e.g. ~/projects/energy/new/runs",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directory for reports and summary CSV files",
    )
    parser.add_argument(
        "--expected-sessions",
        type=int,
        default=5,
        help="Expected independent full campaign passes (default: 5)",
    )
    parser.add_argument(
        "--expected-reps",
        type=int,
        default=10,
        help="Expected repetitions per configuration per session (default: 10)",
    )
    parser.add_argument(
        "--threads",
        default="1,2,4,8,10,16,20",
        help="Expected thread grid for CPU audit (default: 1,2,4,8,10,16,20)",
    )
    parser.add_argument(
        "--baseline-threads",
        type=int,
        default=1,
        help="Reference thread count for trade-off classification (default: 1)",
    )
    parser.add_argument(
        "--tolerance-pct",
        type=float,
        default=1.0,
        help="Near-tie tolerance for trade-off classes, in percent (default: 1.0)",
    )
    parser.add_argument(
        "--campaign",
        default="auto",
        help=(
            "Campaign timestamp/prefix to analyze, 'auto' for the most complete "
            "campaign (default), or 'all' to combine all campaigns"
        ),
    )
    parser.add_argument(
        "--include-quickchecks",
        action="store_true",
        help="Include files/session IDs containing quickcheck, test, or partial",
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        help="Generate optional PNG plots (requires matplotlib)",
    )
    parser.add_argument(
        "--strict-header",
        action="store_true",
        help="Exit non-zero if any CSV header differs from the exact 45-column schema",
    )
    return parser.parse_args()


def parse_threads(value: str) -> list[int]:
    try:
        threads = sorted({int(part.strip()) for part in value.split(",") if part.strip()})
    except ValueError as exc:
        raise SystemExit(f"Invalid --threads value: {value!r}") from exc
    if not threads or any(t <= 0 for t in threads):
        raise SystemExit("--threads must contain positive integers")
    return threads


def bool_series(series: pd.Series) -> pd.Series:
    normalized = series.astype(str).str.strip().str.lower()
    result = pd.Series(pd.NA, index=series.index, dtype="boolean")
    result.loc[normalized.isin(TRUE_VALUES)] = True
    result.loc[normalized.isin(FALSE_VALUES)] = False
    return result


def is_quickcheck(path: Path, frame: pd.DataFrame | None = None) -> bool:
    marker = str(path).lower()
    if re.search(r"quickcheck|(^|[_-])test([_-]|\.|$)|partial", marker):
        return True
    if frame is not None and "session_id" in frame:
        sessions = frame["session_id"].astype(str).str.lower()
        return sessions.str.contains(r"quickcheck|test|partial", regex=True).any()
    return False


def infer_platform(path: Path, frame: pd.DataFrame) -> str:
    text = str(path).lower()
    if "/intel/" in text or "_intel_" in text:
        return "intel"
    if "/amd/" in text or "_amd_" in text:
        return "amd"
    if "3090" in text:
        return "rtx3090"
    if "5050" in text:
        return "rtx5050"
    execution = frame.get("execution_mode", pd.Series(dtype=str)).astype(str)
    if execution.str.startswith("gpu").any():
        device = " ".join(frame.get("device_name", pd.Series(dtype=str)).astype(str).unique()).lower()
        if "3090" in device:
            return "rtx3090"
        if "5050" in device:
            return "rtx5050"
        return "gpu"
    device = " ".join(frame.get("device_name", pd.Series(dtype=str)).astype(str).unique()).lower()
    if "amd" in device or "threadripper" in device or "ryzen" in device:
        return "amd"
    if "intel" in device:
        return "intel"
    return "unknown"


def expected_file_rows(frame: pd.DataFrame, expected_reps: int, expected_threads: Sequence[int]) -> int | None:
    if frame.empty or "workload" not in frame:
        return None
    workloads = frame["workload"].dropna().astype(str).str.upper().unique()
    if len(workloads) != 1:
        return None
    workload = workloads[0]
    problem_count = EXPECTED_PROBLEMS.get(workload)
    execution = frame.get("execution_mode", pd.Series(dtype=str)).astype(str)
    is_cpu = execution.str.startswith("cpu").all() if not execution.empty else True
    if problem_count is None or not is_cpu:
        return None
    return problem_count * len(expected_threads) * expected_reps


def read_and_audit_file(
    path: Path,
    expected_reps: int,
    expected_threads: Sequence[int],
    include_quickchecks: bool,
) -> tuple[pd.DataFrame | None, FileAudit]:
    audit = FileAudit(path=str(path))
    try:
        frame = pd.read_csv(path, low_memory=False)
    except Exception as exc:  # noqa: BLE001 - detailed audit output is intentional
        audit.status = "error"
        audit.error = f"read failed: {exc}"
        return None, audit

    audit.rows = len(frame)
    audit.columns = len(frame.columns)
    audit.header_exact = list(frame.columns) == EXPECTED_COLUMNS
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    audit.missing_required = ";".join(missing)
    if missing:
        audit.status = "error"
        audit.error = "missing required columns"
        return None, audit

    if not include_quickchecks and is_quickcheck(path, frame):
        audit.status = "skipped_quickcheck"
        return None, audit

    for column in NUMERIC_COLUMNS:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame["checksum_ok_bool"] = bool_series(frame["checksum_ok"])
    frame["source_file"] = str(path)
    frame["platform"] = infer_platform(path, frame)

    audit.schema_versions = ";".join(sorted(frame["schema_version"].dropna().astype(str).unique()))
    audit.workloads = ";".join(sorted(frame["workload"].dropna().astype(str).unique()))
    audit.sessions = ";".join(sorted(frame["session_id"].dropna().astype(str).unique()))
    audit.checksum_failures = int((frame["checksum_ok_bool"] != True).fillna(True).sum())  # noqa: E712

    critical_numeric = [
        "repetition",
        "num_threads",
        "problem_size",
        "batches",
        "e2e_time_s",
        "device_energy_j",
        "time_per_op_ms_e2e",
    ]
    audit.invalid_numeric_rows = int(frame[critical_numeric].isna().any(axis=1).sum())

    duplicate_key = [
        "session_id",
        "workload",
        "problem_spec",
        "num_threads",
        "repetition",
    ]
    audit.duplicate_keys = int(frame.duplicated(duplicate_key, keep=False).sum())

    audit.expected_rows = expected_file_rows(frame, expected_reps, expected_threads)
    audit.row_count_ok = (
        len(frame) == audit.expected_rows if audit.expected_rows is not None else None
    )

    if audit.checksum_failures or audit.invalid_numeric_rows or audit.duplicate_keys:
        audit.status = "warning"
    if audit.row_count_ok is False:
        audit.status = "warning"

    return frame, audit


def campaign_id_from_session(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(r"_session\d+$", "", regex=True)


def select_campaign(data: pd.DataFrame, requested: str) -> tuple[pd.DataFrame, str]:
    data = data.copy()
    data["campaign_id"] = campaign_id_from_session(data["session_id"])
    if requested == "all":
        return data, "all"
    available = sorted(data["campaign_id"].dropna().astype(str).unique())
    if not available:
        raise SystemExit("No campaign IDs could be derived from session_id")
    if requested != "auto":
        if requested not in available:
            raise SystemExit(
                f"Requested campaign {requested!r} not found. Available: {', '.join(available)}"
            )
        return data[data["campaign_id"] == requested].copy(), requested

    scores = (
        data.groupby("campaign_id", dropna=False)
        .agg(
            workload_sessions=("session_id", lambda s: data.loc[s.index, ["workload", "session_id"]].drop_duplicates().shape[0]),
            sessions=("session_id", "nunique"),
            workloads=("workload", "nunique"),
            rows=("session_id", "size"),
        )
        .reset_index()
        .sort_values(
            ["workload_sessions", "sessions", "workloads", "rows", "campaign_id"],
            ascending=[False, False, False, False, False],
        )
    )
    selected = str(scores.iloc[0]["campaign_id"])
    print("Campaign candidates:")
    for row in scores.itertuples(index=False):
        marker = "  <-- selected" if str(row.campaign_id) == selected else ""
        print(
            f"  {row.campaign_id}: workloads={row.workloads}, sessions={row.sessions}, "
            f"workload-sessions={row.workload_sessions}, rows={row.rows}{marker}"
        )
    return data[data["campaign_id"] == selected].copy(), selected


def load_campaign(args: argparse.Namespace, expected_threads: Sequence[int]) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    input_root = args.input.expanduser().resolve()
    if not input_root.exists():
        raise SystemExit(f"Input directory does not exist: {input_root}")

    paths = sorted(input_root.rglob("*.csv"))
    if not paths:
        raise SystemExit(f"No CSV files found under: {input_root}")

    frames: list[pd.DataFrame] = []
    audits: list[dict[str, object]] = []
    for path in paths:
        frame, audit = read_and_audit_file(
            path,
            expected_reps=args.expected_reps,
            expected_threads=expected_threads,
            include_quickchecks=args.include_quickchecks,
        )
        audits.append(audit.__dict__)
        if frame is not None:
            frames.append(frame)

    audit_df = pd.DataFrame(audits)
    if not frames:
        raise SystemExit("No usable campaign CSV files remained after filtering")

    data = pd.concat(frames, ignore_index=True, sort=False)
    data, selected_campaign = select_campaign(data, args.campaign)
    selected_files = set(data["source_file"].astype(str))
    audit_df["selected_campaign"] = audit_df["path"].astype(str).isin(selected_files)
    return data, audit_df, selected_campaign


def add_derived_columns(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()

    positive_batches = data["batches"].where(data["batches"] > 0)
    data["package_energy_per_op_j"] = data["device_energy_j"] / positive_batches
    data["dram_energy_per_op_j"] = np.where(
        data["dram_energy_j"] >= 0,
        data["dram_energy_j"] / positive_batches,
        np.nan,
    )
    data["total_energy_per_op_recomputed_j"] = data["total_energy_j"] / positive_batches
    data["edp_package_j_ms"] = (
        data["package_energy_per_op_j"] * data["time_per_op_ms_e2e"]
    )

    has_dram = data["dram_energy_j"] >= 0
    expected_total = np.where(
        has_dram,
        data["device_energy_j"] + data["dram_energy_j"],
        data["device_energy_j"],
    )
    tolerance = np.maximum(1e-6, np.abs(expected_total) * 1e-6)
    data["total_energy_consistent"] = (
        np.abs(data["total_energy_j"] - expected_total) <= tolerance
    )

    data["time_fields_consistent_cpu"] = True
    cpu_rows = data["execution_mode"].astype(str).str.startswith("cpu")
    if cpu_rows.any():
        t = data.loc[cpu_rows, ["e2e_time_s", "kernel_time_s", "wall_time_s"]]
        max_delta = t.max(axis=1) - t.min(axis=1)
        data.loc[cpu_rows, "time_fields_consistent_cpu"] = max_delta <= 1e-6

    data["row_valid"] = (
        (data["checksum_ok_bool"] == True)  # noqa: E712
        & (data["batches"] > 0)
        & (data["e2e_time_s"] > 0)
        & (data["device_energy_j"] > 0)
        & (data["package_energy_per_op_j"] > 0)
        & (data["time_per_op_ms_e2e"] > 0)
        & data["total_energy_consistent"]
        & data["time_fields_consistent_cpu"]
    )
    return data


def quantile_05(series: pd.Series) -> float:
    return float(series.quantile(0.05))


def quantile_95(series: pd.Series) -> float:
    return float(series.quantile(0.95))


def coefficient_of_variation(series: pd.Series) -> float:
    mean = float(series.mean())
    if not math.isfinite(mean) or mean == 0:
        return np.nan
    return float(series.std(ddof=1) / mean)


def make_config_summary(valid: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "platform",
        "device_name",
        "workload",
        "implementation",
        "execution_mode",
        "problem_size",
        "problem_spec",
        "num_threads",
    ]

    aggregations = {
        "session_id": ["nunique"],
        "source_file": ["nunique"],
        "repetition": ["count"],
        "batches": ["median", "min", "max"],
        "time_per_op_ms_e2e": [
            "mean",
            "median",
            "std",
            quantile_05,
            quantile_95,
            coefficient_of_variation,
        ],
        "package_energy_per_op_j": [
            "mean",
            "median",
            "std",
            quantile_05,
            quantile_95,
            coefficient_of_variation,
        ],
        "dram_energy_per_op_j": ["mean", "median"],
        "total_energy_per_op_recomputed_j": ["mean", "median"],
        "avg_power_w": ["mean", "median", "std"],
        "gflops_per_s": ["mean", "median", "std"],
        "temp_c": ["mean", "median", "max"],
        "clock_before_mhz": ["mean", "median"],
        "clock_after_mhz": ["mean", "median"],
        "edp_package_j_ms": ["mean", "median"],
    }

    summary = valid.groupby(group_cols, dropna=False).agg(aggregations).reset_index()
    summary.columns = [
        "_".join(str(part) for part in col if part).rstrip("_")
        if isinstance(col, tuple)
        else str(col)
        for col in summary.columns
    ]

    rename = {
        "session_id_nunique": "sessions",
        "source_file_nunique": "files",
        "repetition_count": "measurements",
        "time_per_op_ms_e2e_coefficient_of_variation": "time_per_op_ms_e2e_cv",
        "package_energy_per_op_j_coefficient_of_variation": "package_energy_per_op_j_cv",
        "time_per_op_ms_e2e_quantile_05": "time_per_op_ms_e2e_p05",
        "time_per_op_ms_e2e_quantile_95": "time_per_op_ms_e2e_p95",
        "package_energy_per_op_j_quantile_05": "package_energy_per_op_j_p05",
        "package_energy_per_op_j_quantile_95": "package_energy_per_op_j_p95",
    }
    summary = summary.rename(columns=rename)
    return summary.sort_values(["platform", "workload", "problem_size", "num_threads"])


def make_completeness(
    data: pd.DataFrame,
    expected_sessions: int,
    expected_reps: int,
    expected_threads: Sequence[int],
) -> pd.DataFrame:
    key = ["platform", "device_name", "workload", "problem_size", "problem_spec", "num_threads"]
    grouped = data.groupby(key, dropna=False).agg(
        measurements=("repetition", "size"),
        sessions=("session_id", "nunique"),
        files=("source_file", "nunique"),
        repetitions_seen=("repetition", lambda s: ",".join(map(str, sorted(set(s.dropna().astype(int)))))),
        valid_measurements=("row_valid", "sum"),
    ).reset_index()
    grouped["expected_measurements"] = expected_sessions * expected_reps
    grouped["measurement_deficit"] = grouped["expected_measurements"] - grouped["measurements"]
    grouped["complete"] = (
        (grouped["measurements"] == grouped["expected_measurements"])
        & (grouped["sessions"] == expected_sessions)
        & (grouped["valid_measurements"] == grouped["expected_measurements"])
    )

    # Add explicit rows for missing CPU thread combinations.
    cpu = data[data["execution_mode"].astype(str).str.startswith("cpu")]
    missing_rows: list[dict[str, object]] = []
    problem_key = ["platform", "device_name", "workload", "problem_size", "problem_spec"]
    for values, group in cpu.groupby(problem_key, dropna=False):
        seen = set(group["num_threads"].dropna().astype(int))
        for thread in expected_threads:
            if thread not in seen:
                row = dict(zip(problem_key, values))
                row.update(
                    {
                        "num_threads": thread,
                        "measurements": 0,
                        "sessions": 0,
                        "files": 0,
                        "repetitions_seen": "",
                        "valid_measurements": 0,
                        "expected_measurements": expected_sessions * expected_reps,
                        "measurement_deficit": expected_sessions * expected_reps,
                        "complete": False,
                    }
                )
                missing_rows.append(row)
    if missing_rows:
        grouped = pd.concat([grouped, pd.DataFrame(missing_rows)], ignore_index=True, sort=False)

    return grouped.sort_values(["platform", "workload", "problem_size", "num_threads"])


def pareto_flags(group: pd.DataFrame) -> pd.Series:
    energy = group["package_energy_per_op_j_median"].to_numpy(float)
    time = group["time_per_op_ms_e2e_median"].to_numpy(float)
    nondominated = np.ones(len(group), dtype=bool)
    for i in range(len(group)):
        dominates_i = (
            (energy <= energy[i])
            & (time <= time[i])
            & ((energy < energy[i]) | (time < time[i]))
        )
        dominates_i[i] = False
        if dominates_i.any():
            nondominated[i] = False
    return pd.Series(nondominated, index=group.index)


def make_pareto(summary: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["platform", "device_name", "workload", "problem_size", "problem_spec"]
    frames: list[pd.DataFrame] = []
    for _, group in summary.groupby(group_cols, dropna=False):
        group = group.copy()
        group["pareto_optimal"] = pareto_flags(group)
        frames.append(group)
    return pd.concat(frames, ignore_index=True) if frames else summary.assign(pareto_optimal=False)


def classify_delta(energy_pct: float, time_pct: float, tolerance_pct: float) -> str:
    e_low = energy_pct < -tolerance_pct
    e_high = energy_pct > tolerance_pct
    t_low = time_pct < -tolerance_pct
    t_high = time_pct > tolerance_pct
    if abs(energy_pct) <= tolerance_pct and abs(time_pct) <= tolerance_pct:
        return "near-equivalent"
    if e_low and t_low:
        return "dominant"
    if e_low and t_high:
        return "energy-efficient compromise"
    if e_high and t_low:
        return "runtime-efficient compromise"
    if e_high and t_high:
        return "bad trade-off"
    return "mixed/near-tie"


def make_tradeoffs(
    summary: pd.DataFrame,
    baseline_threads: int,
    tolerance_pct: float,
) -> pd.DataFrame:
    group_cols = ["platform", "device_name", "workload", "problem_size", "problem_spec"]
    output: list[pd.DataFrame] = []
    for _, group in summary.groupby(group_cols, dropna=False):
        baseline = group[group["num_threads"] == baseline_threads]
        if baseline.empty:
            continue
        base_energy = float(baseline.iloc[0]["package_energy_per_op_j_median"])
        base_time = float(baseline.iloc[0]["time_per_op_ms_e2e_median"])
        current = group.copy()
        current["baseline_threads"] = baseline_threads
        current["energy_change_vs_baseline_pct"] = (current["package_energy_per_op_j_median"] / base_energy - 1.0) * 100.0
        current["runtime_change_vs_baseline_pct"] = (current["time_per_op_ms_e2e_median"] / base_time - 1.0) * 100.0
        current["speedup_vs_baseline"] = base_time / current["time_per_op_ms_e2e_median"]
        current["energy_saving_vs_baseline_pct"] = -current["energy_change_vs_baseline_pct"]
        current["tradeoff_class"] = [
            "baseline" if int(thread) == baseline_threads else classify_delta(e, t, tolerance_pct)
            for thread, e, t in zip(
                current["num_threads"],
                current["energy_change_vs_baseline_pct"],
                current["runtime_change_vs_baseline_pct"],
            )
        ]
        output.append(current)
    return pd.concat(output, ignore_index=True) if output else pd.DataFrame()


def make_best_threads(summary: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["platform", "device_name", "workload", "problem_size", "problem_spec"]
    rows: list[dict[str, object]] = []
    for values, group in summary.groupby(group_cols, dropna=False):
        energy_row = group.loc[group["package_energy_per_op_j_median"].idxmin()]
        runtime_row = group.loc[group["time_per_op_ms_e2e_median"].idxmin()]
        edp_row = group.loc[group["edp_package_j_ms_median"].idxmin()]
        throughput_row = group.loc[group["gflops_per_s_median"].idxmax()]
        row = dict(zip(group_cols, values))
        row.update(
            {
                "energy_best_threads": int(energy_row["num_threads"]),
                "energy_best_package_j_per_op": energy_row["package_energy_per_op_j_median"],
                "runtime_best_threads": int(runtime_row["num_threads"]),
                "runtime_best_ms_per_op": runtime_row["time_per_op_ms_e2e_median"],
                "edp_best_threads": int(edp_row["num_threads"]),
                "edp_best_j_ms": edp_row["edp_package_j_ms_median"],
                "throughput_best_threads": int(throughput_row["num_threads"]),
                "throughput_best_gflops_s": throughput_row["gflops_per_s_median"],
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["platform", "workload", "problem_size"])


def make_winner_counts(best: pd.DataFrame) -> pd.DataFrame:
    records: list[pd.DataFrame] = []
    for criterion, column in [
        ("energy", "energy_best_threads"),
        ("runtime", "runtime_best_threads"),
        ("edp", "edp_best_threads"),
        ("throughput", "throughput_best_threads"),
    ]:
        counts = (
            best.groupby(["platform", "workload", column], dropna=False)
            .size()
            .rename("wins")
            .reset_index()
            .rename(columns={column: "num_threads"})
        )
        counts["criterion"] = criterion
        records.append(counts)
    return pd.concat(records, ignore_index=True).sort_values(
        ["platform", "workload", "criterion", "wins"], ascending=[True, True, True, False]
    )


def make_session_summary(data: pd.DataFrame) -> pd.DataFrame:
    return (
        data.groupby(["platform", "device_name", "workload", "session_id"], dropna=False)
        .agg(
            measurements=("repetition", "size"),
            valid_measurements=("row_valid", "sum"),
            configs=("problem_spec", lambda s: data.loc[s.index, ["problem_spec", "num_threads"]].drop_duplicates().shape[0]),
            package_energy_measured_j=("device_energy_j", "sum"),
            total_energy_measured_j=("total_energy_j", "sum"),
            measured_window_s=("e2e_time_s", "sum"),
            median_temp_c=("temp_c", "median"),
            max_temp_c=("temp_c", "max"),
        )
        .reset_index()
    )


def write_markdown_report(
    path: Path,
    args: argparse.Namespace,
    data: pd.DataFrame,
    valid: pd.DataFrame,
    audit: pd.DataFrame,
    completeness: pd.DataFrame,
    summary: pd.DataFrame,
    tradeoffs: pd.DataFrame,
) -> None:
    workload_counts = data.groupby("workload").size().sort_index()
    runtime_counts = data["runtime_status"].astype(str).value_counts()
    invalid_count = int((~data["row_valid"]).sum())
    incomplete = int((~completeness["complete"]).sum())
    package_kwh = float(valid["device_energy_j"].sum() / 3_600_000.0)
    total_kwh = float(valid["total_energy_j"].sum() / 3_600_000.0)
    dram_available = valid.loc[valid["dram_energy_j"] >= 0, "dram_energy_j"]
    dram_kwh = float(dram_available.sum() / 3_600_000.0) if not dram_available.empty else 0.0
    checksum_failures = int((data["checksum_ok_bool"] != True).fillna(True).sum())  # noqa: E712
    total_inconsistent = int((~data["total_energy_consistent"]).sum())
    time_inconsistent = int((~data["time_fields_consistent_cpu"]).sum())
    exact_headers = int(audit["header_exact"].fillna(False).sum())
    skipped = int((audit["status"] == "skipped_quickcheck").sum())

    lines = [
        "# Benchmark campaign analysis",
        "",
        "## Scope",
        "",
        f"- Input: `{args.input.expanduser()}`",
        f"- Selected campaign: **{data['campaign_id'].iloc[0] if 'campaign_id' in data else 'unknown'}**",
        f"- CSV files discovered: **{len(audit)}**",
        f"- CSV files selected for this campaign: **{int(audit.get('selected_campaign', pd.Series(dtype=bool)).fillna(False).sum())}**",
        f"- Exact 45-column headers: **{exact_headers}/{len(audit)}**",
        f"- Quickcheck/test files skipped: **{skipped}**",
        f"- Rows loaded: **{len(data):,}**",
        f"- Rows used for summaries: **{len(valid):,}**",
        f"- Platforms: **{', '.join(sorted(data['platform'].astype(str).unique()))}**",
        f"- Devices: **{'; '.join(sorted(data['device_name'].astype(str).unique()))}**",
        "",
        "## Audit result",
        "",
        f"- Invalid rows: **{invalid_count}**",
        f"- Checksum failures/unknown values: **{checksum_failures}**",
        f"- Inconsistent total-energy rows: **{total_inconsistent}**",
        f"- Inconsistent CPU time fields: **{time_inconsistent}**",
        f"- Incomplete configuration groups: **{incomplete}**",
        "",
        "## Rows by workload",
        "",
    ]
    for workload, count in workload_counts.items():
        lines.append(f"- {workload}: **{count:,}**")

    lines.extend(["", "## Runtime-status distribution", ""])
    for status, count in runtime_counts.items():
        lines.append(f"- {status}: **{count:,}**")

    lines.extend(
        [
            "",
            "## Measured energy-window totals",
            "",
            f"- Package/device energy: **{package_kwh:.6f} kWh**",
            f"- DRAM energy where available: **{dram_kwh:.6f} kWh**",
            f"- Stored total energy: **{total_kwh:.6f} kWh**",
            "",
            "> These totals cover only the timed measurement windows. They exclude calibration, allocation, cooldown pauses, idle system draw, GPU board/system overhead outside the selected sensor domain, and full wall-socket energy.",
            "",
            "## Primary analysis convention",
            "",
            "Cross-platform CPU comparisons should use `package_energy_per_op_j`, derived from `device_energy_j / batches`. Intel Package+DRAM (`total_energy_j`) is retained only as a platform-specific sensitivity metric because AMD has no comparable DRAM domain.",
            "",
            "## Generated files",
            "",
            "- `file_audit.csv` — per-file structural audit",
            "- `all_runs_merged.csv.gz` — all loaded raw rows plus derived QC fields",
            "- `invalid_rows.csv` — rows excluded from summary calculations",
            "- `configuration_completeness.csv` — expected 50 measurements per configuration",
            "- `config_summary.csv` — aggregate statistics per workload/problem/thread",
            "- `best_threads.csv` — energy, runtime, EDP, and throughput winners",
            "- `winner_counts.csv` — winner counts by workload and criterion",
            "- `pareto_front.csv` — energy/runtime Pareto classification",
            "- `tradeoff_vs_1thread.csv` — dominant/compromise/bad-trade-off classes",
            "- `session_summary.csv` — totals and QC per session/workload",
            "",
        ]
    )

    if not tradeoffs.empty:
        class_counts = tradeoffs["tradeoff_class"].value_counts()
        lines.extend(["## Trade-off classes versus baseline", ""])
        for label, count in class_counts.items():
            lines.append(f"- {label}: **{count:,}**")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def generate_plots(output: Path, summary: pd.DataFrame, winners: pd.DataFrame) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "--plots requested but matplotlib is missing. Install with:\n"
            "  python3 -m pip install --user matplotlib"
        ) from exc

    figure_dir = output / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    for (platform, workload), group in summary.groupby(["platform", "workload"], dropna=False):
        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(
            group["time_per_op_ms_e2e_median"],
            group["package_energy_per_op_j_median"],
            c=group["num_threads"],
            s=35,
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Median runtime per operation [ms]")
        ax.set_ylabel("Median package energy per operation [J]")
        ax.set_title(f"{platform} — {workload}: energy/runtime trade-off")
        fig.colorbar(scatter, ax=ax, label="Threads")
        fig.tight_layout()
        safe = re.sub(r"[^a-z0-9_-]+", "_", f"{platform}_{workload}".lower())
        fig.savefig(figure_dir / f"{safe}_energy_runtime.png", dpi=180)
        plt.close(fig)

    for (platform, workload, criterion), group in winners.groupby(
        ["platform", "workload", "criterion"], dropna=False
    ):
        group = group.sort_values("num_threads")
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(group["num_threads"].astype(str), group["wins"])
        ax.set_xlabel("Threads")
        ax.set_ylabel("Number of problem-size wins")
        ax.set_title(f"{platform} — {workload}: {criterion} winners")
        fig.tight_layout()
        safe = re.sub(r"[^a-z0-9_-]+", "_", f"{platform}_{workload}_{criterion}".lower())
        fig.savefig(figure_dir / f"{safe}_winner_counts.png", dpi=180)
        plt.close(fig)


def main() -> int:
    args = parse_args()
    expected_threads = parse_threads(args.threads)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    data, file_audit, selected_campaign = load_campaign(args, expected_threads)
    data = add_derived_columns(data)
    valid = data[data["row_valid"]].copy()
    if valid.empty:
        raise SystemExit("No valid rows remain after QC checks")

    completeness = make_completeness(
        data,
        expected_sessions=args.expected_sessions,
        expected_reps=args.expected_reps,
        expected_threads=expected_threads,
    )
    config_summary = make_config_summary(valid)
    pareto = make_pareto(config_summary)
    tradeoffs = make_tradeoffs(
        config_summary,
        baseline_threads=args.baseline_threads,
        tolerance_pct=args.tolerance_pct,
    )
    best_threads = make_best_threads(config_summary)
    winner_counts = make_winner_counts(best_threads)
    session_summary = make_session_summary(data)

    file_audit.to_csv(output / "file_audit.csv", index=False)
    data.to_csv(output / "all_runs_merged.csv.gz", index=False, compression="gzip")
    data.loc[~data["row_valid"]].to_csv(output / "invalid_rows.csv", index=False)
    completeness.to_csv(output / "configuration_completeness.csv", index=False)
    config_summary.to_csv(output / "config_summary.csv", index=False)
    pareto.to_csv(output / "pareto_front.csv", index=False)
    tradeoffs.to_csv(output / "tradeoff_vs_1thread.csv", index=False)
    best_threads.to_csv(output / "best_threads.csv", index=False)
    winner_counts.to_csv(output / "winner_counts.csv", index=False)
    session_summary.to_csv(output / "session_summary.csv", index=False)

    metadata = {
        "input": str(args.input.expanduser()),
        "selected_campaign": selected_campaign,
        "expected_sessions": args.expected_sessions,
        "expected_reps": args.expected_reps,
        "expected_threads": expected_threads,
        "baseline_threads": args.baseline_threads,
        "tolerance_pct": args.tolerance_pct,
        "files_discovered": len(file_audit),
        "rows_loaded": len(data),
        "rows_valid": len(valid),
    }
    (output / "analysis_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    write_markdown_report(
        output / "analysis_report.md",
        args,
        data,
        valid,
        file_audit,
        completeness,
        config_summary,
        tradeoffs,
    )

    if args.plots:
        generate_plots(output, config_summary, winner_counts)

    header_problems = int((~file_audit["header_exact"].fillna(False)).sum())
    incomplete = int((~completeness["complete"]).sum())
    invalid = int((~data["row_valid"]).sum())

    print("=" * 72)
    print("Benchmark analysis complete")
    print("=" * 72)
    print(f"Selected campaign:       {selected_campaign}")
    print(f"Files discovered:        {len(file_audit)}")
    print(f"Files selected:          {int(file_audit['selected_campaign'].fillna(False).sum())}")
    print(f"Rows loaded:             {len(data):,}")
    print(f"Rows valid:              {len(valid):,}")
    print(f"Header mismatches:       {header_problems}")
    print(f"Invalid rows:            {invalid}")
    print(f"Incomplete config groups:{incomplete}")
    print(f"Output directory:        {output}")
    print(f"Main report:             {output / 'analysis_report.md'}")

    if args.strict_header and header_problems:
        return 2
    if invalid or incomplete:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

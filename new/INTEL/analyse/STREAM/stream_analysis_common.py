#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import itertools
import math
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SIZES = [
    1_000_000, 2_000_000, 4_000_000, 8_000_000, 16_000_000,
    32_000_000, 64_000_000, 128_000_000, 256_000_000,
]
EXPECTED_SESSIONS = 5
EXPECTED_REPETITIONS = 10
PRACTICAL_TOLERANCE = 0.02
SCHEMA_VERSION = "cpu-gpu-v2"
EXPECTED_COLUMNS = [
    "schema_version", "timestamp", "session_id", "sequence_index", "run_id_global",
    "repetition", "workload", "implementation", "execution_mode", "device_name",
    "num_threads", "problem_size", "problem_spec", "batches", "e2e_time_s",
    "kernel_time_s", "wall_time_s", "device_energy_j", "total_energy_j",
    "dram_energy_j", "energy_per_op_j", "energy_per_second_j", "energy_per_flop_j",
    "time_per_op_ms_kernel", "time_per_op_ms_e2e", "flops_total", "gflops_per_s",
    "logical_bytes_per_op", "avg_power_w", "runtime_status", "pcie_gen", "pcie_width",
    "sm_clock_mhz", "clock_before_mhz", "clock_after_mhz", "mem_clock_mhz", "temp_c",
    "temp_before_c", "temp_after_c", "throttle_reasons", "cpu_cycles",
    "cpu_instructions", "cpu_ipc", "cpu_cache_misses", "checksum_ok",
]

PLATFORM_CONFIG = {
    "AMD": {
        "label": "AMD Threadripper 3970X",
        "kind": "CPU",
        "slug": "amd",
        "threads": [1, 2, 4, 8, 10, 16, 20, 32, 64],
        "implementation": "openmp_triad",
        "mode": "cpu_native",
        "energy_domain": "CPU package RAPL",
        "source_rel": "scripts/STREAM/main_stream_amd.cpp",
        "runner_rel": "scripts/02_run_CPU_AMD_STREAM_only.sh",
    },
    "INTEL": {
        "label": "Intel Core i9-7900X",
        "kind": "CPU",
        "slug": "intel",
        "threads": [1, 2, 4, 8, 10, 16, 20],
        "implementation": "openmp_triad",
        "mode": "cpu_native",
        "energy_domain": "CPU package RAPL",
        "source_rel": "scripts/STREAM/main_stream_intel.cpp",
        "runner_rel": "scripts/02_run_CPU_Intel_STREAM_only.sh",
    },
    "3090": {
        "label": "RTX 3090",
        "kind": "GPU",
        "slug": "3090",
        "threads": [-1],
        "implementation": "cuda_stream_triad_fp32",
        "mode": "gpu_resident",
        "energy_domain": "GPU board NVML",
        "source_rel": "scripts/STREAM/main_stream.cu",
        "runner_rel": "02_run_GPU_3090_STREAM_only.sh",
    },
    "5060ti": {
        "label": "RTX 5060 Ti",
        "kind": "GPU",
        "slug": "5060ti",
        "threads": [-1],
        "implementation": "cuda_stream_triad_fp32",
        "mode": "gpu_resident",
        "energy_domain": "GPU board NVML",
        "source_rel": "scripts/STREAM/main_stream.cu",
        "runner_rel": "scripts/02_run_GPU_5060ti_STREAM_only.sh",
    },
}

@dataclass(frozen=True)
class Context:
    project_root: Path
    platform_root: Path
    platform: str
    config: dict
    run_dir: Path
    result_dir: Path
    figure_dir: Path
    source_path: Path
    runner_path: Path

@dataclass
class Campaign:
    stamp: str
    files: list[Path]
    sessions: list[int]
    data: pd.DataFrame


def context(script_file: str | Path) -> Context:
    script = Path(script_file).resolve()
    platform_root = script.parents[2]
    platform = platform_root.name
    if platform not in PLATFORM_CONFIG:
        raise RuntimeError(f"Unsupported platform directory: {platform}")
    project_root = platform_root.parent
    cfg = PLATFORM_CONFIG[platform]
    result_dir = platform_root / "results" / "STREAM"
    figure_dir = result_dir / "figures"
    result_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    return Context(
        project_root=project_root,
        platform_root=platform_root,
        platform=platform,
        config=cfg,
        run_dir=platform_root / "runs" / "STREAM",
        result_dir=result_dir,
        figure_dir=figure_dir,
        source_path=platform_root / cfg["source_rel"],
        runner_path=platform_root / cfg["runner_rel"],
    )


def parse_args(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--campaign",
        help="Campaign timestamp YYYYMMDD_HHMMSS. Default: latest complete five-session campaign.",
    )
    return parser.parse_args()


def campaign_pattern(platform: str) -> re.Pattern[str]:
    slug = PLATFORM_CONFIG[platform]["slug"]
    return re.compile(rf"^stream_{re.escape(slug)}_(\d{{8}}_\d{{6}})_session([1-5])\.csv$", re.I)


def discover_campaigns(run_dir: Path, platform: str) -> dict[str, list[tuple[int, Path]]]:
    groups: dict[str, list[tuple[int, Path]]] = {}
    if not run_dir.is_dir():
        return groups
    pattern = campaign_pattern(platform)
    for path in run_dir.glob("*.csv"):
        match = pattern.match(path.name)
        if match:
            groups.setdefault(match.group(1), []).append((int(match.group(2)), path))
    return groups


def select_campaign(run_dir: Path, platform: str, requested: str | None) -> tuple[str, list[tuple[int, Path]]]:
    groups = discover_campaigns(run_dir, platform)
    complete = {
        stamp: sorted(entries)
        for stamp, entries in groups.items()
        if sorted(session for session, _ in entries) == [1, 2, 3, 4, 5]
    }
    if requested:
        if requested not in complete:
            available = ", ".join(sorted(complete)) or "none"
            raise RuntimeError(f"Campaign {requested!r} is not complete. Complete campaigns: {available}")
        return requested, complete[requested]
    if not complete:
        partial = {stamp: sorted(s for s, _ in entries) for stamp, entries in groups.items()}
        raise RuntimeError(f"No complete five-session STREAM campaign in {run_dir}. Partial: {partial}")
    stamp = max(complete)
    return stamp, complete[stamp]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"1", "t", "true", "yes", "y"})


def load_campaign(ctx: Context, requested: str | None = None) -> Campaign:
    stamp, entries = select_campaign(ctx.run_dir, ctx.platform, requested)
    frames: list[pd.DataFrame] = []
    files: list[Path] = []
    sessions: list[int] = []
    for session, path in entries:
        frame = pd.read_csv(path)
        frame["source_file"] = path.name
        frame["session_number"] = session
        frames.append(frame)
        files.append(path)
        sessions.append(session)
    data = pd.concat(frames, ignore_index=True, sort=False)
    numeric = [
        "sequence_index", "run_id_global", "repetition", "num_threads", "problem_size",
        "batches", "e2e_time_s", "kernel_time_s", "wall_time_s", "device_energy_j",
        "total_energy_j", "dram_energy_j", "energy_per_op_j", "energy_per_second_j",
        "energy_per_flop_j", "time_per_op_ms_kernel", "time_per_op_ms_e2e", "flops_total",
        "gflops_per_s", "logical_bytes_per_op", "avg_power_w", "pcie_gen", "pcie_width",
        "sm_clock_mhz", "clock_before_mhz", "clock_after_mhz", "mem_clock_mhz", "temp_c",
        "temp_before_c", "temp_after_c", "cpu_cycles", "cpu_instructions", "cpu_ipc",
        "cpu_cache_misses", "session_number",
    ]
    for column in numeric:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    data["checksum_bool"] = normalize_bool(data["checksum_ok"]) if "checksum_ok" in data else False
    return Campaign(stamp=stamp, files=files, sessions=sessions, data=data)


def add_check(checks: list[dict], category: str, name: str, severity: str,
              passed: bool, observed, expected) -> None:
    checks.append({
        "category": category,
        "check": name,
        "severity": severity,
        "status": "PASS" if passed else severity,
        "observed": str(observed),
        "expected": str(expected),
    })


def markdown_table(frame: pd.DataFrame, max_rows: int = 200) -> str:
    if frame.empty:
        return "_None._"
    return frame.head(max_rows).to_markdown(index=False)


def finite_positive(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return np.isfinite(values) & (values > 0)


def close_series(actual: pd.Series, expected: pd.Series | np.ndarray,
                 rtol: float, atol: float) -> pd.Series:
    a = pd.to_numeric(actual, errors="coerce").to_numpy(float)
    e = np.asarray(expected, dtype=float)
    return pd.Series(
        np.isfinite(a) & np.isfinite(e) & (np.abs(a - e) <= atol + rtol * np.abs(e)),
        index=actual.index,
    )


def validate_campaign(ctx: Context, campaign: Campaign) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = campaign.data
    checks: list[dict] = []
    failures: list[pd.DataFrame] = []
    cfg = ctx.config
    expected_threads = cfg["threads"]
    expected_configurations = len(SIZES) * len(expected_threads)
    expected_rows_per_session = expected_configurations * EXPECTED_REPETITIONS

    add_check(checks, "coverage", "five_sessions", "FAIL",
              sorted(campaign.sessions) == [1, 2, 3, 4, 5], campaign.sessions, [1, 2, 3, 4, 5])
    add_check(checks, "schema", "exact_header", "FAIL",
              list(data.columns[:len(EXPECTED_COLUMNS)]) == EXPECTED_COLUMNS,
              list(data.columns[:len(EXPECTED_COLUMNS)]), EXPECTED_COLUMNS)
    add_check(checks, "schema", "schema_version", "FAIL",
              set(data.get("schema_version", pd.Series(dtype=str)).astype(str)) == {SCHEMA_VERSION},
              sorted(set(data.get("schema_version", pd.Series(dtype=str)).astype(str))), SCHEMA_VERSION)

    for session in range(1, EXPECTED_SESSIONS + 1):
        frame = data[data.session_number == session]
        add_check(checks, "coverage", f"session_{session}_row_count", "FAIL",
                  len(frame) == expected_rows_per_session, len(frame), expected_rows_per_session)

    add_check(checks, "semantics", "workload", "FAIL",
              set(data.workload.astype(str)) == {"STREAM"}, sorted(set(data.workload.astype(str))), "STREAM")
    add_check(checks, "semantics", "implementation", "FAIL",
              set(data.implementation.astype(str)) == {cfg["implementation"]},
              sorted(set(data.implementation.astype(str))), cfg["implementation"])
    add_check(checks, "semantics", "execution_mode", "FAIL",
              set(data.execution_mode.astype(str)) == {cfg["mode"]},
              sorted(set(data.execution_mode.astype(str))), cfg["mode"])
    add_check(checks, "semantics", "device_name_consistent", "FAIL",
              data.device_name.astype(str).str.strip().nunique() == 1,
              sorted(data.device_name.astype(str).unique()), "one non-empty device name")
    add_check(checks, "coverage", "problem_sizes", "FAIL",
              sorted(data.problem_size.dropna().astype(int).unique()) == SIZES,
              sorted(data.problem_size.dropna().astype(int).unique()), SIZES)
    add_check(checks, "coverage", "thread_grid", "FAIL",
              sorted(data.num_threads.dropna().astype(int).unique()) == expected_threads,
              sorted(data.num_threads.dropna().astype(int).unique()), expected_threads)

    expected_spec = "elements=" + data.problem_size.astype("Int64").astype(str)
    spec_ok = data.problem_spec.astype(str).eq(expected_spec)
    add_check(checks, "semantics", "problem_spec", "FAIL", bool(spec_ok.all()), int((~spec_ok).sum()), 0)

    key_cols = ["session_number", "problem_size", "num_threads", "repetition"]
    duplicated = data.duplicated(key_cols, keep=False)
    add_check(checks, "coverage", "no_duplicate_measurements", "FAIL",
              not bool(duplicated.any()), int(duplicated.sum()), 0)

    counts = data.groupby(["session_number", "problem_size", "num_threads"])["repetition"].agg(["count", "nunique", "min", "max"])
    repetition_ok = (
        (counts["count"] == EXPECTED_REPETITIONS)
        & (counts["nunique"] == EXPECTED_REPETITIONS)
        & (counts["min"] == 1)
        & (counts["max"] == EXPECTED_REPETITIONS)
    )
    add_check(checks, "coverage", "ten_repetitions_per_configuration", "FAIL",
              bool(repetition_ok.all()) and len(counts) == EXPECTED_SESSIONS * expected_configurations,
              f"bad={int((~repetition_ok).sum())}, groups={len(counts)}",
              f"0 bad, {EXPECTED_SESSIONS * expected_configurations} groups")

    sequence_bad = 0
    for session, frame in data.groupby("session_number"):
        expected = np.arange(1, len(frame) + 1)
        observed = frame.sort_values("sequence_index").sequence_index.to_numpy(int)
        sequence_bad += int(not np.array_equal(observed, expected))
        sequence_bad += int(not np.array_equal(frame.sequence_index.to_numpy(), frame.run_id_global.to_numpy()))
    add_check(checks, "coverage", "sequence_and_run_ids", "FAIL", sequence_bad == 0, sequence_bad, 0)

    positive_columns = [
        "batches", "e2e_time_s", "kernel_time_s", "wall_time_s", "device_energy_j",
        "total_energy_j", "energy_per_op_j", "flops_total", "gflops_per_s",
        "logical_bytes_per_op", "avg_power_w",
    ]
    for column in positive_columns:
        ok = finite_positive(data[column])
        add_check(checks, "numeric", f"positive_{column}", "FAIL", bool(ok.all()), int((~ok).sum()), 0)

    checksum_ok = data.checksum_bool.astype(bool)
    add_check(checks, "correctness", "all_checksums", "FAIL", bool(checksum_ok.all()), int((~checksum_ok).sum()), 0)

    n = data.problem_size.to_numpy(float)
    batches = data.batches.to_numpy(float)
    expected_flops = 2.0 * n * batches
    expected_bytes = 12.0 * n
    expected_energy_op = data.total_energy_j.to_numpy(float) / batches
    expected_kernel_ms = data.kernel_time_s.to_numpy(float) * 1000.0 / batches
    expected_e2e_ms = data.e2e_time_s.to_numpy(float) * 1000.0 / batches
    expected_gflops = data.flops_total.to_numpy(float) / data.kernel_time_s.to_numpy(float) / 1e9
    expected_power = data.total_energy_j.to_numpy(float) / data.wall_time_s.to_numpy(float)

    formula_rules = {
        "flops_total": close_series(data.flops_total, expected_flops, 1e-10, 1e-6),
        "logical_bytes_per_op": close_series(data.logical_bytes_per_op, expected_bytes, 1e-10, 1e-3),
        "energy_per_op_j": close_series(data.energy_per_op_j, expected_energy_op, 5e-6, 5e-10),
        "time_per_op_ms_kernel": close_series(data.time_per_op_ms_kernel, expected_kernel_ms, 2e-5, 1.1e-6),
        "time_per_op_ms_e2e": close_series(data.time_per_op_ms_e2e, expected_e2e_ms, 2e-5, 1.1e-6),
        "gflops_per_s": close_series(data.gflops_per_s, expected_gflops, 2e-3, 1.1e-2),
        "avg_power_w": close_series(data.avg_power_w, expected_power, 2e-3, 1.1e-1),
    }
    for name, ok in formula_rules.items():
        add_check(checks, "formula", name, "FAIL", bool(ok.all()), int((~ok).sum()), 0)
        if (~ok).any():
            bad = data.loc[~ok, ["source_file", "session_number", "sequence_index", "problem_size", "num_threads", "repetition", name]].copy()
            bad["formula"] = name
            failures.append(bad)

    expected_status = np.where(
        data.e2e_time_s.to_numpy(float) < 0.75, "below",
        np.where(data.e2e_time_s.to_numpy(float) <= 1.25, "in_range", "above"),
    )
    status_ok = data.runtime_status.astype(str).to_numpy() == expected_status
    add_check(checks, "runtime", "runtime_status_formula", "FAIL", bool(status_ok.all()), int((~status_ok).sum()), 0)
    add_check(checks, "runtime", "all_runtime_in_target", "WARN",
              bool((data.runtime_status.astype(str) == "in_range").all()),
              data.runtime_status.value_counts().to_dict(), "all in_range preferred")

    if cfg["kind"] == "CPU":
        time_equal = close_series(data.kernel_time_s, data.e2e_time_s.to_numpy(float), 1e-9, 1e-9)
        wall_equal = close_series(data.wall_time_s, data.e2e_time_s.to_numpy(float), 1e-9, 1e-9)
        add_check(checks, "timing", "cpu_time_fields_equal", "FAIL",
                  bool((time_equal & wall_equal).all()), int((~(time_equal & wall_equal)).sum()), 0)
        dram = data.dram_energy_j.to_numpy(float)
        expected_total = data.device_energy_j.to_numpy(float) + np.where(dram >= 0, dram, 0.0)
        total_ok = close_series(data.total_energy_j, expected_total, 5e-6, 1e-5)
        add_check(checks, "energy", "cpu_total_energy_domain_formula", "FAIL",
                  bool(total_ok.all()), int((~total_ok).sum()), 0)
    else:
        kernel_le_e2e = data.kernel_time_s.to_numpy(float) <= data.e2e_time_s.to_numpy(float) + 2e-6
        add_check(checks, "timing", "gpu_kernel_not_above_e2e", "FAIL",
                  bool(kernel_le_e2e.all()), int((~kernel_le_e2e).sum()), 0)
        device_total = close_series(data.device_energy_j, data.total_energy_j.to_numpy(float), 5e-6, 1e-5)
        add_check(checks, "energy", "gpu_device_equals_total", "FAIL",
                  bool(device_total.all()), int((~device_total).sum()), 0)
        add_check(checks, "energy", "gpu_dram_sentinel", "FAIL",
                  bool((data.dram_energy_j == -1).all()), sorted(data.dram_energy_j.unique()), -1)
        add_check(checks, "gpu", "pcie_metadata_present", "WARN",
                  bool((data.pcie_gen > 0).all() and (data.pcie_width > 0).all()),
                  {"gen": sorted(data.pcie_gen.unique()), "width": sorted(data.pcie_width.unique())},
                  "positive PCIe generation and width")

    add_check(checks, "provenance", "source_present", "FAIL", ctx.source_path.is_file(), ctx.source_path, "present")
    add_check(checks, "provenance", "runner_present", "FAIL", ctx.runner_path.is_file(), ctx.runner_path, "present")

    check_df = pd.DataFrame(checks)
    failure_df = pd.concat(failures, ignore_index=True) if failures else pd.DataFrame()
    return check_df, failure_df


def add_derived(data: pd.DataFrame, ctx: Context) -> pd.DataFrame:
    df = data.copy()
    n = df.problem_size.astype(float)
    batches = df.batches.astype(float)
    runtime = df.e2e_time_s.astype(float) / batches
    kernel_runtime = df.kernel_time_s.astype(float) / batches
    package_or_board_energy = df.device_energy_j.astype(float) / batches
    total_energy = df.total_energy_j.astype(float) / batches
    dram_energy = np.where(df.dram_energy_j >= 0, df.dram_energy_j / batches, np.nan)
    logical_flops = 2.0 * n
    logical_bytes = 12.0 * n

    df["platform"] = ctx.platform
    df["platform_label"] = ctx.config["label"]
    df["device_kind"] = ctx.config["kind"]
    df["energy_domain"] = ctx.config["energy_domain"]
    df["configuration"] = np.where(
        ctx.config["kind"] == "CPU",
        df.num_threads.astype(int).astype(str) + "T",
        "gpu_resident",
    )
    df["runtime_per_op_s"] = runtime
    df["kernel_runtime_per_op_s"] = kernel_runtime
    df["primary_energy_per_op_j"] = package_or_board_energy
    df["total_energy_per_op_j"] = total_energy
    df["dram_energy_per_op_j"] = dram_energy
    df["primary_power_w"] = df.device_energy_j / df.wall_time_s
    df["total_power_w"] = df.total_energy_j / df.wall_time_s
    df["edp_primary_j_s"] = package_or_board_energy * runtime
    df["edp_total_j_s"] = total_energy * runtime
    df["throughput_gflops_exact"] = logical_flops / runtime / 1e9
    df["primary_efficiency_gflop_per_j"] = logical_flops / package_or_board_energy / 1e9
    df["logical_bandwidth_gb_s"] = logical_bytes / runtime / 1e9
    df["logical_gb_per_j"] = logical_bytes / package_or_board_energy / 1e9
    df["logical_flops_per_op"] = logical_flops
    df["logical_bytes_expected"] = logical_bytes
    df["working_set_bytes"] = logical_bytes
    df["operational_intensity_flop_per_byte"] = logical_flops / logical_bytes
    df["temperature_rise_c"] = df.temp_after_c - df.temp_before_c
    df["clock_change_pct"] = 100.0 * (df.clock_after_mhz - df.clock_before_mhz) / df.clock_before_mhz.replace(0, np.nan)
    return df


def robust_cv(values: Iterable[float]) -> float:
    x = np.asarray([float(v) for v in values if np.isfinite(v)], dtype=float)
    if len(x) < 2:
        return np.nan
    med = float(np.median(x))
    if med == 0:
        return np.nan
    mad = float(np.median(np.abs(x - med)))
    return 100.0 * 1.4826 * mad / abs(med)


def standard_cv(values: Iterable[float]) -> float:
    x = np.asarray([float(v) for v in values if np.isfinite(v)], dtype=float)
    if len(x) < 2 or np.mean(x) == 0:
        return np.nan
    return 100.0 * float(np.std(x, ddof=1) / abs(np.mean(x)))


def robust_outlier_mask(values: pd.Series, threshold: float = 3.5) -> pd.Series:
    x = pd.to_numeric(values, errors="coerce")
    med = x.median()
    mad = (x - med).abs().median()
    if not np.isfinite(mad) or mad == 0:
        return pd.Series(False, index=values.index)
    score = 0.67448975 * (x - med) / mad
    return score.abs() > threshold


@lru_cache(maxsize=8)
def bootstrap_indices(n: int) -> np.ndarray:
    return np.asarray(list(itertools.product(range(n), repeat=n)), dtype=np.int16)


def bootstrap_median_ci(values: Iterable[float]) -> tuple[float, float, float]:
    x = np.asarray([float(v) for v in values if np.isfinite(v)], dtype=float)
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    med = float(np.median(x))
    if len(x) <= 7:
        draws = np.median(x[bootstrap_indices(len(x))], axis=1)
    else:
        rng = np.random.default_rng(20260721)
        draws = np.median(rng.choice(x, size=(100_000, len(x)), replace=True), axis=1)
    return med, float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def session_medians(data: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "runtime_per_op_s", "kernel_runtime_per_op_s", "primary_energy_per_op_j",
        "total_energy_per_op_j", "dram_energy_per_op_j", "primary_power_w", "total_power_w",
        "edp_primary_j_s", "edp_total_j_s", "throughput_gflops_exact",
        "primary_efficiency_gflop_per_j", "logical_bandwidth_gb_s", "logical_gb_per_j",
        "temp_c", "clock_before_mhz", "clock_after_mhz", "clock_change_pct", "batches",
        "working_set_bytes", "operational_intensity_flop_per_byte",
    ]
    group_cols = [
        "platform", "platform_label", "device_kind", "energy_domain", "session_number",
        "problem_size", "configuration", "num_threads",
    ]
    return (
        data.groupby(group_cols, as_index=False, dropna=False)[metrics]
        .median(numeric_only=True)
        .sort_values(["problem_size", "num_threads", "session_number"])
        .reset_index(drop=True)
    )


def configuration_summary(session: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "runtime_per_op_s", "primary_energy_per_op_j", "total_energy_per_op_j",
        "primary_power_w", "edp_primary_j_s", "throughput_gflops_exact",
        "primary_efficiency_gflop_per_j", "logical_bandwidth_gb_s", "logical_gb_per_j",
        "temp_c", "clock_before_mhz",
    ]
    rows: list[dict] = []
    group_cols = [
        "platform", "platform_label", "device_kind", "energy_domain",
        "problem_size", "configuration", "num_threads",
    ]
    for keys, group in session.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        row["sessions"] = int(group.session_number.nunique())
        row["working_set_bytes"] = float(group.working_set_bytes.median())
        row["operational_intensity_flop_per_byte"] = float(group.operational_intensity_flop_per_byte.median())
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce").to_numpy(float)
            med, lo, hi = bootstrap_median_ci(values)
            row[f"{metric}_median"] = med
            row[f"{metric}_ci95_low"] = lo
            row[f"{metric}_ci95_high"] = hi
            row[f"{metric}_session_cv_pct"] = standard_cv(values)
            row[f"{metric}_robust_cv_pct"] = robust_cv(values)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["problem_size", "num_threads"]).reset_index(drop=True)


def intervals_overlap(a_lo: float, a_hi: float, b_lo: float, b_hi: float) -> bool:
    if not all(np.isfinite([a_lo, a_hi, b_lo, b_hi])):
        return True
    return not (a_hi < b_lo or b_hi < a_lo)


def select_leaders(summary: pd.DataFrame, metric: str, lower: bool,
                   tolerance: float = PRACTICAL_TOLERANCE) -> pd.DataFrame:
    rows: list[dict] = []
    for size, group in summary.groupby("problem_size"):
        group = group.copy().reset_index(drop=True)
        values = group[f"{metric}_median"].astype(float)
        best_pos = int(values.idxmin() if lower else values.idxmax())
        best = group.loc[best_pos]
        best_value = float(best[f"{metric}_median"])
        leader_configs: list[str] = []
        for _, row in group.iterrows():
            value = float(row[f"{metric}_median"])
            gap = value / best_value - 1.0 if lower else best_value / value - 1.0
            overlap = intervals_overlap(
                float(best[f"{metric}_ci95_low"]), float(best[f"{metric}_ci95_high"]),
                float(row[f"{metric}_ci95_low"]), float(row[f"{metric}_ci95_high"]),
            )
            if gap <= tolerance + 1e-15 or overlap:
                leader_configs.append(str(row.configuration))
        if len(group) == 1:
            gap_second = np.nan
            ci_sep = True
            clear = True
        else:
            sorted_values = np.sort(values) if lower else np.sort(values)[::-1]
            second = float(sorted_values[1])
            gap_second = 100.0 * (second / best_value - 1.0 if lower else best_value / second - 1.0)
            others = group.drop(index=best_pos)
            if lower:
                ci_sep = bool((float(best[f"{metric}_ci95_high"]) < others[f"{metric}_ci95_low"].astype(float)).all())
            else:
                ci_sep = bool((float(best[f"{metric}_ci95_low"]) > others[f"{metric}_ci95_high"].astype(float)).all())
            clear = len(leader_configs) == 1 and gap_second > 100.0 * tolerance and ci_sep
        row = {
            "problem_size": int(size),
            "metric": metric,
            "lower_is_better": lower,
            "exact_configuration": best.configuration,
            "exact_num_threads": int(best.num_threads),
            "leader_configurations": ",".join(leader_configs),
            "leader_count": len(leader_configs),
            "selection_status": "clear_leader" if clear else "tie_or_uncertain",
            "gap_to_second_pct": gap_second,
            "ci_separated_from_all": ci_sep,
        }
        for col in summary.columns:
            if col.endswith(("_median", "_ci95_low", "_ci95_high", "_session_cv_pct")):
                row[f"selected_{col}"] = best[col]
        rows.append(row)
    return pd.DataFrame(rows).sort_values("problem_size")


def strict_pareto(group: pd.DataFrame) -> pd.Series:
    values = group[["runtime_per_op_s_median", "primary_energy_per_op_j_median"]].to_numpy(float)
    keep = np.ones(len(group), dtype=bool)
    for i, point in enumerate(values):
        for j, other in enumerate(values):
            if i == j:
                continue
            if np.all(other <= point) and np.any(other < point):
                keep[i] = False
                break
    return pd.Series(keep, index=group.index)


def practical_pareto(group: pd.DataFrame, tolerance: float = PRACTICAL_TOLERANCE) -> pd.Series:
    values = group[["runtime_per_op_s_median", "primary_energy_per_op_j_median"]].to_numpy(float)
    keep = np.ones(len(group), dtype=bool)
    for i, point in enumerate(values):
        for j, other in enumerate(values):
            if i == j:
                continue
            dominates = np.all(other <= point / (1.0 + tolerance))
            if dominates:
                keep[i] = False
                break
    return pd.Series(keep, index=group.index)


def write_manifest(ctx: Context, campaign: Campaign) -> pd.DataFrame:
    rows = []
    for session, path in zip(campaign.sessions, campaign.files):
        frame = campaign.data[campaign.data.session_number == session]
        rows.append({
            "platform": ctx.platform,
            "campaign": campaign.stamp,
            "session_number": session,
            "file": path.name,
            "rows": len(frame),
            "sha256": sha256_file(path),
            "source_path": str(ctx.source_path),
            "source_sha256": sha256_file(ctx.source_path) if ctx.source_path.is_file() else "",
            "runner_path": str(ctx.runner_path),
            "runner_sha256": sha256_file(ctx.runner_path) if ctx.runner_path.is_file() else "",
        })
    manifest = pd.DataFrame(rows)
    manifest.to_csv(ctx.result_dir / "campaign_manifest.csv", index=False)
    (ctx.result_dir / "audited_source_path.txt").write_text(str(ctx.source_path) + "\n", encoding="utf-8")
    (ctx.result_dir / "audited_runner_path.txt").write_text(str(ctx.runner_path) + "\n", encoding="utf-8")
    return manifest


def plot_metric(summary: pd.DataFrame, metric: str, ylabel: str, output: Path,
                logx: bool = True, logy: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for configuration, group in summary.groupby("configuration"):
        group = group.sort_values("problem_size")
        ax.plot(group.problem_size, group[f"{metric}_median"], marker="o", label=configuration)
    if logx:
        ax.set_xscale("log", base=2)
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("Elemente N")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    if summary.configuration.nunique() > 1:
        ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)

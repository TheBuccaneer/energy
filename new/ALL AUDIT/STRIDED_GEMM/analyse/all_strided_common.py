#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import itertools
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

PLATFORMS = ("INTEL", "AMD", "3090", "5060ti")
CPU_PLATFORMS = ("INTEL", "AMD")
GPU_PLATFORMS = ("3090", "5060ti")
PLATFORM_LABELS = {
    "INTEL": "Intel CPU",
    "AMD": "AMD CPU",
    "3090": "RTX 3090",
    "5060ti": "RTX 5060 Ti",
}
DEVICE_KIND = {"INTEL": "CPU", "AMD": "CPU", "3090": "GPU", "5060ti": "GPU"}
ENERGY_DOMAIN = {
    "INTEL": "CPU package RAPL",
    "AMD": "CPU package RAPL",
    "3090": "GPU board NVML",
    "5060ti": "GPU board NVML",
}
SIZES = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
EXPECTED_SESSIONS = 5
PRACTICAL_TOLERANCE = 0.02
RATIO_BOOTSTRAP_DRAWS = 20_000

LOWER_IS_BETTER = {
    "runtime_s": True,
    "energy_j": True,
    "edp_j_s": True,
    "power_w": True,
    "throughput_gflops": False,
    "efficiency_gflop_per_j": False,
}

CPU_SESSION_COLUMNS = {
    "runtime_per_op_s": "runtime_s",
    "package_energy_per_op_j": "energy_j",
    "package_avg_power_w": "power_w",
    "throughput_gflops_exact": "throughput_gflops",
    "edp_package_j_s": "edp_j_s",
    "temp_c": "temperature_c",
    "clock_before_mhz": "clock_mhz",
}
GPU_SESSION_COLUMNS = {
    "runtime_per_op_s": "runtime_s",
    "device_energy_per_op_j": "energy_j",
    "avg_power_w": "power_w",
    "gflops_per_s": "throughput_gflops",
    "edp_j_s": "edp_j_s",
    "temp_c": "temperature_c",
    "sm_clock_mhz": "clock_mhz",
}


@dataclass(frozen=True)
class PlatformFiles:
    platform: str
    root: Path
    results: Path
    runs: Path
    validation_checks: Path
    validation_report: Path
    campaign_manifest: Path
    session_medians: Path
    summary: Path


def project_root(script_file: str | Path) -> Path:
    override = os.environ.get("ENERGY_PROJECT_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    # .../new/ALL AUDIT/STRIDED_GEMM/analyse/file.py -> .../new
    return Path(script_file).resolve().parents[3]


def audit_root(script_file: str | Path) -> Path:
    return project_root(script_file) / "ALL AUDIT" / "STRIDED_GEMM"


def results_dir(script_file: str | Path) -> Path:
    out = audit_root(script_file) / "results"
    (out / "figures").mkdir(parents=True, exist_ok=True)
    return out


def platform_files(script_file: str | Path, platform: str) -> PlatformFiles:
    if platform not in PLATFORMS:
        raise KeyError(platform)
    root = project_root(script_file) / platform
    results = root / "results" / "STRIDED_GEMM"
    runs = root / "runs" / "STRIDED_GEMM"
    if platform in CPU_PLATFORMS:
        session = results / "session_configuration_medians.csv"
        summary = results / "configuration_summary.csv"
    else:
        session = results / "session_medians_by_size.csv"
        summary = results / "size_summary.csv"
    return PlatformFiles(
        platform=platform,
        root=root,
        results=results,
        runs=runs,
        validation_checks=results / "validation_checks.csv",
        validation_report=results / "validation_report.md",
        campaign_manifest=results / "campaign_manifest.csv",
        session_medians=session,
        summary=summary,
    )


def required_input_paths(files: PlatformFiles) -> list[Path]:
    return [
        files.validation_checks,
        files.validation_report,
        files.campaign_manifest,
        files.session_medians,
        files.summary,
    ]


def add_check(
    checks: list[dict],
    category: str,
    check: str,
    severity: str,
    passed: bool,
    observed,
    expected,
    platform: str = "ALL",
) -> None:
    checks.append({
        "platform": platform,
        "category": category,
        "check": check,
        "severity": severity,
        "status": "PASS" if passed else severity,
        "observed": str(observed),
        "expected": str(expected),
    })


def markdown_table(frame: pd.DataFrame, max_rows: int = 200) -> str:
    if frame.empty:
        return "_None._"
    return frame.head(max_rows).to_markdown(index=False)


def normalize_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "t", "true", "yes", "y"}


def exact_bootstrap_median(values: Iterable[float], alpha: float = 0.05) -> tuple[float, float]:
    x = np.asarray([float(v) for v in values if np.isfinite(v)], dtype=float)
    if len(x) == 0:
        return np.nan, np.nan
    if len(x) > 7:
        rng = np.random.default_rng(20260720)
        draws = np.median(rng.choice(x, size=(100_000, len(x)), replace=True), axis=1)
    else:
        indices = np.asarray(list(itertools.product(range(len(x)), repeat=len(x))), dtype=int)
        draws = np.median(x[indices], axis=1)
    return float(np.quantile(draws, alpha / 2)), float(np.quantile(draws, 1 - alpha / 2))


def cv_pct(values: Iterable[float]) -> float:
    x = np.asarray([float(v) for v in values if np.isfinite(v)], dtype=float)
    if len(x) < 2 or np.mean(x) == 0:
        return np.nan
    return float(100.0 * np.std(x, ddof=1) / np.mean(x))


def stable_seed(*parts: object) -> int:
    text = "|".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(text).digest()[:8], "little") % (2**32)


def bootstrap_ratio_ci(
    values_a: Iterable[float],
    values_b: Iterable[float],
    draws: int = RATIO_BOOTSTRAP_DRAWS,
    seed_parts: tuple[object, ...] = (),
) -> tuple[float, float]:
    a = np.asarray([float(v) for v in values_a if np.isfinite(v)], dtype=float)
    b = np.asarray([float(v) for v in values_b if np.isfinite(v)], dtype=float)
    if len(a) == 0 or len(b) == 0 or np.any(b <= 0):
        return np.nan, np.nan
    rng = np.random.default_rng(stable_seed(*seed_parts))
    a_draw = np.median(rng.choice(a, size=(draws, len(a)), replace=True), axis=1)
    b_draw = np.median(rng.choice(b, size=(draws, len(b)), replace=True), axis=1)
    ratios = a_draw / b_draw
    return float(np.quantile(ratios, 0.025)), float(np.quantile(ratios, 0.975))


def cliffs_delta(values_a: Iterable[float], values_b: Iterable[float]) -> float:
    a = np.asarray([float(v) for v in values_a if np.isfinite(v)], dtype=float)
    b = np.asarray([float(v) for v in values_b if np.isfinite(v)], dtype=float)
    if len(a) == 0 or len(b) == 0:
        return np.nan
    gt = sum(x > y for x in a for y in b)
    lt = sum(x < y for x in a for y in b)
    return float((gt - lt) / (len(a) * len(b)))


def probability_a_better(values_a: Iterable[float], values_b: Iterable[float], lower: bool) -> float:
    a = np.asarray([float(v) for v in values_a if np.isfinite(v)], dtype=float)
    b = np.asarray([float(v) for v in values_b if np.isfinite(v)], dtype=float)
    if len(a) == 0 or len(b) == 0:
        return np.nan
    if lower:
        better = sum(x < y for x in a for y in b)
        ties = sum(x == y for x in a for y in b)
    else:
        better = sum(x > y for x in a for y in b)
        ties = sum(x == y for x in a for y in b)
    return float((better + 0.5 * ties) / (len(a) * len(b)))


def ci_overlap(lo_a: float, hi_a: float, lo_b: float, hi_b: float) -> bool:
    vals = [lo_a, hi_a, lo_b, hi_b]
    if not all(np.isfinite(vals)):
        return True
    return not (hi_a < lo_b or hi_b < lo_a)


def practical_gap(value: float, best: float, lower: bool) -> float:
    if not np.isfinite(value) or not np.isfinite(best) or best == 0:
        return np.nan
    if lower:
        return value / best - 1.0
    return best / value - 1.0


def select_leaders(group: pd.DataFrame, metric: str, lower: bool, tolerance: float = PRACTICAL_TOLERANCE) -> dict:
    if group.empty:
        return {
            "exact_index": None,
            "exact_configuration": "",
            "leader_configurations": "",
            "leader_count": 0,
            "selection_status": "missing",
            "gap_to_second_pct": np.nan,
            "practical_gap_gt_tolerance": False,
            "ci_separated_from_all": False,
        }
    data = group.reset_index(drop=False).copy()
    point = data[f"{metric}_median"].astype(float)

    if len(data) == 1:
        exact = data.iloc[0]
        return {
            "exact_index": int(exact["index"]),
            "exact_configuration": str(exact["configuration"]),
            "leader_configurations": str(exact["configuration"]),
            "leader_count": 1,
            "selection_status": "single_configuration",
            "gap_to_second_pct": np.nan,
            "practical_gap_gt_tolerance": False,
            "ci_separated_from_all": True,
        }
    exact_pos = int(point.idxmin() if lower else point.idxmax())
    exact = data.loc[exact_pos]
    best = float(exact[f"{metric}_median"])
    best_lo = float(exact[f"{metric}_ci95_low"])
    best_hi = float(exact[f"{metric}_ci95_high"])

    leaders = []
    for _, row in data.iterrows():
        value = float(row[f"{metric}_median"])
        gap = practical_gap(value, best, lower)
        overlap = ci_overlap(
            best_lo,
            best_hi,
            float(row[f"{metric}_ci95_low"]),
            float(row[f"{metric}_ci95_high"]),
        )
        if gap <= tolerance + 1e-15 or overlap:
            leaders.append(str(row["configuration"]))

    sorted_values = np.sort(point.to_numpy()) if lower else np.sort(point.to_numpy())[::-1]
    if len(sorted_values) >= 2:
        second = float(sorted_values[1])
        gap_second = 100.0 * practical_gap(second, best, lower)
    else:
        gap_second = np.nan

    others = data.drop(index=exact_pos)
    if others.empty:
        ci_sep = True
    elif lower:
        ci_sep = bool((best_hi < others[f"{metric}_ci95_low"].astype(float)).all())
    else:
        ci_sep = bool((best_lo > others[f"{metric}_ci95_high"].astype(float)).all())
    practical_sep = bool(np.isfinite(gap_second) and gap_second > 100.0 * tolerance)
    clear = len(leaders) == 1 and practical_sep and ci_sep

    return {
        "exact_index": int(exact["index"]),
        "exact_configuration": str(exact["configuration"]),
        "leader_configurations": ",".join(leaders),
        "leader_count": len(leaders),
        "selection_status": "clear_leader" if clear else "tie_or_uncertain",
        "gap_to_second_pct": gap_second,
        "practical_gap_gt_tolerance": practical_sep,
        "ci_separated_from_all": ci_sep,
    }


def classify_pairwise_ratio(
    ratio: float,
    ci_low: float,
    ci_high: float,
    platform_a: str,
    platform_b: str,
    lower: bool,
    tolerance: float = PRACTICAL_TOLERANCE,
) -> str:
    low_threshold = 1.0 / (1.0 + tolerance)
    high_threshold = 1.0 + tolerance
    if lower:
        if np.isfinite(ci_high) and ci_high < low_threshold:
            return f"clear_{platform_a}"
        if np.isfinite(ci_low) and ci_low > high_threshold:
            return f"clear_{platform_b}"
        if low_threshold <= ratio <= high_threshold:
            return "practically_equivalent_or_uncertain"
        return f"uncertain_{platform_a if ratio < 1 else platform_b}_advantage"
    # For a higher-is-better metric, A/B > 1 favors A.
    if np.isfinite(ci_low) and ci_low > high_threshold:
        return f"clear_{platform_a}"
    if np.isfinite(ci_high) and ci_high < low_threshold:
        return f"clear_{platform_b}"
    if low_threshold <= ratio <= high_threshold:
        return "practically_equivalent_or_uncertain"
    return f"uncertain_{platform_a if ratio > 1 else platform_b}_advantage"


def normalize_session_medians(script_file: str | Path, platform: str) -> pd.DataFrame:
    files = platform_files(script_file, platform)
    frame = pd.read_csv(files.session_medians)
    if platform in CPU_PLATFORMS:
        rename = CPU_SESSION_COLUMNS
        frame = frame.rename(columns=rename)
        frame["configuration"] = frame["num_threads"].astype(int).astype(str) + "T"
    else:
        rename = GPU_SESSION_COLUMNS
        frame = frame.rename(columns=rename)
        frame["num_threads"] = -1
        frame["configuration"] = "gpu_resident"
    frame["platform"] = platform
    frame["platform_label"] = PLATFORM_LABELS[platform]
    frame["device_kind"] = DEVICE_KIND[platform]
    frame["energy_domain"] = ENERGY_DOMAIN[platform]
    frame["problem_size"] = frame["problem_size"].astype(int)
    frame["session_number"] = frame["session_number"].astype(int)

    # Normalize work-derived metrics from the same primary per-operation values
    # used for placement. For fixed N, logical work is exactly 2*N^3 FLOP.
    # This avoids ratios of separately aggregated medians and guarantees:
    #   throughput == logical FLOP / e2e runtime
    #   efficiency == logical FLOP / measured energy
    logical_flops = 2.0 * frame["problem_size"].astype(float) ** 3
    frame["reported_throughput_gflops"] = pd.to_numeric(
        frame["throughput_gflops"], errors="coerce"
    )
    frame["throughput_gflops"] = logical_flops / frame["runtime_s"] / 1e9
    frame["efficiency_gflop_per_j"] = logical_flops / frame["energy_j"] / 1e9
    frame["throughput_normalization_delta_pct"] = 100.0 * (
        frame["reported_throughput_gflops"] / frame["throughput_gflops"] - 1.0
    )
    expected = [
        "platform", "platform_label", "device_kind", "energy_domain",
        "session_number", "problem_size", "configuration", "num_threads",
        "runtime_s", "energy_j", "power_w", "throughput_gflops",
        "efficiency_gflop_per_j", "edp_j_s", "temperature_c", "clock_mhz",
        "reported_throughput_gflops", "throughput_normalization_delta_pct",
    ]
    for column in expected:
        if column not in frame.columns:
            frame[column] = np.nan
    return frame[expected]


def summarize_configurations(session: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "runtime_s", "energy_j", "power_w", "throughput_gflops",
        "efficiency_gflop_per_j", "edp_j_s", "temperature_c", "clock_mhz",
    ]
    rows: list[dict] = []
    group_cols = [
        "platform", "platform_label", "device_kind", "energy_domain",
        "problem_size", "configuration", "num_threads",
    ]
    for keys, group in session.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        row["sessions"] = int(group["session_number"].nunique())
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce").to_numpy(float)
            finite = values[np.isfinite(values)]
            if len(finite):
                lo, hi = exact_bootstrap_median(finite)
                row[f"{metric}_median"] = float(np.median(finite))
                row[f"{metric}_ci95_low"] = lo
                row[f"{metric}_ci95_high"] = hi
                row[f"{metric}_session_cv_pct"] = cv_pct(finite)
            else:
                row[f"{metric}_median"] = np.nan
                row[f"{metric}_ci95_low"] = np.nan
                row[f"{metric}_ci95_high"] = np.nan
                row[f"{metric}_session_cv_pct"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["problem_size", "platform", "num_threads"]).reset_index(drop=True)


def find_complete_campaigns(run_dir: Path, platform: str) -> dict[str, list[Path]]:
    patterns = {
        "INTEL": re.compile(r"^strided_gemm_intel_(\d{8}_\d{6})_session([1-5])\.csv$", re.I),
        "AMD": re.compile(r"^strided_gemm_amd_(\d{8}_\d{6})_session([1-5])\.csv$", re.I),
        "3090": re.compile(r"^strided_gemm_3090_(\d{8}_\d{6})_session([1-5])\.csv$", re.I),
        "5060ti": re.compile(r"^strided_gemm_5060ti_(\d{8}_\d{6})_session([1-5])\.csv$", re.I),
    }
    pattern = patterns[platform]
    groups: dict[str, list[tuple[int, Path]]] = {}
    if not run_dir.is_dir():
        return {}
    for path in run_dir.glob("*.csv"):
        match = pattern.match(path.name)
        if match:
            groups.setdefault(match.group(1), []).append((int(match.group(2)), path))
    complete = {}
    for stamp, entries in groups.items():
        if sorted(s for s, _ in entries) == [1, 2, 3, 4, 5]:
            complete[stamp] = [p for _, p in sorted(entries)]
    return complete


def campaign_from_manifest(path: Path) -> str:
    frame = pd.read_csv(path)
    if frame.empty or "campaign" not in frame.columns:
        return ""
    return str(frame.iloc[0]["campaign"])


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def holm_adjust(p_values: Iterable[float]) -> np.ndarray:
    # Included for future fixed-configuration analyses; native-best reports do not
    # use p-values because selection and testing on the same five sessions would
    # overstate inferential certainty.
    p = np.asarray(list(p_values), dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    n = len(p)
    for rank, idx in enumerate(order):
        value = min(1.0, (n - rank) * p[idx])
        running = max(running, value)
        adjusted[idx] = running
    return adjusted

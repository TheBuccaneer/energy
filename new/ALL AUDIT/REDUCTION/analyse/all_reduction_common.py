#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import itertools
import math
import os
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

PLATFORMS = ("INTEL", "AMD", "3090", "5060ti")
CPU_PLATFORMS = ("INTEL", "AMD")
GPU_PLATFORMS = ("3090", "5060ti")
LABELS = {"INTEL": "Intel CPU", "AMD": "AMD CPU", "3090": "RTX 3090", "5060ti": "RTX 5060 Ti"}
KINDS = {"INTEL": "CPU", "AMD": "CPU", "3090": "GPU", "5060ti": "GPU"}
ENERGY_DOMAINS = {
    "INTEL": "CPU package RAPL", "AMD": "CPU package RAPL",
    "3090": "GPU board NVML", "5060ti": "GPU board NVML",
}
SIZES = [1_000_000, 2_000_000, 4_000_000, 8_000_000, 16_000_000, 32_000_000, 64_000_000, 128_000_000, 256_000_000]
EXPECTED_SESSIONS = 5
PRACTICAL_TOLERANCE = 0.02
RATIO_DRAWS = 20_000
LOWER_IS_BETTER = {
    "runtime_s": True,
    "energy_j": True,
    "edp_j_s": True,
    "logical_bandwidth_gb_s": False,
    "logical_gb_per_j": False,
}


def project_root(script_file: str | Path) -> Path:
    override = os.environ.get("ENERGY_PROJECT_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return Path(script_file).resolve().parents[3]


def results_dir(script_file: str | Path) -> Path:
    out = project_root(script_file) / "ALL AUDIT" / "REDUCTION" / "results"
    (out / "figures").mkdir(parents=True, exist_ok=True)
    return out


def platform_result_dir(script_file: str | Path, platform: str) -> Path:
    return project_root(script_file) / platform / "results" / "REDUCTION"


def platform_run_dir(script_file: str | Path, platform: str) -> Path:
    return project_root(script_file) / platform / "runs" / "REDUCTION"


def required_platform_outputs(script_file: str | Path, platform: str) -> list[Path]:
    base = platform_result_dir(script_file, platform)
    return [
        base / "validation_checks.csv",
        base / "validation_report.md",
        base / "campaign_manifest.csv",
        base / "session_configuration_medians.csv",
        base / "configuration_summary.csv",
        base / "policy_leaders.csv",
        base / "within_platform_energy_runtime_tradeoffs.csv",
    ]


def markdown_table(frame: pd.DataFrame, max_rows: int = 200) -> str:
    if frame.empty:
        return "_None._"
    return frame.head(max_rows).to_markdown(index=False)


def add_check(rows: list[dict], category: str, check: str, severity: str,
              passed: bool, observed, expected, platform: str = "ALL") -> None:
    rows.append({
        "platform": platform, "category": category, "check": check,
        "severity": severity, "status": "PASS" if passed else severity,
        "observed": str(observed), "expected": str(expected),
    })


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def campaign_pattern(platform: str) -> re.Pattern[str]:
    slug = platform.lower()
    return re.compile(rf"^reduction_{re.escape(slug)}_(\d{{8}}_\d{{6}})_session([1-5])\.csv$", re.I)


def complete_campaigns(script_file: str | Path, platform: str) -> dict[str, list[Path]]:
    groups: dict[str, list[tuple[int, Path]]] = {}
    run_dir = platform_run_dir(script_file, platform)
    if not run_dir.is_dir():
        return {}
    pattern = campaign_pattern(platform)
    for path in run_dir.glob("*.csv"):
        match = pattern.match(path.name)
        if match:
            groups.setdefault(match.group(1), []).append((int(match.group(2)), path))
    return {
        stamp: [p for _, p in sorted(entries)]
        for stamp, entries in groups.items()
        if sorted(s for s, _ in entries) == [1, 2, 3, 4, 5]
    }


def exact_bootstrap_median(values: Iterable[float]) -> tuple[float, float]:
    x = np.asarray([float(v) for v in values if np.isfinite(v)], dtype=float)
    if len(x) == 0:
        return np.nan, np.nan
    indices = np.asarray(list(itertools.product(range(len(x)), repeat=len(x))), dtype=int)
    draws = np.median(x[indices], axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def cv_pct(values: Iterable[float]) -> float:
    x = np.asarray([float(v) for v in values if np.isfinite(v)], dtype=float)
    if len(x) < 2 or np.mean(x) == 0:
        return np.nan
    return 100.0 * float(np.std(x, ddof=1) / abs(np.mean(x)))


def stable_seed(*parts: object) -> int:
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "little") % (2**32)


def bootstrap_ratio_ci(a: Iterable[float], b: Iterable[float], seed_parts: tuple[object, ...]) -> tuple[float, float]:
    va = np.asarray([float(v) for v in a if np.isfinite(v)], dtype=float)
    vb = np.asarray([float(v) for v in b if np.isfinite(v)], dtype=float)
    if len(va) == 0 or len(vb) == 0 or np.any(vb <= 0):
        return np.nan, np.nan
    rng = np.random.default_rng(stable_seed(*seed_parts))
    da = np.median(rng.choice(va, size=(RATIO_DRAWS, len(va)), replace=True), axis=1)
    db = np.median(rng.choice(vb, size=(RATIO_DRAWS, len(vb)), replace=True), axis=1)
    ratios = da / db
    return float(np.quantile(ratios, 0.025)), float(np.quantile(ratios, 0.975))


def cliffs_delta(a: Iterable[float], b: Iterable[float]) -> float:
    va = np.asarray(list(a), dtype=float)
    vb = np.asarray(list(b), dtype=float)
    gt = sum(x > y for x in va for y in vb)
    lt = sum(x < y for x in va for y in vb)
    return (gt - lt) / (len(va) * len(vb))


def probability_a_better(a: Iterable[float], b: Iterable[float], lower: bool) -> float:
    va = np.asarray(list(a), dtype=float)
    vb = np.asarray(list(b), dtype=float)
    if lower:
        better = sum(x < y for x in va for y in vb)
    else:
        better = sum(x > y for x in va for y in vb)
    ties = sum(x == y for x in va for y in vb)
    return (better + 0.5 * ties) / (len(va) * len(vb))


def classify_ratio(ratio: float, lo: float, hi: float, a: str, b: str, lower: bool) -> str:
    low = 1.0 / (1.0 + PRACTICAL_TOLERANCE)
    high = 1.0 + PRACTICAL_TOLERANCE
    if lower:
        if hi < low:
            return f"clear_{a}"
        if lo > high:
            return f"clear_{b}"
        if low <= ratio <= high:
            return "practically_equivalent_or_uncertain"
        return f"uncertain_{a if ratio < 1 else b}_advantage"
    if lo > high:
        return f"clear_{a}"
    if hi < low:
        return f"clear_{b}"
    if low <= ratio <= high:
        return "practically_equivalent_or_uncertain"
    return f"uncertain_{a if ratio > 1 else b}_advantage"


def intervals_overlap(a_lo: float, a_hi: float, b_lo: float, b_hi: float) -> bool:
    return not (a_hi < b_lo or b_hi < a_lo)


def select_leaders(group: pd.DataFrame, metric: str, lower: bool, key_column: str) -> dict:
    group = group.copy().reset_index(drop=True)
    if key_column not in group.columns:
        raise KeyError(f"Missing leader key column: {key_column}")
    values = group[f"{metric}_median"].astype(float)
    pos = int(values.idxmin() if lower else values.idxmax())
    best = group.loc[pos]
    best_value = float(best[f"{metric}_median"])
    leaders = []
    for _, row in group.iterrows():
        value = float(row[f"{metric}_median"])
        gap = value / best_value - 1 if lower else best_value / value - 1
        overlap = intervals_overlap(
            float(best[f"{metric}_ci95_low"]), float(best[f"{metric}_ci95_high"]),
            float(row[f"{metric}_ci95_low"]), float(row[f"{metric}_ci95_high"]),
        )
        if gap <= PRACTICAL_TOLERANCE + 1e-15 or overlap:
            leaders.append(str(row[key_column]))
    if len(group) == 1:
        gap_second = np.nan
        ci_sep = True
        clear = True
    else:
        sorted_values = np.sort(values) if lower else np.sort(values)[::-1]
        second = float(sorted_values[1])
        gap_second = 100.0 * (second / best_value - 1 if lower else best_value / second - 1)
        others = group.drop(index=pos)
        if lower:
            ci_sep = bool((float(best[f"{metric}_ci95_high"]) < others[f"{metric}_ci95_low"].astype(float)).all())
        else:
            ci_sep = bool((float(best[f"{metric}_ci95_low"]) > others[f"{metric}_ci95_high"].astype(float)).all())
        clear = len(leaders) == 1 and gap_second > 2.0 and ci_sep
    return {
        "exact_index": int(best.name),
        "exact_item": str(best[key_column]),
        "leader_items": ",".join(leaders),
        "leader_count": len(leaders),
        "selection_status": "clear_leader" if clear else "tie_or_uncertain",
        "gap_to_second_pct": gap_second,
        "ci_separated_from_all": ci_sep,
    }


def load_sessions(script_file: str | Path, platform: str) -> pd.DataFrame:
    path = platform_result_dir(script_file, platform) / "session_configuration_medians.csv"
    frame = pd.read_csv(path)
    frame["platform"] = platform
    frame["platform_label"] = LABELS[platform]
    frame["device_kind"] = KINDS[platform]
    frame["energy_domain"] = ENERGY_DOMAINS[platform]
    frame["problem_size"] = frame.problem_size.astype(int)
    frame["session_number"] = frame.session_number.astype(int)
    frame["configuration"] = frame.configuration.astype(str)

    # Recompute presentation metrics from primitive runtime and primary energy.
    n = frame.problem_size.astype(float)
    logical_flops = n - 1.0
    logical_bytes = 4.0 * n + 4.0
    frame["runtime_s"] = frame.runtime_per_op_s.astype(float)
    frame["energy_j"] = frame.primary_energy_per_op_j.astype(float)
    frame["total_energy_j"] = frame.total_energy_per_op_j.astype(float)
    frame["power_w"] = frame.primary_power_w.astype(float)
    frame["edp_j_s"] = frame.runtime_s * frame.energy_j
    frame["throughput_gflops"] = logical_flops / frame.runtime_s / 1e9
    frame["efficiency_gflop_per_j"] = logical_flops / frame.energy_j / 1e9
    frame["logical_bandwidth_gb_s"] = logical_bytes / frame.runtime_s / 1e9
    frame["logical_gb_per_j"] = logical_bytes / frame.energy_j / 1e9
    frame["temperature_c"] = frame.temp_c
    frame["clock_mhz"] = frame.clock_before_mhz
    expected = [
        "platform", "platform_label", "device_kind", "energy_domain", "session_number",
        "problem_size", "configuration", "num_threads", "runtime_s", "energy_j",
        "total_energy_j", "power_w", "edp_j_s", "throughput_gflops",
        "efficiency_gflop_per_j", "logical_bandwidth_gb_s", "logical_gb_per_j",
        "temperature_c", "clock_mhz", "working_set_bytes", "operational_intensity_flop_per_byte",
    ]
    return frame[expected]


def summarize_configurations(session: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "runtime_s", "energy_j", "total_energy_j", "power_w", "edp_j_s",
        "throughput_gflops", "efficiency_gflop_per_j", "logical_bandwidth_gb_s",
        "logical_gb_per_j", "temperature_c", "clock_mhz",
    ]
    group_cols = [
        "platform", "platform_label", "device_kind", "energy_domain",
        "problem_size", "configuration", "num_threads",
    ]
    rows = []
    for keys, group in session.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        row["sessions"] = int(group.session_number.nunique())
        row["working_set_bytes"] = float(group.working_set_bytes.median())
        row["operational_intensity_flop_per_byte"] = float(group.operational_intensity_flop_per_byte.median())
        for metric in metrics:
            values = group[metric].to_numpy(float)
            lo, hi = exact_bootstrap_median(values)
            row[f"{metric}_median"] = float(np.median(values))
            row[f"{metric}_ci95_low"] = lo
            row[f"{metric}_ci95_high"] = hi
            row[f"{metric}_session_cv_pct"] = cv_pct(values)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["problem_size", "platform", "num_threads"]).reset_index(drop=True)


def strict_pareto(group: pd.DataFrame) -> pd.Series:
    values = group[["runtime_s_median", "energy_j_median"]].to_numpy(float)
    keep = np.ones(len(group), dtype=bool)
    for i, point in enumerate(values):
        for j, other in enumerate(values):
            if i != j and np.all(other <= point) and np.any(other < point):
                keep[i] = False
                break
    return pd.Series(keep, index=group.index)


def practical_pareto(group: pd.DataFrame) -> pd.Series:
    values = group[["runtime_s_median", "energy_j_median"]].to_numpy(float)
    keep = np.ones(len(group), dtype=bool)
    for i, point in enumerate(values):
        for j, other in enumerate(values):
            if i != j and np.all(other <= point / (1.0 + PRACTICAL_TOLERANCE)):
                keep[i] = False
                break
    return pd.Series(keep, index=group.index)

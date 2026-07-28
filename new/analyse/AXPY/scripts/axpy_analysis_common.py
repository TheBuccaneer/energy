#!/usr/bin/env python3
"""Gemeinsame Hilfsfunktionen für die AXPY-Auswertung."""

from __future__ import annotations

import csv
import json
import math
import random
import re
import statistics as stats
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

SIZES = (
    1_000_000, 2_000_000, 4_000_000, 8_000_000, 16_000_000,
    32_000_000, 64_000_000, 128_000_000, 256_000_000,
)

HEADER_V2 = [
    "schema_version","timestamp","session_id","sequence_index","run_id_global",
    "repetition","workload","implementation","execution_mode","device_name",
    "num_threads","problem_size","problem_spec","batches","e2e_time_s",
    "kernel_time_s","wall_time_s","device_energy_j","total_energy_j",
    "dram_energy_j","energy_per_op_j","energy_per_second_j",
    "energy_per_flop_j","time_per_op_ms_kernel","time_per_op_ms_e2e",
    "flops_total","gflops_per_s","logical_bytes_per_op","avg_power_w",
    "runtime_status","pcie_gen","pcie_width","sm_clock_mhz",
    "clock_before_mhz","clock_after_mhz","mem_clock_mhz","temp_c",
    "temp_before_c","temp_after_c","throttle_reasons","cpu_cycles",
    "cpu_instructions","cpu_ipc","cpu_cache_misses","checksum_ok",
]

TRUE_VALUES = {"1", "t", "true", "yes", "ok"}
SESSION_RE = re.compile(r"^(?P<prefix>.+)_session(?P<session>[1-9][0-9]*)$")



EXPECTED_PROBLEM_SPEC_TEMPLATE = (
    "elements={n};alpha=3.0;x=period29*2^-16;"
    "y0=period31*2^-8;reset=outside_window;max_batches=250000"
)

THROTTLE_REASON_BITS = {
    0x0000000000000001: "gpu_idle",
    0x0000000000000002: "applications_clocks_setting",
    0x0000000000000004: "software_power_cap",
    0x0000000000000008: "hardware_slowdown",
    0x0000000000000010: "sync_boost",
    0x0000000000000020: "software_thermal_slowdown",
    0x0000000000000040: "hardware_thermal_slowdown",
    0x0000000000000080: "hardware_power_brake_slowdown",
    0x0000000000000100: "display_clock_setting",
}


def expected_problem_spec(n: int) -> str:
    return EXPECTED_PROBLEM_SPEC_TEMPLATE.format(n=n)


def sha256_file(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def valid_sha256(value: object) -> bool:
    return re.fullmatch(r"[0-9a-fA-F]{64}", str(value).strip()) is not None


def parse_key_value_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" not in line or line.lstrip().startswith("#"):
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def throttle_labels(mask: int) -> str:
    if mask == 0:
        return "none"
    labels = [name for bit, name in THROTTLE_REASON_BITS.items() if mask & bit]
    known_mask = 0
    for bit in THROTTLE_REASON_BITS:
        known_mask |= bit
    unknown = mask & ~known_mask
    if unknown:
        labels.append(f"unknown_bits_{unknown:#x}")
    return "|".join(labels) if labels else f"unknown_{mask:#x}"


def session_number(session_id: str) -> int:
    match = re.search(r"_session([1-9][0-9]*)$", session_id)
    if not match:
        raise ValueError(f"Sessionnummer nicht erkennbar: {session_id!r}")
    return int(match.group(1))


@dataclass(frozen=True)
class PlatformSpec:
    key: str
    label: str
    relative_dir: str
    kind: str
    threads: tuple[int, ...]
    execution_mode: str
    implementation: str
    device_token: str


PLATFORMS = (
    PlatformSpec(
        "intel", "Intel i9-7900X", "new/INTEL/runs/AXPY", "cpu",
        (1, 2, 4, 8, 10, 16, 20), "cpu_native",
        "openmp_axpy_inplace_fp32", "Intel",
    ),
    PlatformSpec(
        "amd", "AMD Threadripper 3970X", "new/AMD/runs/AXPY", "cpu",
        (1, 2, 4, 8, 10, 16, 20, 32, 64), "cpu_native",
        "openmp_axpy_inplace_fp32", "AMD",
    ),
    PlatformSpec(
        "3090", "RTX 3090", "new/3090/runs/AXPY", "gpu",
        (-1,), "gpu_resident", "cuda_axpy_inplace_fp32", "RTX 3090",
    ),
    PlatformSpec(
        "5060ti", "RTX 5060 Ti", "new/5060ti/runs/AXPY", "gpu",
        (-1,), "gpu_resident", "cuda_axpy_inplace_fp32", "RTX 5060 Ti",
    ),
)


@dataclass
class Issue:
    severity: str
    platform: str
    file: str
    row: int
    message: str


def is_true(value: object) -> bool:
    return str(value).strip().lower() in TRUE_VALUES


def finite_float(value: object) -> float:
    parsed = float(str(value).strip())
    if not math.isfinite(parsed):
        raise ValueError(f"nicht-endlicher Wert {parsed}")
    return parsed


def exact_int(value: object) -> int:
    text = str(value).strip()
    number = float(text)
    if not math.isfinite(number) or not number.is_integer():
        raise ValueError(f"kein exakter Integer: {value!r}")
    return int(number)


def close(a: float, b: float, rel: float = 5e-9, abs_: float = 5e-12) -> bool:
    return math.isclose(a, b, rel_tol=rel, abs_tol=abs_)


def median(values: Sequence[float]) -> float:
    return stats.median(values) if values else math.nan


def mad(values: Sequence[float]) -> float:
    if not values:
        return math.nan
    m = stats.median(values)
    return stats.median(abs(x - m) for x in values)


def cv(values: Sequence[float]) -> float:
    if len(values) < 2:
        return math.nan
    mean = stats.fmean(values)
    return stats.stdev(values) / mean if mean else math.nan


def percentile(sorted_values: Sequence[float], p: float) -> float:
    if not sorted_values:
        return math.nan
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = (len(sorted_values) - 1) * p
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(sorted_values[lower])
    fraction = position - lower
    return float(sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction)


def bootstrap_median_ci(
    values: Sequence[float],
    *,
    confidence: float = 0.95,
    resamples: int = 5000,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile-Bootstrap über Session-Mediane; deterministisch."""
    if not values:
        return math.nan, math.nan
    if len(values) == 1:
        return float(values[0]), float(values[0])
    rng = random.Random(seed)
    n = len(values)
    samples = []
    for _ in range(resamples):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        samples.append(stats.median(sample))
    samples.sort()
    alpha = (1.0 - confidence) / 2.0
    return percentile(samples, alpha), percentile(samples, 1.0 - alpha)


def campaign_candidates(directory: Path) -> dict[str, list[tuple[int, Path]]]:
    groups: dict[str, list[tuple[int, Path]]] = defaultdict(list)
    if not directory.is_dir():
        return groups
    for path in directory.glob("*.csv"):
        if "quickcheck" in path.name.lower():
            continue
        match = SESSION_RE.match(path.stem)
        if match:
            groups[match.group("prefix")].append((int(match.group("session")), path))
    return groups


def select_campaign(
    directory: Path,
    expected_sessions: int,
    campaign_id: str | None = None,
) -> tuple[str, list[tuple[int, Path]], bool]:
    """Select an explicit frozen campaign or, if omitted, the latest complete one."""
    groups = campaign_candidates(directory)
    if not groups:
        raise FileNotFoundError(f"Keine offiziellen *_sessionN.csv-Dateien in {directory}")

    wanted = set(range(1, expected_sessions + 1))
    if campaign_id is not None:
        if campaign_id not in groups:
            available = ", ".join(sorted(groups))
            raise FileNotFoundError(
                f"Eingefrorene Kampagne {campaign_id!r} fehlt in {directory}; "
                f"verfügbar: {available or 'keine'}"
            )
        entries = sorted(groups[campaign_id])
        sessions = {session for session, _ in entries}
        return campaign_id, entries, sessions == wanted

    ranked = []
    for prefix, entries in groups.items():
        sessions = {session for session, _ in entries}
        newest = max(path.stat().st_mtime for _, path in entries)
        complete = sessions == wanted
        ranked.append((complete, len(sessions), newest, prefix, sorted(entries)))

    complete, _, _, prefix, entries = max(
        ranked, key=lambda item: (item[0], item[1], item[2])
    )
    return prefix, entries, complete


def select_latest_complete_campaign(
    directory: Path,
    expected_sessions: int,
) -> tuple[str, list[tuple[int, Path]], bool]:
    return select_campaign(directory, expected_sessions, campaign_id=None)


def write_csv(path: Path, rows: Iterable[dict], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def format_float(value: float, digits: int = 6) -> str:
    return "n/a" if not math.isfinite(value) else f"{value:.{digits}g}"


def fmt_percent(value: float) -> str:
    return "n/a" if not math.isfinite(value) else f"{100.0 * value:.2f} %"


def parse_throttle(value: object) -> int:
    text = str(value).strip().lower()
    if not text.startswith("0x"):
        raise ValueError(f"ungültige Throttle-Maske {value!r}")
    return int(text, 16)


def stability_class(value: float) -> str:
    if not math.isfinite(value):
        return "unknown"
    if value <= 0.05:
        return "stable"
    if value <= 0.10:
        return "elevated"
    return "high_variability"

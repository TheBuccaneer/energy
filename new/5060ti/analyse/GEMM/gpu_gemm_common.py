#!/usr/bin/env python3
from __future__ import annotations

import itertools
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

SIZES = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
EXPECTED_SESSIONS = 5
EXPECTED_REPETITIONS = 10
TARGET_LOW_S = 0.75
TARGET_HIGH_S = 1.25

GPU_COLUMNS = [
    'schema_version','timestamp','session_id','sequence_index','run_id_global','repetition',
    'workload','implementation','execution_mode','device_name','num_threads','problem_size',
    'problem_spec','batches','e2e_time_s','kernel_time_s','wall_time_s','device_energy_j',
    'total_energy_j','dram_energy_j','energy_per_op_j','energy_per_second_j',
    'energy_per_flop_j','time_per_op_ms_kernel','time_per_op_ms_e2e','flops_total',
    'gflops_per_s','logical_bytes_per_op','avg_power_w','runtime_status','pcie_gen',
    'pcie_width','sm_clock_mhz','clock_before_mhz','clock_after_mhz','mem_clock_mhz',
    'temp_c','temp_before_c','temp_after_c','throttle_reasons','cpu_cycles',
    'cpu_instructions','cpu_ipc','cpu_cache_misses','checksum_ok',
]
CANONICAL_V2_COLUMNS = GPU_COLUMNS.copy()
i = CANONICAL_V2_COLUMNS.index('device_energy_j')
j = CANONICAL_V2_COLUMNS.index('total_energy_j')
CANONICAL_V2_COLUMNS[i], CANONICAL_V2_COLUMNS[j] = CANONICAL_V2_COLUMNS[j], CANONICAL_V2_COLUMNS[i]

NUMERIC = [
    'sequence_index','run_id_global','repetition','num_threads','problem_size','batches',
    'e2e_time_s','kernel_time_s','wall_time_s','device_energy_j','total_energy_j',
    'dram_energy_j','energy_per_op_j','energy_per_second_j','energy_per_flop_j',
    'time_per_op_ms_kernel','time_per_op_ms_e2e','flops_total','gflops_per_s',
    'logical_bytes_per_op','avg_power_w','pcie_gen','pcie_width','sm_clock_mhz',
    'clock_before_mhz','clock_after_mhz','mem_clock_mhz','temp_c','temp_before_c',
    'temp_after_c','cpu_cycles','cpu_instructions','cpu_ipc','cpu_cache_misses',
]
THROTTLE_BITS = {
    0x1:'gpu_idle', 0x2:'applications_clocks', 0x4:'software_power_cap',
    0x8:'hardware_slowdown', 0x10:'sync_boost', 0x20:'software_thermal',
    0x40:'hardware_thermal', 0x80:'hardware_power_brake', 0x100:'display_clock_setting',
}
SERIOUS_THROTTLE_MASK = 0x8 | 0x20 | 0x40 | 0x80

@dataclass
class Campaign:
    stamp: str
    files: list[Path]
    sessions: list[int]
    dataframe: pd.DataFrame


def project_root(script_file: str | Path) -> Path:
    return Path(script_file).resolve().parents[3]


def platform_root(script_file: str | Path) -> Path:
    return project_root(script_file) / '5060ti'


def results_dir(script_file: str | Path) -> Path:
    out = platform_root(script_file) / 'results' / 'GEMM'
    (out / 'figures').mkdir(parents=True, exist_ok=True)
    return out


def find_run_dir(root5060ti: Path) -> Path:
    candidates = [
        root5060ti/'runs'/'GEMM'/'GPU'/'RTX_5060_Ti',
        root5060ti/'runs'/'GEMM',
        root5060ti/'runs',
    ]
    for path in candidates:
        if path.is_dir() and any(path.glob('gemm_5060ti_*_session*.csv')):
            return path
    return candidates[0]


def campaign_groups(run_dir: Path) -> dict[str, list[tuple[int, Path]]]:
    pattern = re.compile(r'^gemm_5060ti_(\d{8}_\d{6})_session(\d+)\.csv$', re.I)
    groups: dict[str, list[tuple[int, Path]]] = {}
    for path in sorted(run_dir.glob('*.csv')):
        match = pattern.match(path.name)
        if match:
            groups.setdefault(match.group(1), []).append((int(match.group(2)), path))
    return groups


def select_campaign(script_file: str | Path, requested: str | None = None):
    run_dir = find_run_dir(platform_root(script_file))
    groups = campaign_groups(run_dir)
    if not groups:
        raise FileNotFoundError(
            f'Keine offizielle RTX-5060-Ti-Kampagne in {run_dir}. Erwartet: '
            'gemm_5060ti_<YYYYMMDD_HHMMSS>_session1.csv ... session5.csv. '
            'Quickcheck-Dateien werden absichtlich ignoriert.'
        )
    if requested:
        if requested not in groups:
            raise ValueError(f'Kampagne {requested} nicht gefunden; verfügbar: {sorted(groups)}')
        return requested, sorted(groups[requested]), run_dir
    complete = {
        stamp: entries for stamp, entries in groups.items()
        if sorted(s for s, _ in entries) == list(range(1, EXPECTED_SESSIONS + 1))
    }
    chosen = max(complete) if complete else max(groups)
    return chosen, sorted(groups[chosen]), run_dir


def truthy(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({'1','t','true','yes','y'})


def load_campaign(script_file: str | Path, requested: str | None = None) -> Campaign:
    stamp, entries, _ = select_campaign(script_file, requested)
    frames = []
    for session, path in entries:
        frame = pd.read_csv(path)
        frame['source_file'] = path.name
        frame['session_number'] = session
        frame['checksum_bool'] = truthy(frame.get('checksum_ok', pd.Series(False, index=frame.index)))
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True)
    for column in NUMERIC:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors='coerce')
    return Campaign(stamp, [p for _, p in entries], [s for s, _ in entries], df)


def parse_throttle_mask(value) -> int:
    try:
        text = str(value).strip()
        return int(text, 16) if text.lower().startswith('0x') else int(float(text))
    except Exception:
        return 0


def throttle_labels(mask: int) -> str:
    names = [name for bit, name in THROTTLE_BITS.items() if mask & bit]
    unknown = mask & ~sum(THROTTLE_BITS.keys())
    if unknown:
        names.append(f'unknown_0x{unknown:X}')
    return '|'.join(names) if names else 'none'


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out['runtime_per_op_s'] = out.e2e_time_s / out.batches
    out['kernel_runtime_per_op_s'] = out.kernel_time_s / out.batches
    out['device_energy_per_op_j'] = out.device_energy_j / out.batches
    out['edp_j_s'] = out.device_energy_per_op_j * out.runtime_per_op_s
    out['clock_change_pct'] = np.where(
        out.clock_before_mhz > 0,
        100.0 * (out.clock_after_mhz - out.clock_before_mhz) / out.clock_before_mhz,
        np.nan,
    )
    out['temperature_rise_c'] = out.temp_after_c - out.temp_before_c
    out['kernel_fraction'] = out.kernel_time_s / out.e2e_time_s
    out['throttle_mask'] = out.throttle_reasons.map(parse_throttle_mask)
    out['serious_throttle'] = (out.throttle_mask & SERIOUS_THROTTLE_MASK) != 0
    return out


def rel_close(actual, expected, rtol=1e-6, atol=1e-12) -> pd.Series:
    a = pd.to_numeric(actual, errors='coerce').astype(float)
    e = pd.to_numeric(expected, errors='coerce').astype(float)
    return np.isfinite(a) & np.isfinite(e) & (np.abs(a-e) <= atol + rtol*np.abs(e))


def add_check(checks, category, check, severity, passed, observed, expected) -> None:
    checks.append({
        'category':category, 'check':check, 'severity':severity,
        'status':'PASS' if passed else severity,
        'observed':str(observed), 'expected':str(expected),
    })


def exact_bootstrap_ci(values, alpha=0.05):
    x = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if len(x) == 0:
        return np.nan, np.nan
    stats = [np.median(x[list(idx)]) for idx in itertools.product(range(len(x)), repeat=len(x))]
    return float(np.quantile(stats, alpha/2)), float(np.quantile(stats, 1-alpha/2))


def robust_outlier_mask(group: pd.DataFrame, columns: list[str]) -> pd.Series:
    flag = pd.Series(False, index=group.index)
    for column in columns:
        x = pd.to_numeric(group[column], errors='coerce')
        median = x.median()
        mad = (x-median).abs().median()
        if not np.isfinite(mad) or mad == 0:
            continue
        flag |= (0.67448975 * (x-median) / mad).abs() > 3.5
    return flag


def cv_pct(values) -> float:
    x = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if len(x) < 2 or np.mean(x) == 0:
        return np.nan
    return float(100.0 * np.std(x, ddof=1) / np.mean(x))


def markdown_table(frame: pd.DataFrame, max_rows=100) -> str:
    return '_None._' if frame.empty else frame.head(max_rows).to_markdown(index=False)


def locate_source(root5060ti: Path) -> Path | None:
    for path in [root5060ti/'scripts'/'GEMM'/'main_gemm.cu', root5060ti/'scripts'/'GEMM'/'GPU'/'main_gemm.cu']:
        if path.is_file():
            return path
    return None


def locate_runner(root5060ti: Path) -> Path | None:
    for path in [root5060ti/'02_run_GPU_5060ti_GEMM_only.sh', root5060ti/'scripts'/'02_run_GPU_5060ti_GEMM_only.sh']:
        if path.is_file():
            return path
    return None

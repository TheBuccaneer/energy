#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCHEMA_VERSION = "cpu-gpu-v2"
EXPECTED_SESSIONS = 5
EXPECTED_REPETITIONS = 10
PRACTICAL_TOLERANCE = 0.02
SHAPES = {
    1: dict(N=32, C=64, H=56, W=56, K=64, R=3, S=3, stride=1, pad=1),
    2: dict(N=32, C=64, H=56, W=56, K=128, R=3, S=3, stride=2, pad=1),
    3: dict(N=32, C=128, H=28, W=28, K=256, R=3, S=3, stride=2, pad=1),
    4: dict(N=32, C=256, H=14, W=14, K=512, R=3, S=3, stride=2, pad=1),
    5: dict(N=32, C=3, H=224, W=224, K=64, R=7, S=7, stride=2, pad=3),
    6: dict(N=32, C=256, H=56, W=56, K=256, R=1, S=1, stride=1, pad=0),
}
EXPECTED_COLUMNS = [
    "schema_version","timestamp","session_id","sequence_index","run_id_global",
    "repetition","workload","implementation","execution_mode","device_name",
    "num_threads","problem_size","problem_spec","batches","e2e_time_s",
    "kernel_time_s","wall_time_s","device_energy_j","total_energy_j",
    "dram_energy_j","energy_per_op_j","energy_per_second_j","energy_per_flop_j",
    "time_per_op_ms_kernel","time_per_op_ms_e2e","flops_total","gflops_per_s",
    "logical_bytes_per_op","avg_power_w","runtime_status","pcie_gen","pcie_width",
    "sm_clock_mhz","clock_before_mhz","clock_after_mhz","mem_clock_mhz","temp_c",
    "temp_before_c","temp_after_c","throttle_reasons","cpu_cycles",
    "cpu_instructions","cpu_ipc","cpu_cache_misses","checksum_ok",
]
PLATFORM_CONFIG = {
    "AMD": {
        "label": "AMD Threadripper 3970X", "kind": "CPU", "slug": "amd",
        "threads": [1,2,4,8,10,16,20,32,64], "mode": "cpu_native",
        "implementation_prefix": "onednn_convolution_auto:",
        "energy_domain": "CPU package RAPL plus DRAM RAPL when available",
        "source_rel": "scripts/CONV2D/main_conv2d_amd.cpp",
        "runner_candidates": ["scripts/02_run_CPU_AMD_CONV2D_only.sh", "02_run_CPU_AMD_CONV2D_only.sh"],
        "device_tokens": ["3970X", "Threadripper"],
    },
    "INTEL": {
        "label": "Intel Core i9-7900X", "kind": "CPU", "slug": "intel",
        "threads": [1,2,4,8,10,16,20], "mode": "cpu_native",
        "implementation_prefix": "onednn_convolution_auto:",
        "energy_domain": "CPU package RAPL plus DRAM RAPL when available",
        "source_rel": "scripts/CONV2D/main_conv2d_intel.cpp",
        "runner_candidates": ["scripts/02_run_CPU_Intel_CONV2D_only.sh", "02_run_CPU_Intel_CONV2D_only.sh"],
        "device_tokens": ["7900X", "Intel"],
    },
    "3090": {
        "label": "RTX 3090", "kind": "GPU", "slug": "3090", "threads": [-1],
        "mode": "gpu_resident", "implementation_prefix": "cudnn_convolution_fwd_fp32",
        "energy_domain": "GPU board NVML TotalEnergyConsumption",
        "source_rel": "scripts/CONV2D/main_conv2d.cu",
        "runner_candidates": ["02_run_GPU_3090_CONV2D_only.sh", "scripts/02_run_GPU_3090_CONV2D_only.sh"],
        "device_tokens": ["3090"],
    },
    "5060ti": {
        "label": "RTX 5060 Ti", "kind": "GPU", "slug": "5060ti", "threads": [-1],
        "mode": "gpu_resident", "implementation_prefix": "cudnn_convolution_fwd_fp32",
        "energy_domain": "GPU board NVML TotalEnergyConsumption",
        "source_rel": "scripts/CONV2D/main_conv2d.cu",
        "runner_candidates": ["scripts/02_run_GPU_5060ti_CONV2D_only.sh", "02_run_GPU_5060ti_CONV2D_only.sh"],
        "device_tokens": ["5060", "5060 Ti"],
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
    runner_path: Path | None

@dataclass
class Campaign:
    stamp: str
    files: list[Path]
    sessions: list[int]
    data: pd.DataFrame


def geometry(shape_id: int) -> dict:
    g = dict(SHAPES[shape_id])
    g["Hout"] = (g["H"] + 2*g["pad"] - g["R"]) // g["stride"] + 1
    g["Wout"] = (g["W"] + 2*g["pad"] - g["S"]) // g["stride"] + 1
    return g


def flops_per_op(shape_id: int) -> int:
    g = geometry(shape_id)
    return 2*g["N"]*g["K"]*g["C"]*g["R"]*g["S"]*g["Hout"]*g["Wout"]


def logical_bytes_per_op(shape_id: int) -> int:
    g = geometry(shape_id)
    inp = g["N"]*g["C"]*g["H"]*g["W"]
    weights = g["K"]*g["C"]*g["R"]*g["S"]
    out = g["N"]*g["K"]*g["Hout"]*g["Wout"]
    return 4*(inp + weights + out)


def context(script_file: str | Path) -> Context:
    script = Path(script_file).resolve()
    platform_root = script.parents[2]
    platform = platform_root.name
    if platform not in PLATFORM_CONFIG:
        raise RuntimeError(f"Unsupported platform directory: {platform}")
    project_root = platform_root.parent
    cfg = PLATFORM_CONFIG[platform]
    result_dir = platform_root / "results" / "CONV2D"
    figure_dir = result_dir / "figures"
    result_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    runner = next((platform_root / rel for rel in cfg["runner_candidates"] if (platform_root / rel).is_file()), None)
    return Context(
        project_root=project_root, platform_root=platform_root, platform=platform,
        config=cfg, run_dir=platform_root / "runs" / "CONV2D",
        result_dir=result_dir, figure_dir=figure_dir,
        source_path=platform_root / cfg["source_rel"], runner_path=runner,
    )


def parse_args(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--campaign", help="Campaign timestamp YYYYMMDD_HHMMSS; default latest complete campaign")
    return parser.parse_args()


def campaign_pattern(platform: str) -> re.Pattern[str]:
    slug = PLATFORM_CONFIG[platform]["slug"]
    return re.compile(rf"^conv2d_{re.escape(slug)}_(\d{{8}}_\d{{6}})_session([1-5])\.csv$", re.I)


def discover_campaigns(run_dir: Path, platform: str) -> dict[str, list[tuple[int, Path]]]:
    groups: dict[str, list[tuple[int, Path]]] = {}
    if not run_dir.is_dir():
        return groups
    pat = campaign_pattern(platform)
    for path in run_dir.glob("*.csv"):
        m = pat.match(path.name)
        if m:
            groups.setdefault(m.group(1), []).append((int(m.group(2)), path))
    return groups


def select_campaign(run_dir: Path, platform: str, requested: str | None) -> tuple[str, list[tuple[int, Path]]]:
    groups = discover_campaigns(run_dir, platform)
    complete = {stamp: sorted(entries) for stamp, entries in groups.items()
                if sorted(s for s, _ in entries) == [1,2,3,4,5]}
    if requested:
        if requested not in complete:
            raise RuntimeError(f"Campaign {requested!r} is not complete. Complete campaigns: {sorted(complete)}")
        return requested, complete[requested]
    if not complete:
        partial = {stamp: sorted(s for s, _ in entries) for stamp, entries in groups.items()}
        raise RuntimeError(f"No complete five-session CONV2D campaign in {run_dir}. Partial: {partial}")
    stamp = max(complete)
    return stamp, complete[stamp]


def normalize_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"1","t","true","yes","y","ok"})


def load_campaign(ctx: Context, requested: str | None = None) -> Campaign:
    stamp, entries = select_campaign(ctx.run_dir, ctx.platform, requested)
    frames, files, sessions = [], [], []
    for session, path in entries:
        frame = pd.read_csv(path)
        frame["source_file"] = path.name
        frame["session_number"] = session
        frames.append(frame); files.append(path); sessions.append(session)
    data = pd.concat(frames, ignore_index=True, sort=False)
    numeric = [
        "sequence_index","run_id_global","repetition","num_threads","problem_size","batches",
        "e2e_time_s","kernel_time_s","wall_time_s","device_energy_j","total_energy_j","dram_energy_j",
        "energy_per_op_j","energy_per_second_j","energy_per_flop_j","time_per_op_ms_kernel",
        "time_per_op_ms_e2e","flops_total","gflops_per_s","logical_bytes_per_op","avg_power_w",
        "pcie_gen","pcie_width","sm_clock_mhz","clock_before_mhz","clock_after_mhz","mem_clock_mhz",
        "temp_c","temp_before_c","temp_after_c","cpu_cycles","cpu_instructions","cpu_ipc",
        "cpu_cache_misses","session_number",
    ]
    for col in numeric:
        if col in data:
            data[col] = pd.to_numeric(data[col], errors="coerce")
    data["checksum_bool"] = normalize_bool(data["checksum_ok"]) if "checksum_ok" in data else False
    return Campaign(stamp, files, sessions, data)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_spec(text: str) -> dict[str, str]:
    fields = {}
    for token in str(text).split(";"):
        if "=" in token:
            k, v = token.split("=", 1)
            fields[k.strip()] = v.strip()
    return fields


def add_check(checks: list[dict], category: str, name: str, severity: str,
              passed: bool, observed, expected) -> None:
    checks.append({"category": category, "check": name, "severity": severity,
                   "status": "PASS" if passed else severity,
                   "observed": str(observed), "expected": str(expected)})


def close_array(actual, expected, rtol=1e-8, atol=1e-10) -> np.ndarray:
    a = np.asarray(actual, dtype=float); e = np.asarray(expected, dtype=float)
    return np.isfinite(a) & np.isfinite(e) & (np.abs(a-e) <= atol + rtol*np.abs(e))


def validate_campaign(ctx: Context, campaign: Campaign) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = campaign.data
    cfg = ctx.config
    checks: list[dict] = []
    failures: list[pd.DataFrame] = []
    expected_configs = len(SHAPES) * len(cfg["threads"])
    expected_rows_session = expected_configs * EXPECTED_REPETITIONS

    add_check(checks,"coverage","five_sessions","FAIL",sorted(campaign.sessions)==[1,2,3,4,5],campaign.sessions,[1,2,3,4,5])
    add_check(checks,"schema","exact_header","FAIL",list(d.columns[:45])==EXPECTED_COLUMNS,list(d.columns[:45]),EXPECTED_COLUMNS)
    add_check(checks,"schema","schema_version","FAIL",set(d.schema_version.astype(str))=={SCHEMA_VERSION},sorted(set(d.schema_version.astype(str))),SCHEMA_VERSION)
    for s in range(1,6):
        add_check(checks,"coverage",f"session_{s}_row_count","FAIL",len(d[d.session_number==s])==expected_rows_session,len(d[d.session_number==s]),expected_rows_session)
    add_check(checks,"semantics","workload","FAIL",set(d.workload.astype(str))=={"CONV2D"},sorted(set(d.workload.astype(str))),"CONV2D")
    impl_ok = d.implementation.astype(str).str.startswith(cfg["implementation_prefix"])
    add_check(checks,"semantics","implementation","FAIL",bool(impl_ok.all()),int((~impl_ok).sum()),cfg["implementation_prefix"]+"*")
    add_check(checks,"semantics","execution_mode","FAIL",set(d.execution_mode.astype(str))=={cfg["mode"]},sorted(set(d.execution_mode.astype(str))),cfg["mode"])
    device_values = d.device_name.astype(str).str.strip()
    token_ok = device_values.apply(lambda x: any(t.lower() in x.lower() for t in cfg["device_tokens"]))
    add_check(checks,"semantics","device_identity","FAIL",device_values.nunique()==1 and bool(token_ok.all()),sorted(device_values.unique()),cfg["device_tokens"])
    add_check(checks,"coverage","shape_ids","FAIL",sorted(d.problem_size.dropna().astype(int).unique())==list(SHAPES),sorted(d.problem_size.dropna().astype(int).unique()),list(SHAPES))
    add_check(checks,"coverage","thread_grid","FAIL",sorted(d.num_threads.dropna().astype(int).unique())==cfg["threads"],sorted(d.num_threads.dropna().astype(int).unique()),cfg["threads"])

    key = ["session_number","problem_size","num_threads","repetition"]
    dup = d.duplicated(key, keep=False)
    add_check(checks,"coverage","no_duplicates","FAIL",not bool(dup.any()),int(dup.sum()),0)
    groups = d.groupby(["session_number","problem_size","num_threads"])["repetition"].agg(["count","nunique","min","max"])
    rep_ok = (groups["count"].eq(10)&groups["nunique"].eq(10)&groups["min"].eq(1)&groups["max"].eq(10))
    add_check(checks,"coverage","ten_repetitions_per_configuration","FAIL",bool(rep_ok.all()) and len(groups)==5*expected_configs,{"bad":int((~rep_ok).sum()),"groups":len(groups)},5*expected_configs)

    seq_bad = 0
    for _, f in d.groupby("session_number", sort=False):
        seq_bad += int(not np.array_equal(f.sort_values("sequence_index").sequence_index.to_numpy(int), np.arange(1,len(f)+1)))
        seq_bad += int(not np.array_equal(f.sequence_index.to_numpy(), f.run_id_global.to_numpy()))
    add_check(checks,"coverage","sequence_and_run_ids","FAIL",seq_bad==0,seq_bad,0)

    spec_bad = []
    for idx, row in d.iterrows():
        sid = int(row.problem_size)
        fields = parse_spec(row.problem_spec)
        g = geometry(sid)
        expected = {k:str(v) for k,v in g.items()}
        if cfg["kind"] == "GPU": expected["shape_id"] = str(sid)
        bad = [k for k,v in expected.items() if fields.get(k) != v]
        if fields.get("dtype") != "f32": bad.append("dtype")
        if cfg["kind"] == "GPU":
            for k,v in {"layout":"NCHW","conv":"cross_correlation","math":"FMA"}.items():
                if fields.get(k) != v: bad.append(k)
            if not fields.get("algo") or fields.get("workspace_bytes") is None: bad.append("algo/workspace")
        else:
            for k,v in {"input_layout":"NCHW","weight_layout":"OIHW","output_layout":"NCHW","algorithm_policy":"convolution_auto"}.items():
                if fields.get(k) != v: bad.append(k)
        if bad: spec_bad.append({"row":idx,"source_file":row.source_file,"shape":sid,"bad_fields":",".join(sorted(set(bad)))})
    add_check(checks,"semantics","problem_spec_geometry","FAIL",len(spec_bad)==0,len(spec_bad),0)
    if spec_bad: failures.append(pd.DataFrame(spec_bad))
    spec_stable = d.groupby(["session_number","problem_size","num_threads"]).problem_spec.nunique().le(1)
    add_check(checks,"semantics","plan_or_primitive_stable_within_configuration","FAIL",bool(spec_stable.all()),int((~spec_stable).sum()),0)

    positive_cols = ["batches","e2e_time_s","kernel_time_s","wall_time_s","device_energy_j","total_energy_j","energy_per_op_j","energy_per_second_j","energy_per_flop_j","time_per_op_ms_kernel","time_per_op_ms_e2e","flops_total","gflops_per_s","logical_bytes_per_op","avg_power_w"]
    for col in positive_cols:
        x = pd.to_numeric(d[col], errors="coerce").to_numpy(float)
        ok = np.isfinite(x)&(x>0)
        add_check(checks,"numeric",f"positive_{col}","FAIL",bool(ok.all()),int((~ok).sum()),0)
    add_check(checks,"correctness","all_checksums","FAIL",bool(d.checksum_bool.all()),int((~d.checksum_bool).sum()),0)

    sid = d.problem_size.astype(int).to_numpy()
    batches = d.batches.to_numpy(float)
    expected_flops_op = np.array([flops_per_op(x) for x in sid], float)
    expected_bytes = np.array([logical_bytes_per_op(x) for x in sid], float)
    rules = {
        "flops_total": close_array(d.flops_total, expected_flops_op*batches, 1e-10, 1e-2),
        "logical_bytes_per_op": close_array(d.logical_bytes_per_op, expected_bytes, 1e-10, 1e-2),
        "energy_per_op_j": close_array(d.energy_per_op_j, d.device_energy_j/batches, 5e-6, 5e-10),
        "energy_per_second_j": close_array(d.energy_per_second_j, d.device_energy_j/d.wall_time_s, 5e-6, 1e-6),
        "energy_per_flop_j": close_array(d.energy_per_flop_j, d.device_energy_j/d.flops_total, 5e-6, 1e-18),
        "time_per_op_ms_kernel": close_array(d.time_per_op_ms_kernel, 1000*d.kernel_time_s/batches, 2e-5, 1.1e-6),
        "time_per_op_ms_e2e": close_array(d.time_per_op_ms_e2e, 1000*d.e2e_time_s/batches, 2e-5, 1.1e-6),
        "gflops_per_s": close_array(d.gflops_per_s, d.flops_total/d.kernel_time_s/1e9, 2e-3, 1.1e-2),
        "avg_power_w": close_array(d.avg_power_w, d.device_energy_j/d.wall_time_s, 2e-3, 1.1e-1),
    }
    for name, ok in rules.items():
        add_check(checks,"formula",name,"FAIL",bool(ok.all()),int((~ok).sum()),0)
        if (~ok).any():
            bad = d.loc[~ok,["source_file","session_number","sequence_index","problem_size","num_threads","repetition",name]].copy(); bad["formula"] = name; failures.append(bad)

    expected_status = np.where(d.e2e_time_s.to_numpy(float)<0.75,"below",np.where(d.e2e_time_s.to_numpy(float)<=1.25,"in_range","above"))
    status_ok = d.runtime_status.astype(str).to_numpy()==expected_status
    add_check(checks,"runtime","runtime_status_formula","FAIL",bool(status_ok.all()),int((~status_ok).sum()),0)
    add_check(checks,"runtime","no_below_rows_written","FAIL",not bool((d.runtime_status.astype(str)=="below").any()),d.runtime_status.value_counts().to_dict(),"no below rows")
    add_check(checks,"runtime","all_runtime_in_target","WARN",bool((d.runtime_status.astype(str)=="in_range").all()),d.runtime_status.value_counts().to_dict(),"all in_range preferred")

    wall_equal = close_array(d.wall_time_s, d.e2e_time_s, 1e-9, 1e-9)
    add_check(checks,"timing","wall_equals_e2e","FAIL",bool(wall_equal.all()),int((~wall_equal).sum()),0)
    if cfg["kind"] == "CPU":
        kernel_equal = close_array(d.kernel_time_s, d.e2e_time_s, 1e-9, 1e-9)
        add_check(checks,"timing","cpu_kernel_equals_e2e","FAIL",bool(kernel_equal.all()),int((~kernel_equal).sum()),0)
        dram = d.dram_energy_j.to_numpy(float)
        exp_total = d.device_energy_j.to_numpy(float) + np.where(dram>=0,dram,0.0)
        total_ok = close_array(d.total_energy_j, exp_total, 5e-6, 1e-5)
        add_check(checks,"energy","cpu_total_energy_domain_formula","FAIL",bool(total_ok.all()),int((~total_ok).sum()),0)
        sent = (
            (d.pcie_gen.isna() | d.pcie_gen.eq(-1))
            & (d.pcie_width.isna() | d.pcie_width.eq(-1))
            & (d.mem_clock_mhz.isna() | d.mem_clock_mhz.eq(-1))
        )
        add_check(
            checks,
            "sentinels",
            "cpu_gpu_only_fields",
            "FAIL",
            bool(sent.all()),
            int((~sent).sum()),
            "empty or -1",
        )
    else:
        delta = d.kernel_time_s.to_numpy(float)-d.e2e_time_s.to_numpy(float)
        material = delta > np.maximum(5e-4, 5e-3*d.e2e_time_s.to_numpy(float))
        slight = delta > 2e-6
        add_check(checks,"timing","gpu_kernel_not_materially_above_e2e","FAIL",not bool(material.any()),{"rows":int(material.sum()),"max_ms":float(np.maximum(delta,0).max()*1000)},"0 rows; tolerance max(0.5 ms,0.5%)")
        add_check(checks,"timing","gpu_kernel_slightly_above_e2e","WARN",not bool(slight.any()),{"rows":int(slight.sum()),"max_ms":float(np.maximum(delta,0).max()*1000)},"diagnostic only")
        eq = close_array(d.device_energy_j,d.total_energy_j,5e-6,1e-5)
        add_check(checks,"energy","gpu_device_equals_total","FAIL",bool(eq.all()),int((~eq).sum()),0)
        add_check(checks,"sentinels","gpu_dram_and_cpu_counters","FAIL",bool((d.dram_energy_j.eq(-1)&d.cpu_cycles.eq(-1)&d.cpu_instructions.eq(-1)&d.cpu_ipc.eq(-1)&d.cpu_cache_misses.eq(-1)).all()),"checked","all -1")
        telemetry = (d.pcie_gen.gt(0)&d.pcie_width.gt(0)&d.sm_clock_mhz.gt(0)&d.clock_before_mhz.gt(0)&d.clock_after_mhz.gt(0)&d.mem_clock_mhz.gt(0)&d.temp_c.gt(0)&d.temp_before_c.gt(0)&d.temp_after_c.gt(0))
        add_check(checks,"gpu","telemetry_present","WARN",bool(telemetry.all()),int((~telemetry).sum()),0)
    add_check(checks,"provenance","source_present","FAIL",ctx.source_path.is_file(),ctx.source_path,"present")
    add_check(checks,"provenance","runner_present","WARN",ctx.runner_path is not None,ctx.runner_path,"one known CONV2D runner path")
    return pd.DataFrame(checks), (pd.concat(failures, ignore_index=True, sort=False) if failures else pd.DataFrame())


def add_derived(data: pd.DataFrame, ctx: Context) -> pd.DataFrame:
    d = data.copy()
    batches = d.batches.astype(float)
    d["platform"] = ctx.platform
    d["platform_label"] = ctx.config["label"]
    d["device_kind"] = ctx.config["kind"]
    d["energy_domain"] = ctx.config["energy_domain"]
    d["configuration"] = np.where(ctx.config["kind"]=="CPU", d.num_threads.astype(int).astype(str)+"T", "gpu_resident")
    d["runtime_per_op_s"] = d.e2e_time_s/batches
    d["kernel_runtime_per_op_s"] = d.kernel_time_s/batches
    d["primary_energy_per_op_j"] = d.device_energy_j/batches
    d["total_energy_per_op_j"] = d.total_energy_j/batches
    d["dram_energy_per_op_j"] = np.where(d.dram_energy_j>=0,d.dram_energy_j/batches,np.nan)
    d["edp_total_j_s"] = d.total_energy_per_op_j*d.runtime_per_op_s
    d["logical_flops_per_op"] = d.problem_size.astype(int).map(flops_per_op).astype(float)
    d["logical_bytes_expected"] = d.problem_size.astype(int).map(logical_bytes_per_op).astype(float)
    d["throughput_gflops"] = d.logical_flops_per_op/d.runtime_per_op_s/1e9
    d["energy_efficiency_gflop_per_j"] = d.logical_flops_per_op/d.total_energy_per_op_j/1e9
    d["operational_intensity_flop_per_byte"] = d.logical_flops_per_op/d.logical_bytes_expected
    d["temperature_rise_c"] = d.temp_after_c-d.temp_before_c
    d["clock_change_pct"] = 100*(d.clock_after_mhz-d.clock_before_mhz)/d.clock_before_mhz.replace(0,np.nan)
    return d


def robust_cv(values: Iterable[float]) -> float:
    x=np.asarray([float(v) for v in values if np.isfinite(v)],float)
    if len(x)<2: return np.nan
    med=np.median(x)
    if med==0: return np.nan
    return float(100*1.4826*np.median(np.abs(x-med))/abs(med))


def summarize_platform(d: pd.DataFrame) -> tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    metrics=["runtime_per_op_s","total_energy_per_op_j","edp_total_j_s","throughput_gflops","energy_efficiency_gflop_per_j","avg_power_w","temp_c","sm_clock_mhz"]
    group=["platform","platform_label","device_kind","energy_domain","session_number","problem_size","num_threads","configuration"]
    session=d.groupby(group,dropna=False)[metrics].median().reset_index()
    rows=[]
    for keys,f in session.groupby(["platform","platform_label","device_kind","energy_domain","problem_size","num_threads","configuration"],dropna=False):
        row=dict(zip(["platform","platform_label","device_kind","energy_domain","problem_size","num_threads","configuration"],keys))
        row["session_count"]=f.session_number.nunique()
        for m in metrics:
            x=f[m].dropna().to_numpy(float)
            row[m+"_median"]=float(np.median(x)) if len(x) else np.nan
            row[m+"_min"]=float(np.min(x)) if len(x) else np.nan
            row[m+"_max"]=float(np.max(x)) if len(x) else np.nan
            row[m+"_robust_cv_pct"]=robust_cv(x)
        rows.append(row)
    summary=pd.DataFrame(rows)
    stability=summary[["platform","problem_size","configuration","num_threads","session_count",
                       "runtime_per_op_s_robust_cv_pct","total_energy_per_op_j_robust_cv_pct","edp_total_j_s_robust_cv_pct"]].copy()
    leader_rows=[]
    for sid,f in summary.groupby("problem_size"):
        for objective,col,direction in [
            ("runtime","runtime_per_op_s_median","min"),("energy","total_energy_per_op_j_median","min"),
            ("edp","edp_total_j_s_median","min"),("throughput","throughput_gflops_median","max"),
            ("energy_efficiency","energy_efficiency_gflop_per_j_median","max")]:
            best=f[col].min() if direction=="min" else f[col].max()
            if direction=="min": tie=f[col] <= best*(1+PRACTICAL_TOLERANCE)
            else: tie=f[col] >= best*(1-PRACTICAL_TOLERANCE)
            exact=f[col].eq(best)
            for _,r in f[tie].iterrows():
                leader_rows.append({"problem_size":sid,"objective":objective,"configuration":r.configuration,"num_threads":r.num_threads,"value":r[col],"exact_winner":bool(exact.loc[r.name]),"within_2pct":True})
    leaders=pd.DataFrame(leader_rows)
    pareto_rows=[]
    for sid,f in summary.groupby("problem_size"):
        for idx,r in f.iterrows():
            runtime=r.runtime_per_op_s_median; energy=r.total_energy_per_op_j_median
            other=f.drop(index=idx)
            dominated=((other.runtime_per_op_s_median<=runtime)&(other.total_energy_per_op_j_median<=energy)&((other.runtime_per_op_s_median<runtime)|(other.total_energy_per_op_j_median<energy))).any()
            practical=((other.runtime_per_op_s_median<=runtime*(1-PRACTICAL_TOLERANCE))&(other.total_energy_per_op_j_median<=energy*(1-PRACTICAL_TOLERANCE))).any()
            pareto_rows.append({"problem_size":sid,"configuration":r.configuration,"num_threads":r.num_threads,"runtime_per_op_s":runtime,"total_energy_per_op_j":energy,"strict_pareto":not bool(dominated),"practical_pareto_2pct":not bool(practical)})
    pareto=pd.DataFrame(pareto_rows)
    return session,summary,leaders,pareto.merge(stability,on=["problem_size","configuration","num_threads"],how="left")


def write_manifest(ctx: Context, campaign: Campaign) -> None:
    rows=[]
    for session,path in zip(campaign.sessions,campaign.files):
        rows.append({"platform":ctx.platform,"campaign":campaign.stamp,"session":session,"path":str(path),"sha256":sha256_file(path),"bytes":path.stat().st_size})
    if ctx.source_path.is_file(): rows.append({"platform":ctx.platform,"campaign":campaign.stamp,"session":"source","path":str(ctx.source_path),"sha256":sha256_file(ctx.source_path),"bytes":ctx.source_path.stat().st_size})
    if ctx.runner_path and ctx.runner_path.is_file(): rows.append({"platform":ctx.platform,"campaign":campaign.stamp,"session":"runner","path":str(ctx.runner_path),"sha256":sha256_file(ctx.runner_path),"bytes":ctx.runner_path.stat().st_size})
    pd.DataFrame(rows).to_csv(ctx.result_dir/"campaign_manifest.csv",index=False)
    (ctx.result_dir/"campaign_metadata.json").write_text(json.dumps({"platform":ctx.platform,"campaign":campaign.stamp,"sessions":campaign.sessions,"raw_rows":len(campaign.data),"validation_schema":SCHEMA_VERSION},indent=2),encoding="utf-8")


def markdown_table(df: pd.DataFrame, max_rows: int=200) -> str:
    return "_None._" if df.empty else df.head(max_rows).to_markdown(index=False)


def plot_platform(summary: pd.DataFrame, ctx: Context) -> None:
    for metric,ylabel,name in [
        ("runtime_per_op_s_median","Runtime per convolution [s]","runtime_by_shape.png"),
        ("total_energy_per_op_j_median","Energy per convolution [J]","energy_by_shape.png"),
        ("edp_total_j_s_median","EDP [J s]","edp_by_shape.png"),
        ("throughput_gflops_median","Logical throughput [GFLOP/s]","throughput_by_shape.png"),
    ]:
        fig,ax=plt.subplots(figsize=(9,5.5))
        for cfg,f in summary.groupby("configuration"):
            f=f.sort_values("problem_size")
            ax.plot(f.problem_size,f[metric],marker="o",label=str(cfg))
        ax.set_xlabel("CONV2D shape ID"); ax.set_ylabel(ylabel); ax.set_xticks(list(SHAPES)); ax.grid(True,alpha=.25)
        if summary.configuration.nunique()>1: ax.legend(title="Configuration",ncol=2)
        fig.tight_layout(); fig.savefig(ctx.figure_dir/name,dpi=180); plt.close(fig)

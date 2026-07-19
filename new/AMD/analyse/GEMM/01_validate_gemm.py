#!/usr/bin/env python3
"""Validate one CPU GEMM campaign for completeness, semantics, and plausibility."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from gemm_common import (
    EXPECTED_COLUMNS, EXPECTED_REPETITIONS, EXPECTED_SESSIONS, SIZES,
    TARGET_HIGH_S, TARGET_LOW_S, THREADS, add_derived, campaign_summary, context,
    load_campaign, markdown_table, parse_args, relative_close, write_text,
)


def add_check(checks: list[dict], category: str, name: str, severity: str,
              passed: bool, observed: str, expected: str) -> None:
    checks.append({
        "category": category,
        "check": name,
        "severity": severity,
        "status": "PASS" if passed else severity,
        "observed": observed,
        "expected": expected,
    })


def main() -> None:
    args = parse_args("Validate the local CPU GEMM campaign")
    _, platform, run_dir, result_dir = context(__file__)
    campaign = load_campaign(run_dir, platform, args.campaign)
    raw_df = campaign.data
    df = add_derived(raw_df)
    checks: list[dict] = []

    expected_threads = THREADS[platform]
    expected_rows = len(SIZES) * len(expected_threads) * EXPECTED_REPETITIONS
    expected_sessions = list(range(1, EXPECTED_SESSIONS + 1))

    add_check(checks, "campaign", "session_files", "FAIL",
              campaign.sessions == expected_sessions,
              str(campaign.sessions), str(expected_sessions))

    headers = {path.name: list(pd.read_csv(path, nrows=0).columns) for path in campaign.files}
    audit_metadata = {"source_file", "session_number", "checksum_bool"}
    missing = sorted(set(EXPECTED_COLUMNS) - set(raw_df.columns))
    extras = sorted(set(raw_df.columns) - set(EXPECTED_COLUMNS) - audit_metadata)
    add_check(checks, "schema", "required_columns", "FAIL", not missing,
              f"missing={missing}", "all 45 cpu-gpu-v2 columns")
    add_check(checks, "schema", "column_order", "WARN",
              all(cols == EXPECTED_COLUMNS for cols in headers.values()),
              "identical" if len({tuple(v) for v in headers.values()}) == 1 else "different headers",
              "exact 45-column order")
    add_check(checks, "schema", "unexpected_columns", "WARN", not extras,
              str(extras), "none")
    add_check(checks, "schema", "schema_version", "FAIL",
              set(df.get("schema_version", [])) == {"cpu-gpu-v2"},
              str(sorted(set(df.get("schema_version", [])))), "cpu-gpu-v2")

    file_rows = df.groupby("source_file").size()
    add_check(checks, "coverage", "rows_per_session", "FAIL",
              bool((file_rows == expected_rows).all()),
              str(file_rows.to_dict()), f"{expected_rows} each")
    add_check(checks, "coverage", "problem_sizes", "FAIL",
              sorted(df["problem_size"].dropna().astype(int).unique()) == SIZES,
              str(sorted(df["problem_size"].dropna().astype(int).unique())), str(SIZES))
    add_check(checks, "coverage", "thread_counts", "FAIL",
              sorted(df["num_threads"].dropna().astype(int).unique()) == expected_threads,
              str(sorted(df["num_threads"].dropna().astype(int).unique())), str(expected_threads))

    key = ["session_number", "problem_size", "num_threads", "repetition"]
    duplicates = int(df.duplicated(key).sum())
    add_check(checks, "coverage", "duplicate_configuration_repetitions", "FAIL",
              duplicates == 0, str(duplicates), "0")
    counts = df.groupby(["session_number", "problem_size", "num_threads"]).size()
    add_check(checks, "coverage", "repetitions_per_configuration", "FAIL",
              bool((counts == EXPECTED_REPETITIONS).all()),
              f"min={counts.min()}, max={counts.max()}", str(EXPECTED_REPETITIONS))
    rep_sets = df.groupby(["session_number", "problem_size", "num_threads"])["repetition"].apply(
        lambda x: tuple(sorted(x.dropna().astype(int)))
    )
    expected_rep_set = tuple(range(1, EXPECTED_REPETITIONS + 1))
    add_check(checks, "coverage", "repetition_ids", "FAIL",
              bool((rep_sets == expected_rep_set).all()),
              f"bad_groups={int((rep_sets != expected_rep_set).sum())}", str(expected_rep_set))

    add_check(checks, "semantics", "workload", "FAIL",
              set(df["workload"].astype(str)) == {"GEMM"},
              str(sorted(set(df["workload"].astype(str)))), "GEMM")
    add_check(checks, "semantics", "implementation", "FAIL",
              set(df["implementation"].astype(str)) == {"openblas_sgemm"},
              str(sorted(set(df["implementation"].astype(str)))), "openblas_sgemm")
    add_check(checks, "semantics", "execution_mode", "FAIL",
              set(df["execution_mode"].astype(str)) == {"cpu_native"},
              str(sorted(set(df["execution_mode"].astype(str)))), "cpu_native")
    add_check(checks, "correctness", "checksum", "FAIL",
              bool(df["checksum_bool"].all()),
              f"failed={int((~df['checksum_bool']).sum())}", "0 failures")

    positive_columns = [
        "batches", "e2e_time_s", "kernel_time_s", "wall_time_s",
        "device_energy_j", "total_energy_j", "energy_per_op_j",
        "time_per_op_ms_e2e", "flops_total", "gflops_per_s", "avg_power_w",
    ]
    bad_numeric = pd.Series(False, index=df.index)
    for col in positive_columns:
        values = pd.to_numeric(df[col], errors="coerce")
        bad_numeric |= ~np.isfinite(values) | (values <= 0)
    add_check(checks, "correctness", "finite_positive_measurements", "FAIL",
              not bool(bad_numeric.any()), f"bad_rows={int(bad_numeric.sum())}", "0")

    dram_nonnegative = df["dram_energy_j"].clip(lower=0)
    expected_total = df["device_energy_j"] + dram_nonnegative
    # CSV serialization matters for formula rechecks:
    # - e2e_time_s and time_per_op_ms_e2e are stored with fixed precision(6).
    # - flops_total is stored with scientific precision(6), i.e. about 7 significant digits.
    # Recomputing a per-op value from an already rounded total time therefore needs an
    # absolute error budget derived from both serialized fields.
    expected_time_per_op_ms = 1000.0 * df["e2e_time_s"] / df["batches"]
    time_serialization_atol_ms = 0.5e-6 + (1000.0 * 0.5e-6 / df["batches"])
    time_per_op_ok = (
        np.isfinite(df["time_per_op_ms_e2e"]) &
        np.isfinite(expected_time_per_op_ms) &
        ((df["time_per_op_ms_e2e"] - expected_time_per_op_ms).abs()
         <= time_serialization_atol_ms + 1e-12)
    )

    expected_flops = (
        2.0 *
        df["problem_size"].astype(np.float64) ** 3 *
        df["batches"].astype(np.float64)
    )

    formulas = {
        "cpu_time_fields_equal": relative_close(df["e2e_time_s"], df["kernel_time_s"], 1e-9) &
                                 relative_close(df["e2e_time_s"], df["wall_time_s"], 1e-9),
        "total_energy_device_plus_dram": relative_close(df["total_energy_j"], expected_total, 2e-5, 2e-6),
        "energy_per_op": relative_close(df["energy_per_op_j"], df["total_energy_j"] / df["batches"], 2e-5),
        "energy_per_second": relative_close(df["energy_per_second_j"], df["total_energy_j"] / df["e2e_time_s"], 2e-5),
        "time_per_op": time_per_op_ok,
        "flops_total": relative_close(df["flops_total"], expected_flops, 1.0e-6, 1e-3),
        "gflops_per_s": relative_close(df["gflops_per_s"], df["flops_total"] / df["e2e_time_s"] / 1e9, 2e-3),
        "logical_bytes": relative_close(df["logical_bytes_per_op"], 12.0 * df["problem_size"] ** 2, 2e-7),
        "avg_power": relative_close(df["avg_power_w"], df["total_energy_j"] / df["e2e_time_s"], 2e-4),
    }
    formula_failure_parts = []
    formula_detail_columns = [
        "source_file", "session_number", "sequence_index", "repetition",
        "problem_size", "num_threads", "batches", "e2e_time_s",
        "time_per_op_ms_e2e", "flops_total",
    ]
    for name, mask in formulas.items():
        failed = int((~mask).sum())
        add_check(checks, "formula", name, "FAIL", failed == 0, f"failed_rows={failed}", "0")
        if failed:
            detail = df.loc[~mask, formula_detail_columns].copy()
            detail.insert(0, "failed_formula", name)
            formula_failure_parts.append(detail)

    if formula_failure_parts:
        pd.concat(formula_failure_parts, ignore_index=True).to_csv(
            result_dir / "formula_failures.csv", index=False
        )
    else:
        pd.DataFrame(columns=["failed_formula", *formula_detail_columns]).to_csv(
            result_dir / "formula_failures.csv", index=False
        )

    expected_status = np.select(
        [df["e2e_time_s"] < TARGET_LOW_S, df["e2e_time_s"] > TARGET_HIGH_S],
        ["below", "above"], default="in_range",
    )
    status_bad = int((df["runtime_status"].astype(str).to_numpy() != expected_status).sum())
    add_check(checks, "formula", "runtime_status", "FAIL", status_bad == 0,
              f"failed_rows={status_bad}", "0")

    in_range_share = float(df["e2e_time_s"].between(TARGET_LOW_S, TARGET_HIGH_S).mean())
    add_check(checks, "plausibility", "target_runtime_share", "WARN",
              in_range_share >= 0.90, f"{100*in_range_share:.2f}%", ">=90%")
    max_temp = float(df["temp_c"].max())
    add_check(checks, "plausibility", "maximum_temperature", "WARN",
              max_temp < 95.0, f"{max_temp:.1f} C", "<95 C")
    power_min, power_max = float(df["package_avg_power_w"].min()), float(df["package_avg_power_w"].max())
    add_check(checks, "plausibility", "package_power_range", "WARN",
              power_min > 1.0 and power_max < 500.0,
              f"{power_min:.2f}..{power_max:.2f} W", "1..500 W broad sanity range")

    sequence_bad = 0
    session_id_bad = 0
    for session, group in df.groupby("session_number"):
        expected_seq = list(range(1, len(group) + 1))
        sequence_bad += int(sorted(group["sequence_index"].astype(int)) != expected_seq)
        wanted = f"{campaign.stamp}_session{session}"
        session_id_bad += int(set(group["session_id"].astype(str)) != {wanted})
    add_check(checks, "provenance", "sequence_indices", "FAIL", sequence_bad == 0,
              f"bad_sessions={sequence_bad}", "0")
    add_check(checks, "provenance", "session_ids_match_filenames", "FAIL", session_id_bad == 0,
              f"bad_sessions={session_id_bad}", "0")

    # Threading sanity: exclude small GEMMs, where thread-launch overhead can legitimately dominate.
    summary = campaign_summary(df)
    scaling_rows: list[dict] = []
    for n, group in summary.groupby("problem_size", sort=True):
        base = group[group["num_threads"] == 1]
        if base.empty:
            continue
        base_runtime = float(base.iloc[0]["runtime_per_op_s_median"])
        for _, row in group.sort_values("num_threads").iterrows():
            runtime = float(row["runtime_per_op_s_median"])
            speedup = base_runtime / runtime
            slowdown = runtime / base_runtime
            if int(n) >= 2048 and int(row["num_threads"]) > 1 and slowdown >= 5.0:
                flag = "catastrophic_slowdown"
            elif int(n) >= 4096 and 1 < int(row["num_threads"]) <= 20 and slowdown > 1.25:
                flag = "large_gemm_regression"
            else:
                flag = "ok"
            scaling_rows.append({
                "problem_size": int(n),
                "num_threads": int(row["num_threads"]),
                "runtime_per_op_s_median": runtime,
                "runtime_1thread_s": base_runtime,
                "speedup_vs_1thread": speedup,
                "slowdown_vs_1thread": slowdown,
                "flag": flag,
            })
    scaling = pd.DataFrame(scaling_rows)
    scaling.to_csv(result_dir / "threading_scaling_sanity.csv", index=False)

    catastrophic = scaling[scaling["flag"] == "catastrophic_slowdown"]
    add_check(checks, "threading", "no_catastrophic_multithread_slowdown", "FAIL",
              catastrophic.empty,
              f"flagged={len(catastrophic)}",
              "no N>=2048 multithread runtime >=5x the 1-thread runtime")

    large_best = (
        scaling[(scaling["problem_size"] >= 4096) & (scaling["num_threads"] > 1)]
        .groupby("problem_size", as_index=False)["speedup_vs_1thread"].max()
    )
    weak_large = large_best[large_best["speedup_vs_1thread"] < 1.25]
    add_check(checks, "threading", "large_gemm_has_useful_parallel_speedup", "FAIL",
              weak_large.empty,
              "none" if weak_large.empty else weak_large.to_dict("records").__str__(),
              "for every N>=4096, at least one multithread configuration reaches >=1.25x")

    regressions = scaling[scaling["flag"] == "large_gemm_regression"]
    add_check(checks, "threading", "large_gemm_moderate_thread_regressions", "WARN",
              regressions.empty,
              f"flagged={len(regressions)}",
              "no 2..20-thread configuration at N>=4096 is >25% slower than 1 thread")

    checks_df = pd.DataFrame(checks)
    checks_df.to_csv(result_dir / "validation_checks.csv", index=False)

    manifest = pd.DataFrame({
        "platform": platform,
        "campaign": campaign.stamp,
        "session": campaign.sessions,
        "file": [p.name for p in campaign.files],
        "rows": [int((df["source_file"] == p.name).sum()) for p in campaign.files],
        "bytes": [p.stat().st_size for p in campaign.files],
    })
    manifest.to_csv(result_dir / "campaign_manifest.csv", index=False)

    failed = checks_df[checks_df["status"] == "FAIL"]
    warnings = checks_df[checks_df["status"] == "WARN"]
    overall = "FAIL" if not failed.empty else ("PASS WITH WARNINGS" if not warnings.empty else "PASS")
    report = f"""# {platform} GEMM validation report

- Campaign: `{campaign.stamp}`
- Files: {len(campaign.files)}
- Rows: {len(df)}
- Expected rows/session: {expected_rows}
- Overall verdict: **{overall}**

## Failed checks

{markdown_table(failed)}

## Warnings

{markdown_table(warnings)}

## All checks

{markdown_table(checks_df, max_rows=100)}

## Threading sanity table

The full configuration-level table is written to `threading_scaling_sanity.csv`.
Small GEMMs below N=2048 are intentionally excluded from hard scaling checks because
thread-launch overhead can legitimately make multithreading slower there.

## Metric semantics

Cross-platform energy analysis must use `device_energy_j / batches`, emitted by the
analysis as `package_energy_per_op_j`. The raw `energy_per_op_j` is based on
`total_energy_j` and may include Intel DRAM RAPL, while AMD may not expose a
separate DRAM domain. Therefore raw `energy_per_op_j` is retained for audit only,
not used as the primary Intel-vs-AMD energy metric.
"""
    write_text(result_dir / "validation_report.md", report)
    print(f"[{platform}] validation: {overall}")
    print(result_dir / "validation_report.md")
    if not failed.empty:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

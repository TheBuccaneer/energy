#!/usr/bin/env python3
"""Validate one CPU STRIDED_GEMM campaign and its source provenance."""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from strided_gemm_common import (
    ALTERNATE_COLUMNS, EXPECTED_COLUMNS, EXPECTED_REPETITIONS, EXPECTED_SESSIONS,
    MAX_BATCHES, SIZES, TARGET_HIGH_S, TARGET_LOW_S, THREADS, add_derived,
    campaign_summary, context, load_campaign, markdown_table, parse_args,
    relative_close, write_text,
)


def add_check(checks: list[dict], category: str, name: str, severity: str,
              passed: bool, observed, expected) -> None:
    checks.append({
        "category": category, "check": name, "severity": severity,
        "status": "PASS" if passed else severity,
        "observed": str(observed), "expected": str(expected),
    })


def source_and_runner(root: Path, platform: str) -> tuple[Path, Path]:
    slug = platform.lower() if platform == "INTEL" else "amd"
    source = root / platform / "scripts" / "STRIDED_GEMM" / f"main_gemm_strided_{slug}.cpp"
    runner = root / platform / "scripts" / f"02_run_CPU_{'Intel' if platform == 'INTEL' else 'AMD'}_STRIDED_GEMM_only.sh"
    return source, runner


def check_provenance(checks: list[dict], root: Path, platform: str, result_dir: Path) -> None:
    source, runner = source_and_runner(root, platform)
    add_check(checks, "provenance", "source_present", "FAIL", source.is_file(), source, "present")
    if source.is_file():
        text = source.read_text(encoding="utf-8", errors="replace")
        exact_tokens = {
            "workload_token": '"STRIDED_GEMM"',
            "implementation_token": '"openblas_sgemm_ld2n"',
            "row_major_sgemm": "cblas_sgemm(CblasRowMajor",
            "thread_control": "openblas_set_num_threads",
            "thread_verification": "openblas_get_num_threads",
            "padded_initialization": "if (col < n)",
            "padding_zero_a": "a[index] = 0.0f",
            "padding_zero_b": "b[index] = 0.0f",
            "checksum_uses_ld": "row) * static_cast<size_t>(ld) + col",
            "logical_bytes_formula": "3.0 * n * static_cast<double>(n) * sizeof(float)",
            "flops_formula": "2.0 * n * static_cast<double>(n) * n",
        }
        for name, token in exact_tokens.items():
            add_check(checks, "provenance", name, "FAIL", token in text,
                      "present" if token in text else "missing", token)
        regexes = {
            "ld_equals_2n": r"const\s+int\s+ld\s*=\s*2\s*\*\s*n\s*;",
            "allocation_is_n_times_ld": r"count\s*=\s*static_cast<size_t>\(n\)\s*\*\s*static_cast<size_t>\(ld\)",
            "all_three_leading_dimensions_use_ld": r"a,\s*ld,\s*b,\s*ld,\s*0\.0f,\s*c,\s*ld",
            "problem_spec_records_ld": r'"N="\s*\+.*";ld="',
        }
        for name, pattern in regexes.items():
            passed = bool(re.search(pattern, text, re.S))
            add_check(checks, "provenance", name, "FAIL", passed,
                      "matched" if passed else "not matched", pattern)
        (result_dir / "audited_source_path.txt").write_text(str(source) + "\n", encoding="utf-8")

    add_check(checks, "provenance", "runner_present", "FAIL", runner.is_file(), runner, "present")
    if runner.is_file():
        text = runner.read_text(encoding="utf-8", errors="replace")
        add_check(checks, "provenance", "runner_default_sessions", "FAIL",
                  bool(re.search(r"SESSIONS=\$\{SESSIONS:-5\}", text)), "checked", "default 5")
        add_check(checks, "provenance", "runner_default_repetitions", "FAIL",
                  bool(re.search(r"REPS=\$\{REPS:-10\}", text)), "checked", "default 10")
        pause_tokens = [t for t in ["sleep 300", "SESSION_PAUSE", "Cooling for"] if t in text]
        add_check(checks, "provenance", "no_between_session_pause", "FAIL",
                  not pause_tokens, pause_tokens or "none", "none")
        (result_dir / "audited_runner_path.txt").write_text(str(runner) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args("Validate a CPU STRIDED_GEMM campaign")
    root, platform, run_dir, result_dir = context(__file__)
    campaign = load_campaign(run_dir, platform, args.campaign)
    raw = campaign.data
    df = add_derived(raw)
    checks: list[dict] = []

    expected_threads = THREADS[platform]
    expected_rows = len(SIZES) * len(expected_threads) * EXPECTED_REPETITIONS
    expected_sessions = list(range(1, EXPECTED_SESSIONS + 1))

    add_check(checks, "campaign", "session_files", "FAIL",
              campaign.sessions == expected_sessions, campaign.sessions, expected_sessions)

    headers = {p.name: list(pd.read_csv(p, nrows=0).columns) for p in campaign.files}
    audit_meta = {"source_file", "session_number", "checksum_bool"}
    missing = sorted(set(EXPECTED_COLUMNS) - set(raw.columns))
    extras = sorted(set(raw.columns) - set(EXPECTED_COLUMNS) - audit_meta)
    add_check(checks, "schema", "required_columns", "FAIL", not missing, missing or "none", "all 45")
    valid_orders = all(cols in (EXPECTED_COLUMNS, ALTERNATE_COLUMNS) for cols in headers.values())
    add_check(checks, "schema", "recognized_column_order", "WARN", valid_orders,
              "recognized" if valid_orders else headers, "canonical or known alternate")
    add_check(checks, "schema", "consistent_headers", "FAIL",
              len({tuple(v) for v in headers.values()}) == 1,
              len({tuple(v) for v in headers.values()}), 1)
    add_check(checks, "schema", "unexpected_columns", "WARN", not extras, extras or "none", "none")
    add_check(checks, "schema", "schema_version", "FAIL",
              set(raw["schema_version"].astype(str)) == {"cpu-gpu-v2"},
              sorted(set(raw["schema_version"].astype(str))), "cpu-gpu-v2")

    file_rows = raw.groupby("source_file").size()
    add_check(checks, "coverage", "rows_per_session", "FAIL",
              len(file_rows) == 5 and bool((file_rows == expected_rows).all()),
              file_rows.to_dict(), f"{expected_rows} each")
    add_check(checks, "coverage", "problem_sizes", "FAIL",
              sorted(raw["problem_size"].dropna().astype(int).unique()) == SIZES,
              sorted(raw["problem_size"].dropna().astype(int).unique()), SIZES)
    add_check(checks, "coverage", "thread_counts", "FAIL",
              sorted(raw["num_threads"].dropna().astype(int).unique()) == expected_threads,
              sorted(raw["num_threads"].dropna().astype(int).unique()), expected_threads)

    key = ["session_number", "problem_size", "num_threads", "repetition"]
    duplicates = int(raw.duplicated(key).sum())
    add_check(checks, "coverage", "duplicate_configuration_repetitions", "FAIL",
              duplicates == 0, duplicates, 0)
    counts = raw.groupby(["session_number", "problem_size", "num_threads"]).size()
    add_check(checks, "coverage", "repetitions_per_configuration", "FAIL",
              len(counts) == EXPECTED_SESSIONS * len(SIZES) * len(expected_threads) and
              bool((counts == EXPECTED_REPETITIONS).all()),
              f"groups={len(counts)}, min={counts.min()}, max={counts.max()}",
              f"{EXPECTED_SESSIONS * len(SIZES) * len(expected_threads)} groups; {EXPECTED_REPETITIONS} each")
    rep_sets = raw.groupby(["session_number", "problem_size", "num_threads"])["repetition"].apply(
        lambda x: tuple(sorted(x.dropna().astype(int)))
    )
    wanted_reps = tuple(range(1, EXPECTED_REPETITIONS + 1))
    add_check(checks, "coverage", "repetition_ids", "FAIL",
              bool((rep_sets == wanted_reps).all()), int((rep_sets != wanted_reps).sum()), 0)

    expected_semantics = {
        "workload": {"STRIDED_GEMM"},
        "implementation": {"openblas_sgemm_ld2n"},
        "execution_mode": {"cpu_native"},
    }
    for col, expected in expected_semantics.items():
        observed = set(raw[col].astype(str))
        add_check(checks, "semantics", col, "FAIL", observed == expected,
                  sorted(observed), sorted(expected))
    expected_spec = raw["problem_size"].map(lambda n: f"N={int(n)};ld={2*int(n)}")
    add_check(checks, "semantics", "problem_spec_ld2n", "FAIL",
              bool((raw["problem_spec"].astype(str) == expected_spec).all()),
              f"failed={int((raw['problem_spec'].astype(str) != expected_spec).sum())}", 0)
    add_check(checks, "correctness", "checksum", "FAIL", bool(raw["checksum_bool"].all()),
              int((~raw["checksum_bool"]).sum()), 0)

    positive = [
        "batches", "e2e_time_s", "kernel_time_s", "wall_time_s", "device_energy_j",
        "total_energy_j", "energy_per_op_j", "time_per_op_ms_e2e", "flops_total",
        "gflops_per_s", "avg_power_w",
    ]
    bad = pd.Series(False, index=raw.index)
    for col in positive:
        values = pd.to_numeric(raw[col], errors="coerce")
        bad |= ~np.isfinite(values) | (values <= 0)
    add_check(checks, "correctness", "finite_positive_measurements", "FAIL",
              not bool(bad.any()), int(bad.sum()), 0)

    dram_present = raw["dram_energy_j"] >= 0
    dram_mode = (
        "all_rows" if bool(dram_present.all()) else
        "no_rows" if not bool(dram_present.any()) else
        "mixed"
    )
    add_check(checks, "energy_domain", "dram_rapl_availability", "WARN",
              dram_mode != "mixed",
              f"mode={dram_mode}; available={int(dram_present.sum())}/{len(raw)}",
              "all rows or no rows; DRAM is optional and never required for package-only primary analysis")

    dram_nonnegative = raw["dram_energy_j"].clip(lower=0)
    expected_total = raw["device_energy_j"] + dram_nonnegative
    expected_flops = 2.0 * raw["problem_size"].astype(float) ** 3 * raw["batches"].astype(float)
    expected_logical = 12.0 * raw["problem_size"].astype(float) ** 2
    expected_e2e_ms = 1000.0 * raw["e2e_time_s"] / raw["batches"]
    expected_kernel_ms = 1000.0 * raw["kernel_time_s"] / raw["batches"]
    time_atol = 0.5e-6 + 1000.0 * 0.5e-6 / raw["batches"]

    formulas = {
        "cpu_time_e2e_equals_kernel": relative_close(raw["e2e_time_s"], raw["kernel_time_s"], 1e-9, 1e-9),
        "cpu_time_e2e_equals_wall": relative_close(raw["e2e_time_s"], raw["wall_time_s"], 1e-9, 1e-9),
        "total_energy_package_plus_dram": relative_close(raw["total_energy_j"], expected_total, 2e-5, 2e-6),
        "energy_per_op_uses_total": relative_close(raw["energy_per_op_j"], raw["total_energy_j"] / raw["batches"], 2e-5, 1e-9),
        "energy_per_second_uses_total": relative_close(raw["energy_per_second_j"], raw["total_energy_j"] / raw["e2e_time_s"], 2e-5, 1e-7),
        "energy_per_flop_uses_total": relative_close(raw["energy_per_flop_j"], raw["total_energy_j"] / expected_flops, 3e-5, 1e-20),
        "time_per_op_e2e": (raw["time_per_op_ms_e2e"] - expected_e2e_ms).abs() <= time_atol,
        "time_per_op_kernel": (raw["time_per_op_ms_kernel"] - expected_kernel_ms).abs() <= time_atol,
        "flops_total": relative_close(raw["flops_total"], expected_flops, 1e-6, 1e-2),
        "gflops_per_s": relative_close(raw["gflops_per_s"], raw["flops_total"] / raw["e2e_time_s"] / 1e9, 2e-3, 1e-4),
        "logical_bytes_excludes_padding": relative_close(raw["logical_bytes_per_op"], expected_logical, 2e-7, 1e-3),
        "avg_power_uses_total": relative_close(raw["avg_power_w"], raw["total_energy_j"] / raw["e2e_time_s"], 2e-4, 1e-4),
    }
    failure_parts = []
    detail_cols = [
        "source_file", "session_number", "sequence_index", "repetition", "problem_size",
        "num_threads", "batches", "e2e_time_s", "kernel_time_s", "total_energy_j",
        "device_energy_j", "dram_energy_j", "time_per_op_ms_e2e", "flops_total",
    ]
    for name, mask in formulas.items():
        mask = pd.Series(mask, index=raw.index).fillna(False)
        failed = int((~mask).sum())
        add_check(checks, "formula", name, "FAIL", failed == 0, failed, 0)
        if failed:
            part = raw.loc[~mask, detail_cols].copy()
            part.insert(0, "failed_formula", name)
            failure_parts.append(part)
    failures = pd.concat(failure_parts, ignore_index=True) if failure_parts else pd.DataFrame(columns=["failed_formula", *detail_cols])
    failures.to_csv(result_dir / "formula_failures.csv", index=False)

    expected_status = np.select(
        [raw["e2e_time_s"] < TARGET_LOW_S, raw["e2e_time_s"] > TARGET_HIGH_S],
        ["below", "above"], default="in_range",
    )
    status_bad = int((raw["runtime_status"].astype(str).to_numpy() != expected_status).sum())
    add_check(checks, "formula", "runtime_status", "FAIL", status_bad == 0, status_bad, 0)

    batches_constant = raw.groupby(["session_number", "problem_size", "num_threads"])["batches"].nunique()
    add_check(checks, "calibration", "batches_constant_within_configuration_session", "FAIL",
              bool((batches_constant == 1).all()), int((batches_constant != 1).sum()), 0)
    unavoidable = ((raw["batches"] == 1) & (raw["e2e_time_s"] > TARGET_HIGH_S)) | \
                  ((raw["batches"] == MAX_BATCHES) & (raw["e2e_time_s"] < TARGET_LOW_S))
    all_share = float(raw["e2e_time_s"].between(TARGET_LOW_S, TARGET_HIGH_S).mean())
    actionable = raw.loc[~unavoidable]
    actionable_share = float(actionable["e2e_time_s"].between(TARGET_LOW_S, TARGET_HIGH_S).mean()) if len(actionable) else 1.0
    add_check(checks, "plausibility", "target_runtime_share_all", "WARN",
              all_share >= 0.90, f"{100*all_share:.2f}%", ">=90%")
    add_check(checks, "plausibility", "target_runtime_share_actionable", "WARN",
              actionable_share >= 0.90, f"{100*actionable_share:.2f}%", ">=90% excluding minimum/maximum-batch limits")

    max_temp = float(raw["temp_c"].max())
    temp_threshold = 95.0 if platform == "AMD" else 101.0
    add_check(checks, "plausibility", "maximum_temperature", "WARN",
              max_temp < temp_threshold, f"{max_temp:.1f} C", f"<{temp_threshold:.0f} C")
    total_power = raw["total_energy_j"] / raw["e2e_time_s"]
    add_check(checks, "plausibility", "total_cpu_power_range", "WARN",
              float(total_power.min()) > 1 and float(total_power.max()) < 600,
              f"{total_power.min():.2f}..{total_power.max():.2f} W", "1..600 W")

    sequence_bad = session_id_bad = run_id_bad = 0
    for session, group in raw.groupby("session_number"):
        sequence_bad += int(sorted(group["sequence_index"].astype(int)) != list(range(1, expected_rows + 1)))
        session_ids = set(group["session_id"].astype(str))
        valid_session_id = (
            len(session_ids) == 1
            and all(campaign.stamp in sid and sid.endswith(f"_session{session}") for sid in session_ids)
        )
        session_id_bad += int(not valid_session_id)
        run_id_bad += int(group["run_id_global"].duplicated().any())
    add_check(checks, "provenance", "sequence_indices", "FAIL", sequence_bad == 0, sequence_bad, 0)
    add_check(checks, "provenance", "session_ids_match_files", "FAIL", session_id_bad == 0, session_id_bad, 0)
    add_check(checks, "provenance", "run_ids_unique_within_session", "FAIL", run_id_bad == 0, run_id_bad, 0)

    orders = []
    for session, group in raw.groupby("session_number"):
        first = (group.sort_values("sequence_index")
                 .drop_duplicates(["problem_size", "num_threads"])[["problem_size", "num_threads"]])
        orders.append(tuple(map(tuple, first.to_numpy())))
    add_check(checks, "randomization", "session_configuration_orders_differ", "WARN",
              len(set(orders)) > 1, f"unique_orders={len(set(orders))}", ">1")

    summary = campaign_summary(df)
    scaling_rows = []
    for n, group in summary.groupby("problem_size"):
        base = group[group["num_threads"] == 1]
        if base.empty:
            continue
        t1 = float(base.iloc[0]["runtime_per_op_s_median"])
        for _, row in group.sort_values("num_threads").iterrows():
            runtime = float(row["runtime_per_op_s_median"])
            slowdown = runtime / t1
            if int(n) >= 2048 and int(row["num_threads"]) > 1 and slowdown >= 5:
                flag = "catastrophic_slowdown"
            elif int(n) >= 4096 and 1 < int(row["num_threads"]) <= 20 and slowdown > 1.25:
                flag = "large_problem_regression"
            else:
                flag = "ok"
            scaling_rows.append({
                "problem_size": int(n), "num_threads": int(row["num_threads"]),
                "runtime_per_op_s_median": runtime, "runtime_1thread_s": t1,
                "speedup_vs_1thread": t1 / runtime, "slowdown_vs_1thread": slowdown,
                "flag": flag,
            })
    scaling = pd.DataFrame(scaling_rows)
    scaling.to_csv(result_dir / "threading_scaling_sanity.csv", index=False)
    catastrophic = scaling[scaling["flag"] == "catastrophic_slowdown"]
    add_check(checks, "threading", "no_catastrophic_multithread_slowdown", "FAIL",
              catastrophic.empty, len(catastrophic), 0)
    large_best = (scaling[(scaling["problem_size"] >= 4096) & (scaling["num_threads"] > 1)]
                  .groupby("problem_size", as_index=False)["speedup_vs_1thread"].max())
    weak = large_best[large_best["speedup_vs_1thread"] < 1.25]
    add_check(checks, "threading", "large_problem_has_useful_parallel_speedup", "FAIL",
              weak.empty, weak.to_dict("records") if not weak.empty else "none", ">=1.25x for each N>=4096")
    regressions = scaling[scaling["flag"] == "large_problem_regression"]
    add_check(checks, "threading", "moderate_thread_regressions", "WARN",
              regressions.empty, len(regressions), 0)

    check_provenance(checks, root, platform, result_dir)

    check_df = pd.DataFrame(checks)
    check_df.to_csv(result_dir / "validation_checks.csv", index=False)
    manifest = pd.DataFrame({
        "platform": platform, "workload": "STRIDED_GEMM", "campaign": campaign.stamp,
        "session": campaign.sessions, "file": [p.name for p in campaign.files],
        "rows": [int((raw["source_file"] == p.name).sum()) for p in campaign.files],
        "bytes": [p.stat().st_size for p in campaign.files],
        "dram_available_rows": [int(((raw["source_file"] == p.name) & (raw["dram_energy_j"] >= 0)).sum()) for p in campaign.files],
    })
    manifest.to_csv(result_dir / "campaign_manifest.csv", index=False)

    hard = check_df[(check_df["severity"] == "FAIL") & (check_df["status"] == "FAIL")]
    warns = check_df[(check_df["severity"] == "WARN") & (check_df["status"] == "WARN")]
    verdict = "FAIL" if len(hard) else ("PASS WITH WARNINGS" if len(warns) else "PASS")
    report = f"""# {platform} STRIDED_GEMM validation report

- Campaign: `{campaign.stamp}`
- Files: {len(campaign.files)}
- Rows: {len(raw)}
- Expected rows/session: {expected_rows}
- Workload semantics: square SGEMM with `ld=2N`, logical N×N data and padded rows
- Primary energy domain: CPU package RAPL (`device_energy_j / batches`)
- Optional sensitivity: package + DRAM (`total_energy_j / batches`) where DRAM RAPL exists
- Overall verdict: **{verdict}**

## Failed checks

{markdown_table(hard)}

## Warnings

{markdown_table(warns)}

## All checks

{markdown_table(check_df, 200)}

## Audit interpretation

The source allocates three N×2N matrices but performs the same 2N³ logical FLOPs as dense
GEMM. `logical_bytes_per_op` remains 12N² by design; the analysis separately derives a
24N² allocated footprint. The padding footprint is not itself a claim about bytes physically
transferred by the memory hierarchy.

The primary cross-platform energy metric is CPU package RAPL because AMD DRAM RAPL is unavailable
in this campaign and mixed energy boundaries would invalidate Intel–AMD comparisons. Where DRAM RAPL
exists, package+DRAM remains an explicit within-platform sensitivity analysis.
"""
    write_text(result_dir / "validation_report.md", report)
    print(f"[{platform}] STRIDED_GEMM validation: {verdict}")
    print(result_dir / "validation_report.md")
    if verdict == "FAIL":
        sys.exit(2)


if __name__ == "__main__":
    main()

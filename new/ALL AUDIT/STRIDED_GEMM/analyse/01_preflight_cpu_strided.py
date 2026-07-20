#!/usr/bin/env python3
"""Preflight the AMD and Intel STRIDED_GEMM analyses before combination."""
from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

import pandas as pd

from cpu_strided_common import (
    EXPECTED_REPETITIONS, EXPECTED_SESSIONS, SIZES, THREADS,
    discover_campaigns, load_project_campaign, markdown_table, write_text,
)


def add(checks, category, name, severity, passed, observed, expected):
    checks.append({
        "category": category, "check": name, "severity": severity,
        "status": "PASS" if passed else severity,
        "observed": str(observed), "expected": str(expected),
    })


def normalize_source(text: str) -> str:
    text = text.replace("strided_gemm_amd.csv", "strided_gemm_PLATFORM.csv")
    text = text.replace("strided_gemm_intel.csv", "strided_gemm_PLATFORM.csv")
    return re.sub(r"\s+", " ", text).strip()


def main() -> None:
    script = Path(__file__).resolve()
    root = script.parents[3]
    out = root / "ALL AUDIT" / "STRIDED_GEMM" / "results"
    out.mkdir(parents=True, exist_ok=True)
    checks = []
    campaigns = {}

    for platform in ["AMD", "INTEL"]:
        run_dir = root / platform / "runs" / "STRIDED_GEMM"
        result_dir = root / platform / "results" / "STRIDED_GEMM"
        validation = result_dir / "validation_checks.csv"
        config = result_dir / "configuration_summary.csv"
        sessions = result_dir / "session_configuration_medians.csv"
        manifest = result_dir / "campaign_manifest.csv"
        for name, path in [
            ("validation_checks", validation), ("configuration_summary", config),
            ("session_configuration_medians", sessions), ("campaign_manifest", manifest),
        ]:
            add(checks, platform, f"{name}_present", "FAIL", path.is_file(), path, "present")
        if not all(p.is_file() for p in [validation, config, sessions, manifest]):
            continue

        v = pd.read_csv(validation)
        hard = v[(v["severity"] == "FAIL") & (v["status"] == "FAIL")]
        add(checks, platform, "individual_validation_no_hard_failure", "FAIL", hard.empty,
            len(hard), 0)
        m = pd.read_csv(manifest)
        stamp = str(m["campaign"].iloc[0])
        campaigns[platform] = stamp
        groups = discover_campaigns(run_dir, platform, "STRIDED_GEMM")
        complete = sorted(
            s for s, entries in groups.items()
            if sorted(n for n, _ in entries) == list(range(1, EXPECTED_SESSIONS + 1))
        )
        latest = max(complete) if complete else "none"
        add(checks, platform, "analysis_matches_latest_complete_campaign", "FAIL",
            stamp == latest, stamp, latest)

        c = pd.read_csv(config)
        s = pd.read_csv(sessions)
        expected_configs = len(SIZES) * len(THREADS[platform])
        add(checks, platform, "configuration_summary_rows", "FAIL",
            len(c) == expected_configs, len(c), expected_configs)
        add(checks, platform, "session_median_rows", "FAIL",
            len(s) == expected_configs * EXPECTED_SESSIONS,
            len(s), expected_configs * EXPECTED_SESSIONS)
        add(checks, platform, "five_sessions_per_configuration", "FAIL",
            bool((s.groupby(["problem_size", "num_threads"])["session_number"].nunique() == 5).all()),
            "checked", 5)
        needed = {
            "runtime_per_op_s_median", "total_energy_per_op_j_median",
            "package_energy_per_op_j_median", "edp_total_j_s_median",
            "throughput_gflops_exact_median", "total_efficiency_gflop_per_j_median",
        }
        add(checks, platform, "required_summary_metrics", "FAIL", needed.issubset(c.columns),
            sorted(needed - set(c.columns)) or "none missing", "all present")

        raw = load_project_campaign(root, platform, "STRIDED_GEMM").data
        add(checks, platform, "dram_available_for_primary_metric", "FAIL",
            bool((raw["dram_energy_j"] >= 0).all()),
            f"{int((raw['dram_energy_j'] >= 0).sum())}/{len(raw)}", f"{len(raw)}/{len(raw)}")

        dense_groups = discover_campaigns(root / platform / "runs" / "GEMM", platform, "GEMM")
        dense_complete = sorted(
            st for st, entries in dense_groups.items()
            if sorted(n for n, _ in entries) == list(range(1, EXPECTED_SESSIONS + 1))
        )
        add(checks, platform, "dense_gemm_campaign_for_layout_comparison", "FAIL",
            bool(dense_complete), dense_complete[-1] if dense_complete else "none", "one complete campaign")
        if dense_complete:
            dense = load_project_campaign(root, platform, "GEMM").data
            add(checks, platform, "dense_gemm_dram_available", "FAIL",
                bool((dense["dram_energy_j"] >= 0).all()),
                f"{int((dense['dram_energy_j'] >= 0).sum())}/{len(dense)}", f"{len(dense)}/{len(dense)}")

    sources = {
        "AMD": root / "AMD" / "scripts" / "STRIDED_GEMM" / "main_gemm_strided_amd.cpp",
        "INTEL": root / "INTEL" / "scripts" / "STRIDED_GEMM" / "main_gemm_strided_intel.cpp",
    }
    normalized = {}
    for platform, path in sources.items():
        add(checks, "source_parity", f"{platform.lower()}_source_present", "FAIL", path.is_file(), path, "present")
        if path.is_file():
            text = normalize_source(path.read_text(encoding="utf-8", errors="replace"))
            normalized[platform] = text
            (out / f"{platform.lower()}_normalized_source_sha256.txt").write_text(
                hashlib.sha256(text.encode()).hexdigest() + "\n", encoding="utf-8")
    if len(normalized) == 2:
        add(checks, "source_parity", "amd_intel_source_equivalent", "FAIL",
            normalized["AMD"] == normalized["INTEL"],
            "identical after normalizing default filename" if normalized["AMD"] == normalized["INTEL"] else "different",
            "identical")

    check_df = pd.DataFrame(checks)
    check_df.to_csv(out / "preflight_checks.csv", index=False)
    hard = check_df[(check_df["severity"] == "FAIL") & (check_df["status"] == "FAIL")]
    warns = check_df[(check_df["severity"] == "WARN") & (check_df["status"] == "WARN")]
    verdict = "FAIL" if len(hard) else ("PASS WITH WARNINGS" if len(warns) else "PASS")
    report = f"""# CPU STRIDED_GEMM combined preflight

- AMD campaign: `{campaigns.get('AMD', 'unavailable')}`
- Intel campaign: `{campaigns.get('INTEL', 'unavailable')}`
- Primary energy domain: CPU package + DRAM
- Package-only retained as sensitivity
- Overall verdict: **{verdict}**

## Failed checks

{markdown_table(hard)}

## Warnings

{markdown_table(warns)}

## All checks

{markdown_table(check_df, 200)}

## Scope

This preflight combines only individually validated campaigns. It also requires a complete
dense GEMM campaign for each CPU so that the layout penalty of `ld=2N` can be computed from
raw data using the same package+DRAM definition.
"""
    write_text(out / "preflight_report.md", report)
    print(f"[CPU STRIDED_GEMM] preflight: {verdict}")
    print(out / "preflight_report.md")
    if verdict == "FAIL":
        sys.exit(2)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import re
import sys
import pandas as pd

from all_stream_common import *


def normalized_source(path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")
    text = text.replace("stream_amd.csv", "stream_PLATFORM.csv").replace("stream_intel.csv", "stream_PLATFORM.csv")
    text = text.replace("platform=AMD", "platform=PLATFORM").replace("platform=INTEL", "platform=PLATFORM")
    return re.sub(r"\s+", " ", text).strip()


def main() -> None:
    out = results_dir(__file__)
    checks = []
    manifest_rows = []

    for platform in PLATFORMS:
        required = required_platform_outputs(__file__, platform)
        missing = [str(path) for path in required if not path.is_file()]
        add_check(checks, "inputs", "required_platform_outputs", "FAIL", not missing,
                  missing or "all present", "all required files", platform)
        if missing:
            continue
        validation = pd.read_csv(platform_result_dir(__file__, platform) / "validation_checks.csv")
        hard = validation[(validation.severity == "FAIL") & (validation.status == "FAIL")]
        warnings = validation[(validation.severity == "WARN") & (validation.status == "WARN")]
        add_check(checks, "validation", "no_hard_failures", "FAIL", hard.empty, len(hard), 0, platform)
        add_check(checks, "validation", "warnings_recorded", "WARN", warnings.empty, len(warnings), 0, platform)

        manifest = pd.read_csv(platform_result_dir(__file__, platform) / "campaign_manifest.csv")
        campaign = str(manifest.iloc[0].campaign)
        complete = complete_campaigns(__file__, platform)
        latest = max(complete) if complete else ""
        add_check(checks, "freshness", "complete_raw_campaign", "FAIL", bool(complete), sorted(complete), "at least one", platform)
        add_check(checks, "freshness", "analysis_matches_latest_complete", "FAIL",
                  bool(latest) and campaign == latest, f"manifest={campaign}; latest={latest}", "same", platform)

        sessions = load_sessions(__file__, platform)
        sizes = sorted(sessions.problem_size.unique().tolist())
        counts = sessions.groupby(["problem_size", "configuration"]).session_number.nunique()
        add_check(checks, "coverage", "size_grid", "FAIL", sizes == SIZES, sizes, SIZES, platform)
        add_check(checks, "coverage", "five_session_medians", "FAIL",
                  bool((counts == 5).all()), f"min={counts.min()}, max={counts.max()}", 5, platform)
        manifest_rows.append({
            "platform": platform, "campaign": campaign, "raw_rows": int(manifest.rows.sum()),
            "session_median_rows": len(sessions), "configurations": sessions[["problem_size", "configuration"]].drop_duplicates().shape[0],
            "energy_domain": ENERGY_DOMAINS[platform], "result_directory": str(platform_result_dir(__file__, platform)),
        })

    root = project_root(__file__)
    cpu_sources = [root / "AMD/scripts/STREAM/main_stream_amd.cpp", root / "INTEL/scripts/STREAM/main_stream_intel.cpp"]
    gpu_sources = [root / "3090/scripts/STREAM/main_stream.cu", root / "5060ti/scripts/STREAM/main_stream.cu"]
    add_check(checks, "provenance", "cpu_sources_present", "FAIL", all(p.is_file() for p in cpu_sources), cpu_sources, "both", "CPU")
    add_check(checks, "provenance", "gpu_sources_present", "FAIL", all(p.is_file() for p in gpu_sources), gpu_sources, "both", "GPU")
    if all(p.is_file() for p in cpu_sources):
        add_check(checks, "provenance", "cpu_sources_semantically_identical", "FAIL",
                  normalized_source(cpu_sources[0]) == normalized_source(cpu_sources[1]),
                  [sha256_file(p) for p in cpu_sources], "identical except platform label/default output", "CPU")
    if all(p.is_file() for p in gpu_sources):
        hashes = [sha256_file(p) for p in gpu_sources]
        add_check(checks, "provenance", "gpu_sources_byte_identical", "FAIL", len(set(hashes)) == 1, hashes, "same SHA-256", "GPU")
        required_tokens = [
            "stream_triad_kernel", "initialize_stream_vectors", "SCALAR", "BENCH_SIZE_FILTER",
            "BENCH_EXPECTED_GPU", "nvmlDeviceGetTotalEnergyConsumption", "gpu_resident",
            "cuda_stream_triad_fp32", "nvmlDeviceGetHandleByPciBusId",
        ]
        for platform, path in zip(GPU_PLATFORMS, gpu_sources):
            text = path.read_text(encoding="utf-8", errors="replace")
            missing = [token for token in required_tokens if token not in text]
            add_check(checks, "provenance", "gpu_required_tokens", "FAIL", not missing,
                      missing or "all present", required_tokens, platform)

    add_check(checks, "statistics", "session_median_primary_unit", "FAIL", True,
              "five session medians", "no n=50 pseudoreplication")
    add_check(checks, "statistics", "native_best_is_descriptive", "WARN", False,
              "selection and summary use same five sessions", "descriptive post-selection")
    add_check(checks, "semantics", "logical_not_physical_bandwidth", "WARN", False,
              "12*N logical bytes", "must never be called measured DRAM/VRAM traffic")
    add_check(checks, "semantics", "energy_domain_asymmetry", "WARN", False,
              "CPU package RAPL; GPU board NVML", "must remain explicit")
    add_check(checks, "semantics", "gpu_resident_scope", "WARN", False,
              "allocations and PCIe excluded", "must remain explicit")

    checks_df = pd.DataFrame(checks)
    checks_df.to_csv(out / "preflight_checks.csv", index=False)
    pd.DataFrame(manifest_rows).to_csv(out / "input_manifest.csv", index=False)
    hard = checks_df[(checks_df.severity == "FAIL") & (checks_df.status == "FAIL")]
    warnings = checks_df[(checks_df.severity == "WARN") & (checks_df.status == "WARN")]
    verdict = "FAIL" if len(hard) else ("PASS WITH WARNINGS" if len(warnings) else "PASS")
    report = f"""# All-platform STREAM preflight

## Verdict

**{verdict}**

## Hard failures

{markdown_table(hard)}

## Warnings

{markdown_table(warnings)}

## All checks

{markdown_table(checks_df, 300)}
"""
    (out / "preflight_report.md").write_text(report, encoding="utf-8")
    print(f"[ALL STREAM] preflight: {verdict}")
    if len(hard):
        raise SystemExit(2)


if __name__ == "__main__":
    main()

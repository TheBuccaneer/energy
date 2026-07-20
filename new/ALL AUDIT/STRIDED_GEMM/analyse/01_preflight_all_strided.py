#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from all_strided_common import *


def main() -> None:
    out = results_dir(__file__)
    checks: list[dict] = []
    manifests: list[dict] = []
    observed_size_sets: dict[str, list[int]] = {}

    for platform in PLATFORMS:
        files = platform_files(__file__, platform)
        missing = [str(path) for path in required_input_paths(files) if not path.is_file()]
        add_check(
            checks, "inputs", "required_analysis_outputs", "FAIL", not missing,
            missing or "all present", "all required files", platform,
        )
        if missing:
            continue

        validation = pd.read_csv(files.validation_checks)
        hard_failed = validation[(validation["severity"] == "FAIL") & (validation["status"] == "FAIL")]
        warnings = validation[(validation["severity"] == "WARN") & (validation["status"] == "WARN")]
        add_check(
            checks, "validation", "no_hard_failures", "FAIL", hard_failed.empty,
            f"hard_failures={len(hard_failed)}", 0, platform,
        )
        add_check(
            checks, "validation", "individual_warnings_recorded", "WARN", warnings.empty,
            f"warnings={len(warnings)}", 0, platform,
        )

        manifest_frame = pd.read_csv(files.campaign_manifest)
        manifest_campaign = campaign_from_manifest(files.campaign_manifest)
        if "rows" in manifest_frame.columns:
            if platform in CPU_PLATFORMS:
                raw_rows = int(pd.to_numeric(manifest_frame["rows"], errors="coerce").sum())
            else:
                raw_rows = int(pd.to_numeric(manifest_frame["rows"], errors="coerce").iloc[0])
        else:
            raw_rows = -1
        complete = find_complete_campaigns(files.runs, platform)
        latest_complete = max(complete) if complete else ""
        add_check(
            checks, "freshness", "complete_raw_campaign_exists", "FAIL", bool(complete),
            sorted(complete) or "none", "at least one 5-session campaign", platform,
        )
        add_check(
            checks, "freshness", "analysis_matches_latest_complete_campaign", "FAIL",
            bool(latest_complete) and manifest_campaign == latest_complete,
            f"manifest={manifest_campaign}, latest_raw={latest_complete}",
            "same campaign", platform,
        )

        session = normalize_session_medians(__file__, platform)
        sizes = sorted(session["problem_size"].dropna().astype(int).unique().tolist())
        observed_size_sets[platform] = sizes
        session_counts = session.groupby(["problem_size", "configuration"])["session_number"].nunique()
        add_check(
            checks, "coverage", "problem_sizes", "FAIL", sizes == SIZES,
            sizes, SIZES, platform,
        )
        add_check(
            checks, "coverage", "five_session_medians_per_configuration", "FAIL",
            not session_counts.empty and bool((session_counts == EXPECTED_SESSIONS).all()),
            f"min={session_counts.min() if len(session_counts) else 'NA'}, max={session_counts.max() if len(session_counts) else 'NA'}",
            EXPECTED_SESSIONS, platform,
        )

        manifests.append({
            "platform": platform,
            "platform_label": PLATFORM_LABELS[platform],
            "device_kind": DEVICE_KIND[platform],
            "energy_domain": ENERGY_DOMAIN[platform],
            "campaign": manifest_campaign,
            "latest_complete_raw_campaign": latest_complete,
            "validation_hard_failures": len(hard_failed),
            "validation_warnings": len(warnings),
            "raw_measurement_rows": raw_rows,
            "session_median_rows": len(session),
            "configurations": session[["problem_size", "configuration"]].drop_duplicates().shape[0],
            "analysis_results_directory": str(files.results),
            "run_directory": str(files.runs),
        })

    if len(observed_size_sets) == len(PLATFORMS):
        all_equal = all(sizes == SIZES for sizes in observed_size_sets.values())
        add_check(
            checks, "comparability", "same_problem_size_grid", "FAIL", all_equal,
            observed_size_sets, SIZES, "ALL",
        )

    # GPU source identity is a strong provenance check, but exact-byte inequality
    # remains a warning because comments/formatting can differ without changing semantics.
    root = project_root(__file__)
    gpu_sources = {
        p: root / p / "scripts" / "STRIDED_GEMM" / "main_strided_gemm.cu" for p in GPU_PLATFORMS
    }
    missing_sources = [str(path) for path in gpu_sources.values() if not path.is_file()]
    add_check(
        checks, "provenance", "gpu_sources_present", "FAIL", not missing_sources,
        missing_sources or "both present", "both source files", "GPU",
    )
    if not missing_sources:
        hashes = {p: sha256_file(path) for p, path in gpu_sources.items()}
        identical = len(set(hashes.values())) == 1
        add_check(
            checks, "provenance", "gpu_source_files_byte_identical", "WARN", identical,
            hashes, "identical SHA-256", "GPU",
        )
        for platform, path in gpu_sources.items():
            text = path.read_text(encoding="utf-8", errors="replace")
            required_tokens = [
                "CUBLAS_COMPUTE_32F_PEDANTIC",
                "CUBLAS_PEDANTIC_MATH",
                "nvmlDeviceGetTotalEnergyConsumption",
                "gpu_resident",
                "cublasGemmEx",
                "STRIDED_GEMM",
                "cublas_gemm_ex_fp32_pedantic_ld2n",
                "PADDING_C",
                "ld = 2 * n",
            ]
            missing_tokens = [token for token in required_tokens if token not in text]
            add_check(
                checks, "provenance", "gpu_required_measurement_tokens", "FAIL", not missing_tokens,
                missing_tokens or "all present", required_tokens, platform,
            )


    dense_unified = root / "ALL AUDIT" / "GEMM" / "results" / "unified_session_medians.csv"
    add_check(
        checks, "dense_comparison", "dense_gemm_unified_results_available", "WARN",
        dense_unified.is_file(), str(dense_unified),
        "present for automatic dense-vs-strided comparison", "ALL",
    )

    add_check(
        checks, "semantics", "energy_domain_asymmetry_acknowledged", "WARN", False,
        "CPU=package RAPL; GPU=board NVML",
        "must remain explicit in every cross-device interpretation", "ALL",
    )
    add_check(
        checks, "statistics", "primary_unit_is_session_median", "FAIL", True,
        "five session medians per configuration", "no n=50 pseudoreplication", "ALL",
    )
    add_check(
        checks, "statistics", "native_best_inference_is_descriptive", "WARN", False,
        "configuration selected and summarized on same five sessions",
        "ratio CIs are descriptive; no confirmatory p-values", "ALL",
    )

    check_df = pd.DataFrame(checks)
    check_df.to_csv(out / "preflight_checks.csv", index=False)
    pd.DataFrame(manifests).to_csv(out / "input_manifest.csv", index=False)

    hard = check_df[(check_df["severity"] == "FAIL") & (check_df["status"] == "FAIL")]
    warns = check_df[(check_df["severity"] == "WARN") & (check_df["status"] == "WARN")]
    verdict = "FAIL" if len(hard) else ("PASS WITH WARNINGS" if len(warns) else "PASS")

    report = (
        "# All-platform STRIDED_GEMM preflight audit\n\n"
        f"- Platforms: {', '.join(PLATFORM_LABELS[p] for p in PLATFORMS)}\n"
        f"- Overall verdict: **{verdict}**\n\n"
        "## Hard failures\n\n"
        f"{markdown_table(hard)}\n\n"
        "## Warnings and interpretation constraints\n\n"
        f"{markdown_table(warns)}\n\n"
        "## All checks\n\n"
        f"{markdown_table(check_df, 300)}\n\n"
        "## Statistical contract\n\n"
        "The primary unit is the median of the ten technical repetitions within each of five sessions. "
        "All cross-platform summaries therefore use n=5 session medians per selected configuration. "
        "Native-best configuration selection and its ratio bootstrap are descriptive because the same "
        "five sessions are used for selecting and summarizing a configuration. No confirmatory p-values "
        "are reported for these post-selection comparisons.\n\n"
        "## Energy-domain contract\n\n"
        "CPU values use package-only RAPL (`device_energy_j / batches`), while GPU values use NVML "
        "board energy, including device memory. Cross-device energy comparisons describe the measured "
        "device domains and must not be presented as whole-system or thermally normalized architecture-only claims.\n"
    )
    (out / "preflight_report.md").write_text(report, encoding="utf-8")

    print(f"[ALL STRIDED_GEMM] preflight: {verdict}")
    print(out / "preflight_report.md")
    if verdict == "FAIL":
        sys.exit(2)


if __name__ == "__main__":
    main()

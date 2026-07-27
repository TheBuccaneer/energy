#!/usr/bin/env python3
from __future__ import annotations

import re
import hashlib
import pandas as pd

from all_reduction_common import *


def normalized_source(path: Path) -> str:
    """Normalize only the frozen, platform-specific CPU source literals.

    AMD and Intel intentionally share the same REDUCTION implementation.
    Their source files differ only in:
      - the default output filename; and
      - the human-readable platform name in the startup banner.

    Keep every other byte semantically significant so a real kernel,
    calibration, checksum, formula, or measurement-code difference still
    fails the all-platform preflight.
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    replacements = {
        "reduction_amd.csv": "reduction_PLATFORM.csv",
        "reduction_intel.csv": "reduction_PLATFORM.csv",
        "REDUCTION(sum) | AMD | ": "REDUCTION(sum) | PLATFORM | ",
        "REDUCTION(sum) | Intel | ": "REDUCTION(sum) | PLATFORM | ",
        "platform=AMD": "platform=PLATFORM",
        "platform=INTEL": "platform=PLATFORM",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
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
            "session_median_rows": len(sessions),
            "configurations": sessions[["problem_size", "configuration"]].drop_duplicates().shape[0],
            "energy_domain": ENERGY_DOMAINS[platform],
            "result_directory": str(platform_result_dir(__file__, platform)),
        })

    root = project_root(__file__)
    cpu_sources = [
        root / "AMD/scripts/REDUCTION/main_reduction_amd.cpp",
        root / "INTEL/scripts/REDUCTION/main_reduction_intel.cpp",
    ]
    gpu_sources = [
        root / "3090/scripts/REDUCTION/main_reduction.cu",
        root / "5060ti/scripts/REDUCTION/main_reduction.cu",
    ]
    add_check(checks, "provenance", "cpu_sources_present", "FAIL", all(p.is_file() for p in cpu_sources), cpu_sources, "both", "CPU")
    add_check(checks, "provenance", "gpu_sources_present", "FAIL", all(p.is_file() for p in gpu_sources), gpu_sources, "both", "GPU")

    cpu_tokens = [
        "BLOCK_SIZE", "reduction_sum", "omp simd reduction", "expected_result",
        "openmp_blocked_sum_fp32", "BENCH_SIZE_FILTER", "BENCH_THREAD_FILTER",
        "omp_set_dynamic(0)",
    ]
    if all(p.is_file() for p in cpu_sources):
        normalized_cpu_sources = [normalized_source(path) for path in cpu_sources]
        normalized_cpu_hashes = [
            hashlib.sha256(value.encode("utf-8")).hexdigest()
            for value in normalized_cpu_sources
        ]
        cpu_sources_match = normalized_cpu_sources[0] == normalized_cpu_sources[1]
        add_check(
            checks,
            "provenance",
            "cpu_sources_semantically_identical",
            "FAIL",
            cpu_sources_match,
            {
                "raw_sha256": [sha256_file(path) for path in cpu_sources],
                "normalized_sha256": normalized_cpu_hashes,
            },
            "identical except default output filename and startup platform label",
            "CPU",
        )
        if not cpu_sources_match:
            import difflib
            diff = difflib.unified_diff(
                normalized_cpu_sources[0].splitlines(),
                normalized_cpu_sources[1].splitlines(),
                fromfile=str(cpu_sources[0]),
                tofile=str(cpu_sources[1]),
                lineterm="",
            )
            (out / "cpu_source_normalized_diff.txt").write_text(
                "\n".join(diff) + "\n",
                encoding="utf-8",
            )
        for platform, path in zip(CPU_PLATFORMS, cpu_sources):
            text = path.read_text(encoding="utf-8", errors="replace")
            missing = [token for token in cpu_tokens if token not in text]
            add_check(checks, "provenance", "cpu_required_tokens", "FAIL", not missing,
                      missing or "all present", cpu_tokens, platform)

    gpu_tokens = [
        "cub::DeviceReduce::Sum", "cub_device_reduce_sum_fp32",
        "nvmlDeviceGetTotalEnergyConsumption", "nvmlDeviceGetHandleByPciBusId",
        "gpu_resident", "BENCH_SIZE_FILTER", "BENCH_EXPECTED_GPU",
    ]
    if all(p.is_file() for p in gpu_sources):
        hashes = [sha256_file(p) for p in gpu_sources]
        add_check(checks, "provenance", "gpu_sources_byte_identical", "FAIL", len(set(hashes)) == 1, hashes, "same SHA-256", "GPU")
        for platform, path in zip(GPU_PLATFORMS, gpu_sources):
            text = path.read_text(encoding="utf-8", errors="replace")
            missing = [token for token in gpu_tokens if token not in text]
            add_check(checks, "provenance", "gpu_required_tokens", "FAIL", not missing,
                      missing or "all present", gpu_tokens, platform)

    add_check(checks, "statistics", "session_median_primary_unit", "FAIL", True,
              "five session medians", "no n=50 pseudoreplication")
    add_check(checks, "statistics", "native_best_is_descriptive", "WARN", False,
              "selection and summary use same five sessions", "descriptive post-selection")
    add_check(checks, "semantics", "logical_not_physical_bandwidth", "WARN", False,
              "4*N+4 logical bytes", "must never be called measured DRAM/VRAM traffic")
    add_check(checks, "semantics", "internal_reduction_traffic_excluded", "WARN", False,
              "CPU partials and CUB workspace excluded", "semantic data-volume anchor only")
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
    report = f"""# All-platform REDUCTION preflight

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
    print(f"[ALL REDUCTION] preflight: {verdict}")
    if len(hard):
        raise SystemExit(2)


if __name__ == "__main__":
    main()

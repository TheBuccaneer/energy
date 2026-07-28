#!/usr/bin/env python3
"""
Validiert alle vier offiziellen AXPY-Kampagnen und aggregiert session-bewusst.

Punktschätzer:
1. Median der zehn Wiederholungen innerhalb jeder Session.
2. Median der fünf Session-Mediane.

Konfidenzintervalle:
Deterministischer Percentile-Bootstrap über die fünf Session-Mediane.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

from axpy_analysis_common import (
    HEADER_V2,
    PLATFORMS,
    SIZES,
    Issue,
    PlatformSpec,
    bootstrap_median_ci,
    close,
    cv,
    exact_int,
    expected_problem_spec,
    finite_float,
    format_float,
    is_true,
    mad,
    median,
    parse_throttle,
    session_number,
    select_campaign,
    throttle_labels,
    stability_class,
    write_csv,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AXPY: Validierung und session-bewusste Aggregation aller Plattformen."
    )
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("new/analyse/AXPY/all_platforms"))
    parser.add_argument("--sessions", type=int, default=5)
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--intel-dir", type=Path)
    parser.add_argument("--amd-dir", type=Path)
    parser.add_argument("--gpu3090-dir", type=Path)
    parser.add_argument("--gpu5060ti-dir", type=Path)
    parser.add_argument("--intel-campaign")
    parser.add_argument("--amd-campaign")
    parser.add_argument("--gpu3090-campaign")
    parser.add_argument("--gpu5060ti-campaign")
    return parser.parse_args()


def cpu_throttle_sentinel_ok(value: object) -> bool:
    """CPU-Autorenquellen schreiben throttle_reasons historisch leer; -1 ist ebenfalls gültig."""
    raw = str(value).strip()
    if raw == "":
        return True
    try:
        return finite_float(raw) == -1.0
    except (TypeError, ValueError):
        return False


def add_issue(
    issues: list[Issue],
    severity: str,
    spec: PlatformSpec,
    path: Path | None,
    row: int,
    message: str,
) -> None:
    issues.append(
        Issue(
            severity=severity,
            platform=spec.key,
            file=path.name if path else "",
            row=row,
            message=message,
        )
    )


def validate_row(
    *,
    row: dict,
    row_number: int,
    path: Path,
    spec: PlatformSpec,
    expected_session_id: str,
    repetitions: int,
    issues: list[Issue],
) -> dict | None:
    where = f"{path.name}:{row_number}"
    row_failed = False

    def fail(message: str) -> None:
        nonlocal row_failed
        row_failed = True
        add_issue(issues, "FAIL", spec, path, row_number, message)

    def warn(message: str) -> None:
        add_issue(issues, "WARN", spec, path, row_number, message)

    try:
        if list(row.keys()) != HEADER_V2:
            fail("Spaltenreihenfolge weicht vom 45-Spalten-Schema cpu-gpu-v2 ab.")
            return None

        if row["schema_version"] != "cpu-gpu-v2":
            fail(f"schema_version={row['schema_version']!r}, erwartet cpu-gpu-v2.")
        if row["session_id"] != expected_session_id:
            fail(f"session_id={row['session_id']!r}, erwartet {expected_session_id!r}.")
        if row["workload"] != "AXPY":
            fail(f"workload={row['workload']!r}, erwartet AXPY.")
        if row["implementation"] != spec.implementation:
            fail(
                f"implementation={row['implementation']!r}, "
                f"erwartet {spec.implementation!r}."
            )
        if row["execution_mode"] != spec.execution_mode:
            fail(
                f"execution_mode={row['execution_mode']!r}, "
                f"erwartet {spec.execution_mode!r}."
            )
        if spec.device_token.lower() not in row["device_name"].lower():
            fail(
                f"device_name={row['device_name']!r} enthält "
                f"{spec.device_token!r} nicht."
            )
        if not is_true(row["checksum_ok"]):
            fail("checksum_ok ist nicht wahr.")

        sequence = exact_int(row["sequence_index"])
        run_id = exact_int(row["run_id_global"])
        repetition = exact_int(row["repetition"])
        threads = exact_int(row["num_threads"])
        n = exact_int(row["problem_size"])
        batches = exact_int(row["batches"])
        flops_total = exact_int(row["flops_total"])
        logical_bytes = exact_int(row["logical_bytes_per_op"])

        e2e = finite_float(row["e2e_time_s"])
        kernel = finite_float(row["kernel_time_s"])
        wall = finite_float(row["wall_time_s"])
        device_energy = finite_float(row["device_energy_j"])
        total_energy = finite_float(row["total_energy_j"])
        dram_energy = finite_float(row["dram_energy_j"])
        energy_per_op = finite_float(row["energy_per_op_j"])
        energy_per_second = finite_float(row["energy_per_second_j"])
        energy_per_flop = finite_float(row["energy_per_flop_j"])
        time_kernel_ms = finite_float(row["time_per_op_ms_kernel"])
        time_e2e_ms = finite_float(row["time_per_op_ms_e2e"])
        gflops = finite_float(row["gflops_per_s"])
        avg_power = finite_float(row["avg_power_w"])
        pcie_gen = finite_float(row["pcie_gen"])
        pcie_width = finite_float(row["pcie_width"])
        sm_clock = finite_float(row["sm_clock_mhz"])
        clock_before = finite_float(row["clock_before_mhz"])
        clock_after = finite_float(row["clock_after_mhz"])
        mem_clock = finite_float(row["mem_clock_mhz"])
        temp_c = finite_float(row["temp_c"])
        temp_before = finite_float(row["temp_before_c"])
        temp_after = finite_float(row["temp_after_c"])

        if n not in SIZES:
            fail(f"Unerwartete Problemgröße N={n}.")
        if threads not in spec.threads:
            fail(f"Unerwartete Threadzahl {threads}; erwartet {spec.threads}.")
        if not 1 <= repetition <= repetitions:
            fail(f"repetition={repetition} außerhalb 1..{repetitions}.")
        if not 1 <= batches <= 250_000:
            fail(f"batches={batches} außerhalb 1..250000.")

        expected_spec = expected_problem_spec(n)
        if row["problem_spec"] != expected_spec:
            fail(
                "problem_spec verletzt den eingefrorenen AXPY-Vertrag: "
                f"actual={row['problem_spec']!r}, expected={expected_spec!r}."
            )

        positive_metrics = {
            "e2e_time_s": e2e,
            "kernel_time_s": kernel,
            "wall_time_s": wall,
            "device_energy_j": device_energy,
            "total_energy_j": total_energy,
            "energy_per_op_j": energy_per_op,
            "energy_per_second_j": energy_per_second,
            "energy_per_flop_j": energy_per_flop,
            "time_per_op_ms_kernel": time_kernel_ms,
            "time_per_op_ms_e2e": time_e2e_ms,
            "gflops_per_s": gflops,
            "avg_power_w": avg_power,
        }
        for name, value in positive_metrics.items():
            if value <= 0.0:
                fail(f"{name} ist nicht positiv: {value}.")

        if spec.kind == "cpu":
            if not (close(e2e, kernel) and close(e2e, wall)):
                fail("CPU-Zeitidentität e2e=kernel=wall verletzt.")
        else:
            if not close(e2e, wall):
                fail("GPU-Zeitidentität e2e=wall verletzt.")
            if kernel > e2e:
                excess = kernel - e2e
                allowed = max(0.0005, 0.005 * e2e)
                if excess > allowed:
                    fail(
                        "Materieller GPU-Timing-Widerspruch: "
                        f"kernel-e2e={excess:.9g}s > {allowed:.9g}s."
                    )
                else:
                    warn(
                        "Kleine GPU-Timing-Inversion innerhalb der erlaubten "
                        f"Toleranz: kernel-e2e={excess:.9g}s."
                    )

        expected_total = (
            device_energy + dram_energy if dram_energy >= 0.0 else device_energy
        )
        if dram_energy < 0.0 and dram_energy != -1.0:
            fail(f"dram_energy_j={dram_energy}, erwartet -1 oder >=0.")
        if not close(total_energy, expected_total):
            fail(
                f"total_energy_j={total_energy} passt nicht zu "
                f"device+dram={expected_total}."
            )

        expected_flops = 2 * n * batches
        expected_bytes = 12 * n
        if flops_total != expected_flops:
            fail(f"flops_total={flops_total}, erwartet {expected_flops}.")
        if logical_bytes != expected_bytes:
            fail(f"logical_bytes_per_op={logical_bytes}, erwartet {expected_bytes}.")

        formulae = (
            ("energy_per_op_j", energy_per_op, device_energy / batches),
            ("energy_per_second_j", energy_per_second, device_energy / wall),
            ("energy_per_flop_j", energy_per_flop, device_energy / flops_total),
            ("time_per_op_ms_kernel", time_kernel_ms, 1000.0 * kernel / batches),
            ("time_per_op_ms_e2e", time_e2e_ms, 1000.0 * e2e / batches),
            ("gflops_per_s", gflops, flops_total / kernel / 1.0e9),
            ("avg_power_w", avg_power, device_energy / wall),
        )
        for name, actual, expected in formulae:
            if not close(actual, expected):
                fail(
                    f"Formelanker {name}: actual={actual:.17g}, "
                    f"expected={expected:.17g}."
                )

        expected_status = (
            "below" if e2e < 0.75 else ("in_range" if e2e <= 1.25 else "above")
        )
        if row["runtime_status"] != expected_status:
            fail(
                f"runtime_status={row['runtime_status']!r}, "
                f"rechnerisch {expected_status!r}."
            )
        if expected_status == "below":
            fail("Offizielle Messzeile liegt unter 0,75 s.")
        elif expected_status == "above":
            # Wissenschaftlich nutzbar, aber separat und kompakt zusammenfassen.
            pass

        throttle = -1
        throttle_hex = "-1"
        throttle_text = "not_applicable"
        if spec.kind == "gpu":
            throttle = parse_throttle(row["throttle_reasons"])
            throttle_hex = f"0x{throttle:x}"
            throttle_text = throttle_labels(throttle)
            for key, value in (
                ("pcie_gen", pcie_gen), ("pcie_width", pcie_width),
                ("sm_clock_mhz", sm_clock), ("clock_before_mhz", clock_before),
                ("clock_after_mhz", clock_after), ("mem_clock_mhz", mem_clock),
                ("temp_c", temp_c), ("temp_before_c", temp_before),
                ("temp_after_c", temp_after),
            ):
                if value <= 0:
                    fail(f"{key} muss auf der GPU positiv sein.")
            if threads != -1:
                fail("GPU num_threads muss -1 sein.")
            if dram_energy != -1.0:
                fail("GPU dram_energy_j muss -1 sein.")
            for key in ("cpu_cycles", "cpu_instructions", "cpu_cache_misses"):
                if finite_float(row[key]) != -1.0:
                    fail(f"GPU-Sentinel {key} muss -1 sein.")
            if finite_float(row["cpu_ipc"]) != -1.0:
                fail("GPU-Sentinel cpu_ipc muss numerisch -1 sein.")
        else:
            for key, value in (
                ("pcie_gen", pcie_gen), ("pcie_width", pcie_width),
                ("sm_clock_mhz", sm_clock), ("mem_clock_mhz", mem_clock),
            ):
                if value != -1.0:
                    fail(f"CPU-Sentinel {key} muss -1 sein.")
            for key, value in (
                ("clock_before_mhz", clock_before),
                ("clock_after_mhz", clock_after),
                ("temp_c", temp_c),
                ("temp_before_c", temp_before),
                ("temp_after_c", temp_after),
            ):
                if value <= 0:
                    fail(f"CPU-Telemetrie {key} muss positiv sein.")
            for key in ("cpu_cycles", "cpu_instructions", "cpu_cache_misses"):
                if finite_float(row[key]) != -1.0:
                    fail(f"CPU-Sentinel {key} muss -1 sein.")
            if finite_float(row["cpu_ipc"]) != -1.0:
                fail("CPU-Sentinel cpu_ipc muss numerisch -1 sein.")
            if not cpu_throttle_sentinel_ok(row["throttle_reasons"]):
                fail("CPU-Sentinel throttle_reasons muss leer oder numerisch -1 sein.")

        if row_failed:
            return None

        time_e2e_op_s = e2e / batches
        time_kernel_op_s = kernel / batches
        device_energy_op_j = device_energy / batches
        total_energy_op_j = total_energy / batches
        dram_energy_op_j = dram_energy / batches if dram_energy >= 0 else math.nan
        logical_bandwidth_gb_s = logical_bytes / time_e2e_op_s / 1.0e9
        edp_device_j_s = device_energy_op_j * time_e2e_op_s
        ed2p_device_j_s2 = device_energy_op_j * time_e2e_op_s * time_e2e_op_s

        return {
            "platform": spec.key,
            "platform_label": spec.label,
            "kind": spec.kind,
            "campaign_id": expected_session_id.rsplit("_session", 1)[0],
            "session_id": expected_session_id,
            "session_number": session_number(expected_session_id),
            "sequence_index": sequence,
            "repetition": repetition,
            "device_name": row["device_name"],
            "threads": threads,
            "problem_size": n,
            "problem_spec": row["problem_spec"],
            "batches": batches,
            "runtime_status": row["runtime_status"],
            "time_e2e_op_s": time_e2e_op_s,
            "time_kernel_op_s": time_kernel_op_s,
            "device_energy_op_j": device_energy_op_j,
            "total_energy_op_j": total_energy_op_j,
            "dram_energy_op_j": dram_energy_op_j,
            "avg_power_w": avg_power,
            "gflops_per_s": gflops,
            "logical_bandwidth_gb_s": logical_bandwidth_gb_s,
            "edp_device_j_s": edp_device_j_s,
            "ed2p_device_j_s2": ed2p_device_j_s2,
            "pcie_gen": pcie_gen,
            "pcie_width": pcie_width,
            "sm_clock_mhz": sm_clock,
            "clock_before_mhz": clock_before,
            "clock_after_mhz": clock_after,
            "mem_clock_mhz": mem_clock,
            "temp_c": temp_c,
            "temp_before_c": temp_before,
            "temp_after_c": temp_after,
            "clock_after_before_ratio": clock_after / clock_before,
            "throttle_reasons_hex": throttle_hex,
            "throttle_reasons_int": throttle,
            "throttle_labels": throttle_text,
            "throttle_nonzero": 1 if throttle > 0 else 0,
        }
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        fail(f"Parse-/Validierungsfehler bei {where}: {exc}")
        return None


def aggregate_session(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                row["platform"], row["platform_label"], row["kind"],
                row["campaign_id"], row["session_id"], row["session_number"],
                row["problem_size"], row["threads"],
            )
        ].append(row)

    metrics = (
        "time_e2e_op_s", "time_kernel_op_s", "device_energy_op_j",
        "total_energy_op_j", "dram_energy_op_j", "avg_power_w",
        "gflops_per_s", "logical_bandwidth_gb_s", "edp_device_j_s",
        "ed2p_device_j_s2", "temp_c", "temp_before_c", "temp_after_c",
        "clock_before_mhz", "clock_after_mhz", "clock_after_before_ratio",
        "sm_clock_mhz", "mem_clock_mhz",
    )
    result = []
    for key, group in sorted(grouped.items()):
        (
            platform, platform_label, kind, campaign_id, session_id,
            session_no, problem_size, threads,
        ) = key
        item = {
            "platform": platform,
            "platform_label": platform_label,
            "kind": kind,
            "campaign_id": campaign_id,
            "session_id": session_id,
            "session_number": session_no,
            "problem_size": problem_size,
            "threads": threads,
            "n_repetitions": len(group),
            "n_in_range": sum(r["runtime_status"] == "in_range" for r in group),
            "n_above": sum(r["runtime_status"] == "above" for r in group),
            "n_below": sum(r["runtime_status"] == "below" for r in group),
            "throttle_nonzero_rows": sum(r["throttle_nonzero"] for r in group),
            "throttle_masks": "|".join(sorted({r["throttle_reasons_hex"] for r in group})),
            "median_batches": median([float(r["batches"]) for r in group]),
        }
        for metric in metrics:
            values = [
                float(r[metric]) for r in group
                if math.isfinite(float(r[metric]))
            ]
            item[f"median_{metric}"] = median(values)
            item[f"mad_{metric}"] = mad(values)
            item[f"cv_{metric}"] = cv(values)
        result.append(item)
    return result


def aggregate_config(
    raw_rows: list[dict],
    session_rows: list[dict],
    bootstrap_resamples: int,
) -> list[dict]:
    raw_grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in raw_rows:
        raw_grouped[
            (
                row["platform"], row["platform_label"], row["kind"],
                row["campaign_id"], row["problem_size"], row["threads"],
            )
        ].append(row)

    session_grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in session_rows:
        session_grouped[
            (
                row["platform"], row["platform_label"], row["kind"],
                row["campaign_id"], row["problem_size"], row["threads"],
            )
        ].append(row)

    metrics = (
        "time_e2e_op_s", "time_kernel_op_s", "device_energy_op_j",
        "total_energy_op_j", "dram_energy_op_j", "avg_power_w",
        "gflops_per_s", "logical_bandwidth_gb_s", "edp_device_j_s",
        "ed2p_device_j_s2", "temp_c", "temp_before_c", "temp_after_c",
        "clock_before_mhz", "clock_after_mhz", "clock_after_before_ratio",
        "sm_clock_mhz", "mem_clock_mhz",
    )

    result = []
    for key, sessions in sorted(session_grouped.items()):
        platform, platform_label, kind, campaign_id, problem_size, threads = key
        raw = raw_grouped[key]
        item = {
            "platform": platform,
            "platform_label": platform_label,
            "kind": kind,
            "campaign_id": campaign_id,
            "problem_size": problem_size,
            "threads": threads,
            "n_rows": len(raw),
            "n_sessions": len(sessions),
            "n_in_range": sum(r["runtime_status"] == "in_range" for r in raw),
            "n_above": sum(r["runtime_status"] == "above" for r in raw),
            "n_below": sum(r["runtime_status"] == "below" for r in raw),
            "throttle_nonzero_rows": sum(r["throttle_nonzero"] for r in raw),
            "throttle_masks": "|".join(sorted({r["throttle_reasons_hex"] for r in raw})),
            "median_batches": median([float(r["batches"]) for r in raw]),
        }

        for metric_index, metric in enumerate(metrics):
            session_values = [
                float(s[f"median_{metric}"]) for s in sessions
                if math.isfinite(float(s[f"median_{metric}"]))
            ]
            raw_values = [
                float(r[metric]) for r in raw
                if math.isfinite(float(r[metric]))
            ]
            estimate = median(session_values)
            seed = (
                0x41585059
                + int(problem_size)
                + 1009 * int(threads)
                + 65537 * metric_index
                + sum(ord(ch) for ch in platform)
            )
            ci_low, ci_high = bootstrap_median_ci(
                session_values,
                resamples=bootstrap_resamples,
                seed=seed,
            )
            item[f"median_{metric}"] = estimate
            item[f"ci95_low_{metric}"] = ci_low
            item[f"ci95_high_{metric}"] = ci_high
            item[f"mad_session_{metric}"] = mad(session_values)
            item[f"cv_session_{metric}"] = cv(session_values)
            item[f"cv_all_rows_{metric}"] = cv(raw_values)

        energy_cv = item["cv_all_rows_device_energy_op_j"]
        time_cv = item["cv_all_rows_time_e2e_op_s"]
        item["stability_time"] = stability_class(time_cv)
        item["stability_energy"] = stability_class(energy_cv)

        dram = item["median_dram_energy_op_j"]
        total = item["median_total_energy_op_j"]
        item["dram_share_total"] = (
            dram / total
            if math.isfinite(dram) and math.isfinite(total) and total > 0
            else math.nan
        )
        result.append(item)
    return result


def main() -> int:
    args = parse_args()
    repo = args.repo.expanduser().resolve()
    output = (repo / args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    overrides = {
        "intel": args.intel_dir,
        "amd": args.amd_dir,
        "3090": args.gpu3090_dir,
        "5060ti": args.gpu5060ti_dir,
    }
    campaign_overrides = {
        "intel": args.intel_campaign,
        "amd": args.amd_campaign,
        "3090": args.gpu3090_campaign,
        "5060ti": args.gpu5060ti_campaign,
    }

    issues: list[Issue] = []
    all_rows: list[dict] = []
    rejected_rows: list[dict] = []
    campaign_manifest: list[dict] = []

    for spec in PLATFORMS:
        directory = (
            (repo / overrides[spec.key]).resolve()
            if overrides[spec.key] is not None
            else (repo / spec.relative_dir).resolve()
        )
        try:
            prefix, files, complete = select_campaign(
                directory, args.sessions, campaign_overrides[spec.key]
            )
        except FileNotFoundError as exc:
            add_issue(issues, "FAIL", spec, None, 0, str(exc))
            continue

        if not complete:
            severity = "WARN" if args.allow_incomplete else "FAIL"
            add_issue(
                issues, severity, spec, None, 0,
                f"Kampagne {prefix!r} ist nicht vollständig: "
                f"Sessions {[session for session, _ in files]}.",
            )

        expected_session_set = set(range(1, args.sessions + 1))
        actual_session_set = {session for session, _ in files}
        if not args.allow_incomplete and actual_session_set != expected_session_set:
            continue

        campaign_manifest.append(
            {
                "platform": spec.key,
                "platform_label": spec.label,
                "directory": str(directory),
                "campaign_id": prefix,
                "selection_mode": "locked" if campaign_overrides[spec.key] else "latest_complete",
                "sessions": sorted(actual_session_set),
                "complete": complete,
                "manifest_file": str(directory / f"{prefix}_manifest.txt"),
                "manifest_exists": (directory / f"{prefix}_manifest.txt").exists(),
            }
        )
        if not (directory / f"{prefix}_manifest.txt").exists():
            add_issue(
                issues, "WARN", spec, None, 0,
                f"Provenienzmanifest fehlt: {prefix}_manifest.txt.",
            )

        expected_rows_per_session = len(SIZES) * len(spec.threads) * args.reps

        for session_number, path in files:
            try:
                with path.open(newline="", encoding="utf-8") as handle:
                    reader = csv.DictReader(handle)
                    header = reader.fieldnames or []
                    rows = list(reader)
            except (OSError, csv.Error) as exc:
                add_issue(issues, "FAIL", spec, path, 0, f"CSV-Lesefehler: {exc}")
                continue

            if header != HEADER_V2:
                add_issue(
                    issues, "FAIL", spec, path, 1,
                    f"Header stimmt nicht exakt; {len(header)} statt 45 Spalten "
                    "oder falsche Reihenfolge.",
                )
                continue
            if len(rows) != expected_rows_per_session:
                add_issue(
                    issues, "FAIL", spec, path, 0,
                    f"{len(rows)} Zeilen statt erwarteter "
                    f"{expected_rows_per_session}.",
                )
            if not path.with_suffix(".log").exists():
                add_issue(
                    issues, "WARN", spec, path, 0,
                    "Zugehörige Logdatei fehlt.",
                )

            expected_session_id = f"{prefix}_session{session_number}"
            seen = Counter()
            sequence_values = []
            run_ids = []

            for row_number, row in enumerate(rows, start=2):
                issue_start = len(issues)
                normalized = validate_row(
                    row=row,
                    row_number=row_number,
                    path=path,
                    spec=spec,
                    expected_session_id=expected_session_id,
                    repetitions=args.reps,
                    issues=issues,
                )
                if normalized is None:
                    reasons = " | ".join(
                        issue.message for issue in issues[issue_start:]
                        if issue.severity == "FAIL"
                    ) or "Zeile wurde verworfen."
                    rejected_rows.append({
                        "platform": spec.key,
                        "file": path.name,
                        "row": row_number,
                        "problem_size": row.get("problem_size", ""),
                        "num_threads": row.get("num_threads", ""),
                        "repetition": row.get("repetition", ""),
                        "reasons": reasons,
                        "raw_row_json": json.dumps(row, ensure_ascii=False, sort_keys=True),
                    })
                    continue
                all_rows.append(normalized)
                sequence_values.append(normalized["sequence_index"])
                run_ids.append(exact_int(row["run_id_global"]))
                seen[
                    (
                        normalized["problem_size"],
                        normalized["threads"],
                        normalized["repetition"],
                    )
                ] += 1

            if sequence_values != list(range(1, len(rows) + 1)):
                add_issue(
                    issues, "FAIL", spec, path, 0,
                    "sequence_index ist nicht lückenlos 1..Zeilenzahl.",
                )
            if run_ids != sequence_values:
                add_issue(
                    issues, "FAIL", spec, path, 0,
                    "run_id_global entspricht nicht sequence_index.",
                )
            expected_keys = {
                (n, threads, repetition)
                for n in SIZES
                for threads in spec.threads
                for repetition in range(1, args.reps + 1)
            }
            if set(seen) != expected_keys:
                add_issue(
                    issues, "FAIL", spec, path, 0,
                    f"Konfigurationsraster unvollständig oder unerwartet: "
                    f"{len(seen)} statt {len(expected_keys)} Schlüssel.",
                )
            duplicate_count = sum(count - 1 for count in seen.values() if count > 1)
            if duplicate_count:
                add_issue(
                    issues, "FAIL", spec, path, 0,
                    f"{duplicate_count} doppelte Messschlüssel.",
                )

    session_rows = aggregate_session(all_rows)
    config_rows = aggregate_config(
        all_rows, session_rows, args.bootstrap_resamples
    )

    raw_fields = list(all_rows[0].keys()) if all_rows else []
    session_fields = list(session_rows[0].keys()) if session_rows else []
    config_fields = list(config_rows[0].keys()) if config_rows else []

    if raw_fields:
        write_csv(output / "axpy_normalized_rows.csv", all_rows, raw_fields)
    if session_fields:
        write_csv(output / "axpy_session_summary.csv", session_rows, session_fields)
    if config_fields:
        write_csv(output / "axpy_config_summary.csv", config_rows, config_fields)

    write_csv(
        output / "axpy_rejected_rows.csv",
        rejected_rows,
        ["platform", "file", "row", "problem_size", "num_threads", "repetition", "reasons", "raw_row_json"],
    )

    runtime_summary = []
    for spec in PLATFORMS:
        rows = [row for row in all_rows if row["platform"] == spec.key]
        above = [row for row in rows if row["runtime_status"] == "above"]
        values = [float(row["time_e2e_op_s"]) * float(row["batches"]) for row in above]
        runtime_summary.append({
            "platform": spec.key,
            "platform_label": spec.label,
            "total_rows": len(rows),
            "above_rows": len(above),
            "above_fraction": (len(above) / len(rows)) if rows else math.nan,
            "min_above_e2e_s": min(values) if values else math.nan,
            "median_above_e2e_s": median(values),
            "max_above_e2e_s": max(values) if values else math.nan,
        })
    write_csv(
        output / "axpy_runtime_window_summary.csv",
        runtime_summary,
        list(runtime_summary[0].keys()) if runtime_summary else [],
    )

    throttle_summary = []
    for spec in PLATFORMS:
        counts = Counter(
            (row["throttle_reasons_hex"], row["throttle_labels"])
            for row in all_rows if row["platform"] == spec.key
        )
        for (mask, labels), count in sorted(counts.items()):
            throttle_summary.append({
                "platform": spec.key,
                "platform_label": spec.label,
                "throttle_reasons_hex": mask,
                "throttle_labels": labels,
                "rows": count,
            })
    write_csv(
        output / "axpy_throttle_summary.csv",
        throttle_summary,
        ["platform", "platform_label", "throttle_reasons_hex", "throttle_labels", "rows"],
    )

    # Telemetrie bleibt in der normalisierten Ausgabe erhalten und wird
    # kompakt nach Plattform und Konfiguration zusammengefasst.
    telemetry_by_config = []
    grouped_telemetry = defaultdict(list)
    for row in all_rows:
        grouped_telemetry[(row["platform"], row["problem_size"], row["threads"])].append(row)
    for (platform, n, threads), rows in sorted(grouped_telemetry.items()):
        temps = [float(row["temp_c"]) for row in rows]
        clock_before_values = [float(row["clock_before_mhz"]) for row in rows]
        clock_after_values = [float(row["clock_after_mhz"]) for row in rows]
        ratios = [float(row["clock_after_before_ratio"]) for row in rows]
        telemetry_by_config.append({
            "platform": platform,
            "problem_size": n,
            "threads": threads,
            "rows": len(rows),
            "median_temp_c": median(temps),
            "max_temp_c": max(temps),
            "rows_temp_ge_95c": sum(value >= 95.0 for value in temps),
            "rows_temp_ge_100c": sum(value >= 100.0 for value in temps),
            "median_clock_before_mhz": median(clock_before_values),
            "median_clock_after_mhz": median(clock_after_values),
            "median_clock_after_before_ratio": median(ratios),
            "min_clock_after_before_ratio": min(ratios),
        })
    write_csv(
        output / "axpy_telemetry_by_config.csv",
        telemetry_by_config,
        list(telemetry_by_config[0].keys()) if telemetry_by_config else [],
    )

    telemetry_summary = []
    for spec in PLATFORMS:
        rows = [row for row in all_rows if row["platform"] == spec.key]
        if not rows:
            continue
        temps = [float(row["temp_c"]) for row in rows]
        clocks_after = [float(row["clock_after_mhz"]) for row in rows]
        ratios = [float(row["clock_after_before_ratio"]) for row in rows]
        hot_rows = [row for row in rows if float(row["temp_c"]) >= 95.0]
        cool_rows = [row for row in rows if float(row["temp_c"]) < 90.0]
        hot_clock = median([float(row["clock_after_mhz"]) for row in hot_rows])
        cool_clock = median([float(row["clock_after_mhz"]) for row in cool_rows])
        hot_to_cool = (
            hot_clock / cool_clock
            if math.isfinite(hot_clock) and math.isfinite(cool_clock) and cool_clock > 0
            else math.nan
        )
        hot_after_before = median([
            float(row["clock_after_before_ratio"]) for row in hot_rows
        ])
        drop_evidence = (
            "yes"
            if spec.kind == "cpu" and len(hot_rows) >= 5
            and math.isfinite(hot_after_before) and hot_after_before < 0.90
            else "no"
            if spec.kind == "cpu" and len(hot_rows) >= 5
            else "not_assessable"
        )
        item = {
            "platform": spec.key,
            "platform_label": spec.label,
            "kind": spec.kind,
            "rows": len(rows),
            "median_temp_c": median(temps),
            "max_temp_c": max(temps),
            "rows_temp_ge_95c": sum(value >= 95.0 for value in temps),
            "rows_temp_ge_100c": sum(value >= 100.0 for value in temps),
            "median_clock_after_mhz": median(clocks_after),
            "median_clock_after_before_ratio": median(ratios),
            "min_clock_after_before_ratio": min(ratios),
            "hot_rows": len(hot_rows),
            "cool_rows": len(cool_rows),
            "hot_median_clock_after_mhz": hot_clock,
            "cool_median_clock_after_mhz": cool_clock,
            "hot_to_cool_clock_ratio": hot_to_cool,
            "hot_median_clock_after_before_ratio": hot_after_before,
            "within_window_clock_drop_detected": drop_evidence,
        }
        telemetry_summary.append(item)
        if spec.kind == "cpu" and item["max_temp_c"] >= 95.0:
            add_issue(
                issues, "WARN", spec, None, 0,
                f"CPU-Telemetrie erreicht {item['max_temp_c']:.1f} °C; "
                f"{item['rows_temp_ge_95c']} Zeilen liegen bei mindestens 95 °C. "
                "Siehe axpy_telemetry_report.md.",
            )
        if drop_evidence == "yes":
            add_issue(
                issues, "WARN", spec, None, 0,
                f"Möglicher Takteinbruch in heißen Zeilen: Median clock_after/clock_before "
                f"={hot_after_before:.3f}. Teilanalyse erforderlich, keine automatische Neumessung.",
            )
    write_csv(
        output / "axpy_telemetry_summary.csv",
        telemetry_summary,
        list(telemetry_summary[0].keys()) if telemetry_summary else [],
    )

    telemetry_report = [
        "# AXPY – Temperatur- und Takttelemetrie",
        "",
        "Die Telemetrie ist ein Diagnoseanker. Hohe Temperatur allein verwirft keine "
        "Messung; ein möglicher thermischer Effekt wird separat über den Taktvergleich "
        "heißer (≥95 °C) und kühler (<90 °C) CPU-Zeilen markiert.",
        "",
        "| Plattform | Median Temp. | Max Temp. | ≥95 °C | ≥100 °C | Median Takt nachher | heiß: nachher/vorher | heiß/kühl Takt | Taktabfall im Messfenster |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in telemetry_summary:
        telemetry_report.append(
            f"| {item['platform_label']} | {format_float(item['median_temp_c'], 5)} °C | "
            f"{format_float(item['max_temp_c'], 5)} °C | {item['rows_temp_ge_95c']} | "
            f"{item['rows_temp_ge_100c']} | "
            f"{format_float(item['median_clock_after_mhz'], 6)} MHz | "
            f"{format_float(item['hot_median_clock_after_before_ratio'], 5)} | "
            f"{format_float(item['hot_to_cool_clock_ratio'], 5)} | "
            f"{item['within_window_clock_drop_detected']} |"
        )
    telemetry_report += [
        "",
        "Konfigurationsdetails stehen in `axpy_telemetry_by_config.csv`. "
        "Die Kennzahl `within_window_clock_drop_detected` verwendet primär "
        "`clock_after/clock_before` innerhalb der heißen Zeilen. Sie erkennt nur "
        "zusätzlichen Taktabfall im Messfenster und schließt keinen bereits vor "
        "Messbeginn reduzierten Taktzustand aus. "
        "`not_assessable` bedeutet, dass weniger als fünf heiße CPU-Zeilen vorliegen.",
    ]
    (output / "axpy_telemetry_report.md").write_text(
        "\n".join(telemetry_report) + "\n", encoding="utf-8"
    )

    issue_rows = [
        {
            "severity": issue.severity,
            "platform": issue.platform,
            "file": issue.file,
            "row": issue.row,
            "message": issue.message,
        }
        for issue in issues
    ]
    write_csv(
        output / "axpy_validation_issues.csv",
        issue_rows,
        ["severity", "platform", "file", "row", "message"],
    )
    write_json(output / "axpy_selected_campaigns.json", campaign_manifest)

    fail_count = sum(issue.severity == "FAIL" for issue in issues)
    warn_count = sum(issue.severity == "WARN" for issue in issues)
    status = "FAIL" if fail_count else ("PASS MIT WARNUNGEN" if warn_count else "PASS")

    platform_counts = Counter(row["platform"] for row in all_rows)
    report = [
        "# AXPY – Validierung und Aggregation",
        "",
        f"**Status: {status}**",
        "",
        "## Ausgewählte Kampagnen",
        "",
        "| Plattform | Kampagne | Sessions | Manifest | Rohzeilen |",
        "|---|---|---:|---|---:|",
    ]
    for item in campaign_manifest:
        report.append(
            f"| {item['platform_label']} | `{item['campaign_id']}` | "
            f"{len(item['sessions'])} | "
            f"{'ja' if item['manifest_exists'] else 'nein'} | "
            f"{platform_counts[item['platform']]} |"
        )

    report += [
        "",
        "## Laufzeitfenster",
        "",
        "| Plattform | Zeilen | über 1,25 s | Anteil | Median der Warnungen | Maximum |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in runtime_summary:
        report.append(
            f"| {item['platform_label']} | {item['total_rows']} | {item['above_rows']} | "
            f"{format_float(item['above_fraction'] * 100, 4)} % | "
            f"{format_float(item['median_above_e2e_s'], 5)} s | "
            f"{format_float(item['max_above_e2e_s'], 5)} s |"
        )

    report += [
        "",
        "## Eingefrorener AXPY-Vertrag",
        "",
        "Jede Zeile wurde positionsgenau gegen `problem_spec` mit `alpha=3.0`, "
        "den eingefrorenen Periodenmustern, Reset außerhalb des Messfensters "
        "und `max_batches=250000` geprüft. Fehlerhafte Zeilen werden verworfen "
        "und in `axpy_rejected_rows.csv` dokumentiert.",
        "",
        "## Aggregationsmethode",
        "",
        "Für jede Konfiguration wird zuerst innerhalb jeder Session der Median "
        "der zehn Wiederholungen berechnet. Der finale Punktschätzer ist der "
        "Median der fünf Session-Mediane. Die 95-%-Intervalle sind ein "
        "deterministischer Percentile-Bootstrap über die Session-Mediane.",
        "",
        "Die primäre Laufzeitmetrik ist E2E-Zeit pro logischer AXPY-Operation. "
        "Die primäre Energiegröße ist `device_energy_j / batches`: auf CPUs "
        "Package-Energie, auf GPUs Board-Energie.",
        "",
        "## Stabilität",
        "",
        "| Plattform | Konfigurationen | Median Zeit-CV | Max Zeit-CV | "
        "Median Energie-CV | Max Energie-CV |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for spec in PLATFORMS:
        rows = [row for row in config_rows if row["platform"] == spec.key]
        time_cvs = [
            row["cv_all_rows_time_e2e_op_s"] for row in rows
            if math.isfinite(row["cv_all_rows_time_e2e_op_s"])
        ]
        energy_cvs = [
            row["cv_all_rows_device_energy_op_j"] for row in rows
            if math.isfinite(row["cv_all_rows_device_energy_op_j"])
        ]
        report.append(
            f"| {spec.label} | {len(rows)} | "
            f"{format_float(median(time_cvs) * 100, 4)} % | "
            f"{format_float(max(time_cvs) * 100 if time_cvs else math.nan, 4)} % | "
            f"{format_float(median(energy_cvs) * 100, 4)} % | "
            f"{format_float(max(energy_cvs) * 100 if energy_cvs else math.nan, 4)} % |"
        )

    report += [
        "",
        "## Temperatur- und Takttelemetrie",
        "",
        "Die vollständige Zusammenfassung steht in `axpy_telemetry_report.md`; "
        "die Felder `clock_before_mhz`, `clock_after_mhz`, `temp_before_c` und "
        "`temp_after_c` bleiben in den normalisierten Daten erhalten.",
        "",
        "## GPU-Throttle-Masken",
        "",
        "Die konkreten NVML-Bitmasken bleiben in den normalisierten Zeilen erhalten "
        "und werden nicht auf ein bloßes Ja/Nein reduziert.",
        "",
        "| Plattform | Maske | Dekodierung | Zeilen |",
        "|---|---|---|---:|",
    ]
    for item in throttle_summary:
        if item["platform"] in {"3090", "5060ti"}:
            report.append(
                f"| {item['platform_label']} | `{item['throttle_reasons_hex']}` | "
                f"{item['throttle_labels']} | {item['rows']} |"
            )
    report += [
        "",
        "## Wissenschaftliche Grenzen",
        "",
        "- Cross-device-Energie ist eine Device-Domain-Größe: CPU-Package gegen "
        "GPU-Board. Bei AXPY fehlt auf der CPU-Seite insbesondere externer "
        "DDR4-Verbrauch, während GPU-VRAM im Board-Zähler enthalten ist.",
        "- `logical_bytes_per_op=12N` ist ein semantischer Anker, kein gemessener "
        "physischer Speicherverkehr.",
        "- Fünf Sessions erlauben robuste deskriptive Aussagen, aber die "
        "Bootstrap-Intervalle ersetzen keine umfassende inferenzstatistische "
        "Modellierung.",
        "",
        "## Meldungen",
        "",
        f"**FAIL: {fail_count}; WARN: {warn_count}**",
        "",
    ]
    if issues:
        for issue in issues[:50]:
            location = f"{issue.file}:{issue.row}" if issue.file else "Kampagne"
            report.append(
                f"- **{issue.severity} [{issue.platform}] {location}:** "
                f"{issue.message}"
            )
        if len(issues) > 50:
            report.append(f"- … {len(issues) - 50} weitere Meldungen in CSV.")
    else:
        report.append("- Keine Validierungsprobleme gefunden.")

    (output / "axpy_validation_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )

    if fail_count == 0:
        write_json(
            output / "axpy_validation_complete.json",
            {
                "status": status,
                "accepted_rows": len(all_rows),
                "rejected_rows": len(rejected_rows),
                "campaigns": {item["platform"]: item["campaign_id"] for item in campaign_manifest},
            },
        )

    print("=" * 76)
    print(f"AXPY VALIDIERUNG: {status}")
    for item in campaign_manifest:
        print(
            f"{item['platform']:>7}: {item['campaign_id']} "
            f"({platform_counts[item['platform']]} Zeilen)"
        )
    print(f"Konfigurationssummen: {len(config_rows)}")
    print(f"Ausgabe: {output}")
    print("=" * 76)
    return 1 if fail_count else 0


if __name__ == "__main__":
    raise SystemExit(main())

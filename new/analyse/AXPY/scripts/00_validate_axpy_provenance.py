#!/usr/bin/env python3
"""Prüft AXPY-Provenienz und erzeugt einen expliziten Kampagnen-Lock.

Harte Fehler:
- unvollständige/fehlende eingefrorene Kampagnen,
- FATAL/Checksum-Fehler in offiziellen Session-Logs,
- Widersprüche in vorhandenen Session-CSV-/Log-Hashes,
- falsche GPU-Gerätebindung,
- nicht byte-identische aktuelle GPU-Sources.

Dokumentierte Warnungen:
- fehlende historische Manifeste,
- fehlende oder nur indirekt belegbare Quickchecks,
- aktuelle Source/Runner/Binary weichen vom historischen Manifest ab.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

from axpy_analysis_common import (
    PLATFORMS,
    SIZES,
    parse_key_value_file,
    select_campaign,
    sha256_file,
    valid_sha256,
    write_csv,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AXPY-Provenienzprüfung")
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sessions", type=int, default=5)
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--intel-dir", type=Path)
    parser.add_argument("--amd-dir", type=Path)
    parser.add_argument("--gpu3090-dir", type=Path)
    parser.add_argument("--gpu5060ti-dir", type=Path)
    parser.add_argument("--intel-campaign")
    parser.add_argument("--amd-campaign")
    parser.add_argument("--gpu3090-campaign")
    parser.add_argument("--gpu5060ti-campaign")
    return parser.parse_args()


def inspect_quickcheck(path: Path, kind: str, expected_rows: int = 6) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    result_lines = [line for line in lines if line.startswith("[AXPY]")]
    pass_marker = "[quickcheck] PASS" in text
    anti_pass = re.search(r"\[ANTI_COLLAPSE\].*gate=PASS", text) is not None
    fatal = "FATAL:" in text
    checksum_failure = (
        "checksum=FAIL" in text
        or "checksum_ok=f" in text
        or "checksum1=FAIL" in text
        or "checksum2=FAIL" in text
    )
    all_result_checksums_ok = bool(result_lines) and all(
        "checksum=OK" in line for line in result_lines
    )
    enough_rows = len(result_lines) >= expected_rows
    source_match = re.search(r"source_sha256=([0-9a-fA-F]{64})", text)

    # GPU-Runner schreiben den abschließenden Shell-PASS-Marker nicht zwingend
    # in das Benchmark-Log. Daher genügt der direkte Messnachweis.
    direct_evidence = (
        enough_rows
        and all_result_checksums_ok
        and not fatal
        and not checksum_failure
        and anti_pass
    )
    if kind == "gpu":
        # Fail-closed: Ein PASS-Marker darf FATAL/Checksumfehler niemals überstimmen.
        marker_evidence = (
            pass_marker
            and anti_pass
            and enough_rows
            and all_result_checksums_ok
            and not fatal
            and not checksum_failure
        )
        qualifies = marker_evidence or direct_evidence
    else:
        marker_evidence = (
            pass_marker and not fatal and not checksum_failure
        )
        qualifies = marker_evidence or direct_evidence

    basis = (
        "shell_pass_marker"
        if marker_evidence
        else "measurement_and_anti_collapse_evidence"
        if direct_evidence
        else "insufficient_or_failed"
    )
    return {
        "path": path,
        "pass_marker": pass_marker,
        "anti_pass": anti_pass,
        "fatal": fatal,
        "checksum_failure": checksum_failure,
        "result_rows": len(result_lines),
        "all_result_checksums_ok": all_result_checksums_ok,
        "source_sha256": source_match.group(1).lower() if source_match else "",
        "qualifies": qualifies,
        "qualification_basis": basis,
    }


def select_quickcheck(directory: Path, before_mtime: float, kind: str):
    candidates = sorted(
        directory.glob("*quickcheck.log"),
        key=lambda path: path.stat().st_mtime,
    )
    if not candidates:
        return None, []
    before = [
        path for path in candidates
        if path.stat().st_mtime <= before_mtime + 1.0
    ]
    pool = before or candidates
    inspected = [inspect_quickcheck(path, kind) for path in pool]
    passed = [item for item in inspected if item["qualifies"]]
    return (passed[-1] if passed else None), inspected


def platform_paths(repo: Path, key: str) -> tuple[Path, Path, Path]:
    if key == "intel":
        base = repo / "new" / "INTEL"
        return (
            base / "scripts" / "AXPY" / "main_axpy_intel.cpp",
            base / "scripts" / "02_run_CPU_Intel_AXPY_only.sh",
            base / "scripts" / "AXPY" / ".build" / "main_axpy_intel",
        )
    if key == "amd":
        base = repo / "new" / "AMD"
        return (
            base / "scripts" / "AXPY" / "main_axpy_amd.cpp",
            base / "scripts" / "02_run_CPU_AMD_AXPY_only.sh",
            base / "scripts" / "AXPY" / ".build" / "main_axpy_amd",
        )
    if key == "3090":
        base = repo / "new" / "3090"
        return (
            base / "scripts" / "AXPY" / "main_axpy.cu",
            base / "02_run_GPU_3090_AXPY_only.sh",
            base / "scripts" / "AXPY" / ".build" / "main_axpy",
        )
    base = repo / "new" / "5060ti"
    return (
        base / "scripts" / "AXPY" / "main_axpy.cu",
        base / "scripts" / "02_run_GPU_5060ti_AXPY_only.sh",
        base / "scripts" / "AXPY" / ".build" / "main_axpy",
    )


def current_hash_status(
    path: Path,
    manifest_hash: str,
    *,
    platform: str,
    scope: str,
    issue,
) -> tuple[str, str]:
    if not path.exists():
        return "not_available", ""
    actual = sha256_file(path)
    if not manifest_hash:
        return "current_hash_only", actual
    if not valid_sha256(manifest_hash):
        return "manifest_hash_invalid", actual
    if actual == manifest_hash.lower():
        return "verified_against_current_file", actual
    issue(
        "WARN",
        platform,
        scope,
        f"Aktuelle Datei {path} weicht vom historischen Manifest-Hash ab; "
        "die archivierten Kampagnendaten werden dadurch nicht nachträglich ungültig.",
    )
    return "current_file_differs_from_manifest", actual


def main() -> int:
    args = parse_args()
    repo = args.repo.expanduser().resolve()
    output = args.output.expanduser().resolve()
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
    issues: list[dict] = []
    records: list[dict] = []
    campaign_lock: list[dict] = []
    gpu_source_hashes: dict[str, str] = {}

    def issue(severity: str, platform: str, scope: str, message: str) -> None:
        issues.append(
            {
                "severity": severity,
                "platform": platform,
                "scope": scope,
                "message": message,
            }
        )

    for spec in PLATFORMS:
        directory = (
            (repo / overrides[spec.key]).resolve()
            if overrides[spec.key]
            else (repo / spec.relative_dir).resolve()
        )
        requested_campaign = campaign_overrides[spec.key]
        try:
            prefix, files, complete = select_campaign(
                directory, args.sessions, requested_campaign
            )
        except FileNotFoundError as exc:
            issue("FAIL", spec.key, "campaign", str(exc))
            continue
        if not complete:
            issue("FAIL", spec.key, "campaign", f"Kampagne {prefix} ist unvollständig.")

        expected_rows = len(SIZES) * len(spec.threads) * args.reps
        manifest_path = directory / f"{prefix}_manifest.txt"
        manifest = parse_key_value_file(manifest_path) if manifest_path.exists() else {}
        manifest_parsed = manifest_path.exists()

        if not manifest_path.exists():
            issue("WARN", spec.key, "manifest", f"Manifest fehlt: {manifest_path.name}")
        else:
            for key in ("source_sha256", "runner_sha256", "binary_sha256"):
                if key in manifest and not valid_sha256(manifest[key]):
                    issue("FAIL", spec.key, "manifest", f"{key} ist kein SHA-256.")
            if manifest.get("workload") != "AXPY":
                issue("FAIL", spec.key, "manifest", "workload ist nicht AXPY.")
            if manifest.get("sessions") not in (None, str(args.sessions)):
                issue("FAIL", spec.key, "manifest", "sessions stimmt nicht.")
            if manifest.get("repetitions") not in (None, str(args.reps)):
                issue("FAIL", spec.key, "manifest", "repetitions stimmt nicht.")
            if (
                spec.kind == "gpu"
                and spec.device_token.lower()
                not in manifest.get("expected_gpu", "").lower()
            ):
                issue("FAIL", spec.key, "manifest", "expected_gpu passt nicht zur Plattform.")

        source_path, runner_path, binary_path = platform_paths(repo, spec.key)
        source_status, direct_source_hash = current_hash_status(
            source_path,
            manifest.get("source_sha256", ""),
            platform=spec.key,
            scope="source",
            issue=issue,
        )
        runner_status, direct_runner_hash = current_hash_status(
            runner_path,
            manifest.get("runner_sha256", ""),
            platform=spec.key,
            scope="runner",
            issue=issue,
        )
        binary_status, direct_binary_hash = current_hash_status(
            binary_path,
            manifest.get("binary_sha256", ""),
            platform=spec.key,
            scope="binary",
            issue=issue,
        )
        if spec.kind == "gpu" and direct_source_hash:
            gpu_source_hashes[spec.key] = direct_source_hash

        log_verified = 0
        csv_hash_verified = 0
        log_hash_verified = 0
        locked_files = []
        recalibration_events = 0
        below_retry_events = 0

        for session_no, csv_path in files:
            session_id = f"{prefix}_session{session_no}"
            log_path = csv_path.with_suffix(".log")
            csv_hash = sha256_file(csv_path)
            log_hash = sha256_file(log_path) if log_path.exists() else ""

            if not log_path.exists():
                issue("WARN", spec.key, session_id, "Session-Log fehlt.")
            else:
                text = log_path.read_text(encoding="utf-8", errors="replace")
                log_lines = text.splitlines()
                result_lines = [line for line in log_lines if line.startswith("[AXPY]")]
                recalibration_events += sum("[RECALIBRATION]" in line for line in log_lines)
                below_retry_events += sum(
                    ("action=retry_same_repetition" in line)
                    or ("[RECALIBRATION]" in line and "below" in line.lower())
                    for line in log_lines
                )
                if "FATAL:" in text or "checksum=FAIL" in text:
                    issue("FAIL", spec.key, session_id, "Session-Log enthält FATAL oder checksum=FAIL.")
                if session_id not in text:
                    issue("WARN", spec.key, session_id, "Session-ID fehlt im Logtext.")
                if len(result_lines) != expected_rows:
                    issue(
                        "WARN", spec.key, session_id,
                        f"Log enthält {len(result_lines)} statt {expected_rows} [AXPY]-Zeilen.",
                    )
                if result_lines and any("checksum=OK" not in line for line in result_lines):
                    issue("FAIL", spec.key, session_id, "Nicht jede [AXPY]-Zeile enthält checksum=OK.")
                elif len(result_lines) == expected_rows:
                    log_verified += 1

            if manifest:
                expected_id = manifest.get(f"session_{session_no}_id")
                if expected_id is not None and expected_id != session_id:
                    issue("FAIL", spec.key, session_id, "Session-ID im Manifest stimmt nicht.")
                expected_csv_hash = manifest.get(f"session_{session_no}_csv_sha256")
                if expected_csv_hash:
                    if not valid_sha256(expected_csv_hash) or csv_hash != expected_csv_hash.lower():
                        issue("FAIL", spec.key, session_id, "CSV-Hash stimmt nicht mit Manifest überein.")
                    else:
                        csv_hash_verified += 1
                expected_log_hash = manifest.get(f"session_{session_no}_log_sha256")
                if expected_log_hash and log_path.exists():
                    if not valid_sha256(expected_log_hash) or log_hash != expected_log_hash.lower():
                        issue("FAIL", spec.key, session_id, "Log-Hash stimmt nicht mit Manifest überein.")
                    else:
                        log_hash_verified += 1

            locked_files.append(
                {
                    "session": session_no,
                    "session_id": session_id,
                    "csv": str(csv_path),
                    "csv_sha256": csv_hash,
                    "log": str(log_path),
                    "log_exists": log_path.exists(),
                    "log_sha256": log_hash,
                }
            )

        first_mtime = min(path.stat().st_mtime for _, path in files)
        selected_quick, quick_candidates = select_quickcheck(directory, first_mtime, spec.kind)
        quick_status = "missing"
        quick_basis = "none"
        quick_source_hash = ""
        selected_quick_path = ""
        if not quick_candidates:
            issue(
                "WARN", spec.key, "quickcheck",
                "Kein Quickcheck-Log gefunden; Kampagne bleibt analysierbar, Provenienzstatus ist partial.",
            )
        elif selected_quick is None:
            names = ", ".join(item["path"].name for item in quick_candidates[-3:])
            issue(
                "WARN", spec.key, "quickcheck",
                "Kein bestandener Quickcheck vor Kampagnenbeginn gefunden. "
                f"Kandidaten: {names}. Dies ist kein Fehler der offiziellen Session-Daten.",
            )
            quick_status = "historical_failure"
        else:
            selected_quick_path = str(selected_quick["path"])
            quick_source_hash = selected_quick["source_sha256"]
            quick_status = "pass"
            quick_basis = selected_quick["qualification_basis"]
            if (
                manifest.get("source_sha256")
                and quick_source_hash
                and quick_source_hash != manifest["source_sha256"].lower()
            ):
                issue("WARN", spec.key, "quickcheck", "Quickcheck-Sourcehash weicht vom Kampagnenmanifest ab.")

        if recalibration_events or below_retry_events:
            issue(
                "WARN", spec.key, "recalibration",
                f"Offizielle Logs enthalten recalibration_events={recalibration_events}, "
                f"below_retry_events={below_retry_events}; im Paper dokumentieren.",
            )

        selected_quick_hash = (
            sha256_file(selected_quick["path"]) if selected_quick is not None else ""
        )
        local_fail = any(item["severity"] == "FAIL" and item["platform"] == spec.key for item in issues)
        partial = (
            not manifest_path.exists()
            or csv_hash_verified < len(files)
            or log_hash_verified < len(files)
            or quick_status != "pass"
        )
        provenance_status = "failed" if local_fail else ("partial" if partial else "verified")

        records.append(
            {
                "platform": spec.key,
                "platform_label": spec.label,
                "campaign_id": prefix,
                "selection_mode": "locked" if requested_campaign else "latest_complete",
                "manifest": str(manifest_path),
                "manifest_parsed": manifest_parsed,
                "session_csv_hashes_verified": csv_hash_verified,
                "session_log_hashes_verified": log_hash_verified,
                "session_logs_content_verified": log_verified,
                "sessions": len(files),
                "quickcheck_log": selected_quick_path,
                "quickcheck_candidates": len(quick_candidates),
                "quickcheck_status": quick_status,
                "quickcheck_qualification_basis": quick_basis,
                "quickcheck_sha256": selected_quick_hash,
                "recalibration_events": recalibration_events,
                "below_retry_events": below_retry_events,
                "source_path": str(source_path),
                "source_hash_status": source_status,
                "source_sha256_current": direct_source_hash,
                "source_sha256_manifest": manifest.get("source_sha256", ""),
                "runner_path": str(runner_path),
                "runner_hash_status": runner_status,
                "runner_sha256_current": direct_runner_hash,
                "runner_sha256_manifest": manifest.get("runner_sha256", ""),
                "binary_path": str(binary_path),
                "binary_hash_status": binary_status,
                "binary_sha256_current": direct_binary_hash,
                "binary_sha256_manifest": manifest.get("binary_sha256", ""),
                "source_sha256_quickcheck": quick_source_hash,
                "provenance_status": provenance_status,
            }
        )
        campaign_lock.append(
            {
                "platform": spec.key,
                "platform_label": spec.label,
                "campaign_id": prefix,
                "directory": str(directory),
                "manifest": str(manifest_path),
                "manifest_exists": manifest_path.exists(),
                "manifest_sha256": sha256_file(manifest_path) if manifest_path.exists() else "",
                "quickcheck": {
                    "status": quick_status,
                    "qualification_basis": quick_basis,
                    "path": selected_quick_path,
                    "sha256": selected_quick_hash,
                },
                "recalibration_events": recalibration_events,
                "below_retry_events": below_retry_events,
                "files": locked_files,
            }
        )

    if set(gpu_source_hashes) == {"3090", "5060ti"}:
        if gpu_source_hashes["3090"] != gpu_source_hashes["5060ti"]:
            issue("FAIL", "gpu_pair", "source_identity", "GPU-Sources sind nicht byte-identisch.")
    else:
        issue("WARN", "gpu_pair", "source_identity", "GPU-Sourceidentität konnte nicht vollständig bestätigt werden.")

    write_csv(output / "axpy_provenance.csv", records, list(records[0].keys()) if records else [])
    write_csv(
        output / "axpy_provenance_issues.csv",
        issues,
        ["severity", "platform", "scope", "message"],
    )
    write_json(
        output / "axpy_campaign_lock.json",
        {
            "schema": "axpy-campaign-lock-v2",
            "sessions_expected": args.sessions,
            "repetitions_expected": args.reps,
            "campaigns": campaign_lock,
        },
    )
    recalibration_rows = [
        {
            "platform": item["platform"],
            "platform_label": item["platform_label"],
            "campaign_id": item["campaign_id"],
            "recalibration_events": item["recalibration_events"],
            "below_retry_events": item["below_retry_events"],
        }
        for item in records
    ]
    write_csv(
        output / "axpy_recalibration_summary.csv",
        recalibration_rows,
        ["platform", "platform_label", "campaign_id", "recalibration_events", "below_retry_events"],
    )

    fails = sum(item["severity"] == "FAIL" for item in issues)
    warns = sum(item["severity"] == "WARN" for item in issues)
    status = "FAIL" if fails else ("PASS MIT WARNUNGEN" if warns else "PASS")

    report = [
        "# AXPY – Provenienzprüfung",
        "",
        f"**Status: {status}**",
        "",
        "| Plattform | Kampagne | Auswahl | Manifest | CSV-Hashes | Log-Hashes | Loginhalt | Quickcheck | Status |",
        "|---|---|---|---|---:|---:|---:|---|---|",
    ]
    for record in records:
        report.append(
            f"| {record['platform_label']} | `{record['campaign_id']}` | "
            f"{record['selection_mode']} | {'ja' if record['manifest_parsed'] else 'nein'} | "
            f"{record['session_csv_hashes_verified']}/{record['sessions']} | "
            f"{record['session_log_hashes_verified']}/{record['sessions']} | "
            f"{record['session_logs_content_verified']}/{record['sessions']} | "
            f"{record['quickcheck_status']} ({record['quickcheck_qualification_basis']}) | "
            f"{record['provenance_status']} |"
        )
    report += [
        "",
        "`manifest_parsed` bedeutet nur, dass ein Manifest gelesen wurde. "
        "Source-, Runner- und Binary-Status werden separat ausgewiesen; dadurch wird "
        "kein nicht geprüfter Hash als verifiziert bezeichnet.",
        "",
        "GPU-Quickchecks dürfen durch Messnachweis statt Shell-PASS-Marker bestätigt werden: "
        "mindestens sechs AXPY-Zeilen, alle Checksummen OK, Anti-Collapse PASS und kein FATAL.",
        "",
        "## Rekalibrierungsereignisse in offiziellen Logs",
        "",
        "| Plattform | Rekalibrierungen | Below-Retries |",
        "|---|---:|---:|",
    ]
    for record in records:
        report.append(
            f"| {record['platform_label']} | {record['recalibration_events']} | "
            f"{record['below_retry_events']} |"
        )
    report += [
        "",
        f"FAIL: {fails}; WARN: {warns}",
    ]
    for item in issues[:80]:
        report.append(f"- **{item['severity']} [{item['platform']}/{item['scope']}]:** {item['message']}")
    if len(issues) > 80:
        report.append(f"- … {len(issues) - 80} weitere Meldungen in `axpy_provenance_issues.csv`.")
    (output / "axpy_provenance_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    if not fails:
        write_json(
            output / "axpy_provenance_complete.json",
            {"status": status, "failures": fails, "warnings": warns, "records": len(records)},
        )
    print(f"AXPY PROVENIENZ: {status}; FAIL={fails}; WARN={warns}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())

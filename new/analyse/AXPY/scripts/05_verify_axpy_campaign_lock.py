#!/usr/bin/env python3
"""Verifiziert einen AXPY-Campaign-Lock fail-closed und erzeugt Runner-Argumente."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from axpy_analysis_common import PLATFORMS, sha256_file, valid_sha256

EXPECTED_PLATFORMS = ("intel", "amd", "3090", "5060ti")
FLAGS = {
    "intel": ("--intel-dir", "--intel-campaign"),
    "amd": ("--amd-dir", "--amd-campaign"),
    "3090": ("--gpu3090-dir", "--gpu3090-campaign"),
    "5060ti": ("--gpu5060ti-dir", "--gpu5060ti-campaign"),
}
RELATIVE_DIRS = {spec.key: spec.relative_dir for spec in PLATFORMS}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AXPY-Campaign-Lock bytegenau prüfen")
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--sessions", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--args-file", type=Path, required=True)
    return parser.parse_args()


def resolve_locked_file(path_text: str, repo: Path, platform: str) -> Path:
    path = Path(path_text).expanduser()
    if path.is_file():
        return path.resolve()
    fallback = (repo / RELATIVE_DIRS[platform] / path.name).resolve()
    if fallback.is_file():
        return fallback
    raise ValueError(f"Gesperrte Datei fehlt: {path} (Fallback ebenfalls fehlt: {fallback})")


def resolve_locked_directory(path_text: str, repo: Path, platform: str) -> Path:
    path = Path(path_text).expanduser()
    if path.is_dir():
        return path.resolve()
    fallback = (repo / RELATIVE_DIRS[platform]).resolve()
    if fallback.is_dir():
        return fallback
    raise ValueError(f"Gesperrtes Kampagnenverzeichnis fehlt: {path}; Fallback: {fallback}")


def require_hash(path: Path, expected: object, label: str) -> None:
    if not valid_sha256(expected):
        raise ValueError(f"{label}: ungültiger SHA-256 im Lock: {expected!r}")
    actual = sha256_file(path)
    if actual != str(expected).lower():
        raise ValueError(
            f"{label}: SHA-256-Abweichung für {path}: expected={expected}, actual={actual}"
        )


def validate_lock(lock: dict, repo: Path, sessions: int, repetitions: int) -> list[str]:
    if lock.get("schema") != "axpy-campaign-lock-v2":
        raise ValueError(
            "Lock-Schema muss axpy-campaign-lock-v2 sein. "
            "Einen alten v1-Lock einmal ohne AXPY_CAMPAIGN_LOCK neu erzeugen."
        )
    if lock.get("sessions_expected") != sessions:
        raise ValueError(
            f"sessions_expected={lock.get('sessions_expected')!r}, erwartet {sessions}."
        )
    if lock.get("repetitions_expected") != repetitions:
        raise ValueError(
            f"repetitions_expected={lock.get('repetitions_expected')!r}, erwartet {repetitions}."
        )

    campaigns = lock.get("campaigns")
    if not isinstance(campaigns, list):
        raise ValueError("campaigns muss eine Liste sein.")
    keys = [item.get("platform") for item in campaigns if isinstance(item, dict)]
    if len(keys) != len(set(keys)):
        raise ValueError(f"Doppelte Plattformen im Lock: {keys}")
    if set(keys) != set(EXPECTED_PLATFORMS) or len(campaigns) != 4:
        raise ValueError(
            f"Lock muss exakt {list(EXPECTED_PLATFORMS)} enthalten; gefunden: {keys}"
        )

    by_key = {item["platform"]: item for item in campaigns}
    runner_args: list[str] = []
    for platform in EXPECTED_PLATFORMS:
        item = by_key[platform]
        campaign_id = item.get("campaign_id")
        if not isinstance(campaign_id, str) or not campaign_id.strip():
            raise ValueError(f"{platform}: campaign_id fehlt.")
        directory = resolve_locked_directory(str(item.get("directory", "")), repo, platform)

        files = item.get("files")
        if not isinstance(files, list) or len(files) != sessions:
            raise ValueError(f"{platform}: exakt {sessions} Sessiondateien erforderlich.")
        session_numbers = [entry.get("session") for entry in files if isinstance(entry, dict)]
        if sorted(session_numbers) != list(range(1, sessions + 1)):
            raise ValueError(f"{platform}: Sessionnummern inkonsistent: {session_numbers}")

        for entry in files:
            session = int(entry["session"])
            expected_id = f"{campaign_id}_session{session}"
            if entry.get("session_id") != expected_id:
                raise ValueError(
                    f"{platform}: session_id={entry.get('session_id')!r}, erwartet {expected_id!r}."
                )
            csv_path = resolve_locked_file(str(entry.get("csv", "")), repo, platform)
            log_path = resolve_locked_file(str(entry.get("log", "")), repo, platform)
            if csv_path.parent != directory or log_path.parent != directory:
                raise ValueError(f"{platform}/{expected_id}: Dateien liegen nicht im gesperrten Verzeichnis.")
            if csv_path.name != f"{expected_id}.csv" or log_path.name != f"{expected_id}.log":
                raise ValueError(f"{platform}/{expected_id}: unerwartete Dateinamen.")
            require_hash(csv_path, entry.get("csv_sha256"), f"{platform}/{expected_id}/csv")
            require_hash(log_path, entry.get("log_sha256"), f"{platform}/{expected_id}/log")

        manifest_exists = bool(item.get("manifest_exists"))
        manifest_text = str(item.get("manifest", ""))
        if manifest_exists:
            manifest_path = resolve_locked_file(manifest_text, repo, platform)
            require_hash(manifest_path, item.get("manifest_sha256"), f"{platform}/manifest")

        quick = item.get("quickcheck")
        if not isinstance(quick, dict):
            raise ValueError(f"{platform}: quickcheck-Objekt fehlt im v2-Lock.")
        if quick.get("status") == "pass":
            quick_path = resolve_locked_file(str(quick.get("path", "")), repo, platform)
            require_hash(quick_path, quick.get("sha256"), f"{platform}/quickcheck")
        elif quick.get("status") not in {"missing", "historical_failure"}:
            raise ValueError(f"{platform}: ungültiger quickcheck.status={quick.get('status')!r}.")

        dir_flag, campaign_flag = FLAGS[platform]
        runner_args.extend([dir_flag, str(directory), campaign_flag, campaign_id])

    return runner_args


def main() -> int:
    args = parse_args()
    try:
        lock = json.loads(args.lock.read_text(encoding="utf-8"))
        runner_args = validate_lock(
            lock,
            args.repo.expanduser().resolve(),
            args.sessions,
            args.repetitions,
        )
        args.args_file.parent.mkdir(parents=True, exist_ok=True)
        args.args_file.write_text("\n".join(runner_args) + "\n", encoding="utf-8")
    except (OSError, json.JSONDecodeError, TypeError, ValueError, KeyError) as exc:
        print(f"FATAL: Campaign-Lock-Verifikation fehlgeschlagen: {exc}", file=sys.stderr)
        return 2
    print(f"Campaign-Lock bytegenau verifiziert: {args.lock}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

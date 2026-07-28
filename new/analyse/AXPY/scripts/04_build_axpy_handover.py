#!/usr/bin/env python3
"""Erzeugt ein selbstbeschreibendes AXPY-Analyse-Handover-Archiv."""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import zipfile
from pathlib import Path

from axpy_analysis_common import sha256_file, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AXPY-Handover-Archiv erstellen")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scripts-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output_dir.resolve()
    scripts = args.scripts_dir.resolve()
    lock_path = output / "axpy_campaign_lock.json"
    if not lock_path.is_file():
        raise SystemExit(f"Campaign-Lock fehlt: {lock_path}")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))

    archive_path = output / "AXPY_ANALYSIS_HANDOVER.zip"
    with tempfile.TemporaryDirectory(prefix="axpy_handover_") as temp_name:
        temp = Path(temp_name) / "AXPY_ANALYSIS_HANDOVER"
        results_dir = temp / "results"
        scripts_dir = temp / "analysis_scripts"
        manifests_dir = temp / "campaign_manifests"
        quickchecks_dir = temp / "quickcheck_logs"
        results_dir.mkdir(parents=True)
        scripts_dir.mkdir(parents=True)
        manifests_dir.mkdir(parents=True)
        quickchecks_dir.mkdir(parents=True)

        for path in sorted(output.iterdir()):
            if not path.is_file() or path.name == archive_path.name:
                continue
            shutil.copy2(path, results_dir / path.name)

        for path in sorted(scripts.iterdir()):
            if path.is_file() and path.suffix in {".py", ".sh", ".md"}:
                shutil.copy2(path, scripts_dir / path.name)

        copied_manifests = []
        copied_quickchecks = []
        for campaign in lock.get("campaigns", []):
            manifest = Path(campaign.get("manifest", ""))
            if manifest.is_file():
                target = manifests_dir / f"{campaign['platform']}__{manifest.name}"
                shutil.copy2(manifest, target)
                copied_manifests.append(str(target.relative_to(temp)))
            quick = campaign.get("quickcheck", {})
            quick_path = Path(quick.get("path", ""))
            if quick.get("status") == "pass" and quick_path.is_file():
                target = quickchecks_dir / f"{campaign['platform']}__{quick_path.name}"
                shutil.copy2(quick_path, target)
                copied_quickchecks.append(str(target.relative_to(temp)))

        readme = """# AXPY analysis handover

Dieses Archiv friert die ausgewählten Kampagnen, die Rohdatei-Hashes,
die Analyse-Skripte und sämtliche abgeleiteten Ergebnisse zusammen ein.
Die großen Roh-CSVs und Session-Logs werden nicht dupliziert; ihre Pfade und
SHA-256-Prüfsummen stehen in `results/axpy_campaign_lock.json`. Die ausgewählten
Quickcheck-Logs werden dagegen direkt unter `quickcheck_logs/` mitgeführt.

Für eine vollständige Wiederholung müssen die dort referenzierten Rohdateien
am Repository-Standort vorhanden sein. Dann kann der Lock über
`AXPY_CAMPAIGN_LOCK=/pfad/axpy_campaign_lock.json` an den Runner übergeben werden.
"""
        (temp / "README.md").write_text(readme, encoding="utf-8")

        files = []
        for path in sorted(temp.rglob("*")):
            if path.is_file():
                files.append(
                    {
                        "path": str(path.relative_to(temp)),
                        "sha256": sha256_file(path),
                        "bytes": path.stat().st_size,
                    }
                )
        write_json(
            temp / "BUNDLE_MANIFEST.json",
            {
                "schema": "axpy-analysis-handover-v2",
                "files": files,
                "copied_campaign_manifests": copied_manifests,
                "copied_quickcheck_logs": copied_quickchecks,
            },
        )

        if archive_path.exists():
            archive_path.unlink()
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(temp.rglob("*")):
                if path.is_file():
                    archive.write(path, Path(temp.name) / path.relative_to(temp))

    write_json(
        output / "axpy_handover_complete.json",
        {
            "status": "PASS",
            "archive": archive_path.name,
            "sha256": sha256_file(archive_path),
            "bytes": archive_path.stat().st_size,
        },
    )
    print(f"AXPY HANDOVER: PASS; {archive_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

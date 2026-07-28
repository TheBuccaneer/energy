#!/usr/bin/env python3
"""Regressionstests für die v5-Audit-Patches."""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))


def load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPT_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


common = load("common_v5", "axpy_analysis_common.py")
prov = load("prov_v5", "00_validate_axpy_provenance.py")
robust = load("robust_v5", "03_assess_axpy_robustness.py")
lockmod = load("lock_v5", "05_verify_axpy_campaign_lock.py")
validate = load("validate_v5_1", "01_validate_and_aggregate_axpy.py")


def test_gpu_quickcheck_direct_evidence() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "quickcheck.log"
        path.write_text(
            "\n".join(
                ["[AXPY] N=1000000 checksum=OK"] * 6
                + ["[ANTI_COLLAPSE] ratio=2.0 gate=PASS"]
            )
            + "\n",
            encoding="utf-8",
        )
        result = prov.inspect_quickcheck(path, "gpu")
        assert result["qualifies"]
        assert result["qualification_basis"] == "measurement_and_anti_collapse_evidence"


def test_gpu_quickcheck_pass_marker_cannot_override_failure() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "quickcheck.log"
        path.write_text(
            "\n".join(
                ["[quickcheck] PASS"]
                + ["[AXPY] N=1000000 checksum=OK"] * 6
                + ["[ANTI_COLLAPSE] ratio=2.0 gate=PASS", "FATAL: bad"]
            )
            + "\n",
            encoding="utf-8",
        )
        result = prov.inspect_quickcheck(path, "gpu")
        assert not result["qualifies"]
        assert result["qualification_basis"] == "insufficient_or_failed"


def make_lock_repo(base: Path) -> tuple[Path, dict]:
    campaigns = []
    for platform, relative in lockmod.RELATIVE_DIRS.items():
        directory = base / relative
        directory.mkdir(parents=True, exist_ok=True)
        campaign_id = f"axpy_{platform}_frozen"
        files = []
        for session in range(1, 6):
            session_id = f"{campaign_id}_session{session}"
            csv_path = directory / f"{session_id}.csv"
            log_path = directory / f"{session_id}.log"
            csv_path.write_text(f"csv {platform} {session}\n", encoding="utf-8")
            log_path.write_text(f"log {platform} {session}\n", encoding="utf-8")
            files.append(
                {
                    "session": session,
                    "session_id": session_id,
                    "csv": str(csv_path),
                    "csv_sha256": common.sha256_file(csv_path),
                    "log": str(log_path),
                    "log_exists": True,
                    "log_sha256": common.sha256_file(log_path),
                }
            )
        quick = directory / f"{campaign_id}_quickcheck.log"
        quick.write_text("quickcheck\n", encoding="utf-8")
        campaigns.append(
            {
                "platform": platform,
                "platform_label": platform,
                "campaign_id": campaign_id,
                "directory": str(directory),
                "manifest": str(directory / f"{campaign_id}_manifest.txt"),
                "manifest_exists": False,
                "manifest_sha256": "",
                "quickcheck": {
                    "status": "pass",
                    "qualification_basis": "test",
                    "path": str(quick),
                    "sha256": common.sha256_file(quick),
                },
                "recalibration_events": 0,
                "below_retry_events": 0,
                "files": files,
            }
        )
    return base, {
        "schema": "axpy-campaign-lock-v2",
        "sessions_expected": 5,
        "repetitions_expected": 10,
        "campaigns": campaigns,
    }


def test_lock_verification_is_byte_exact_and_fail_closed() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        repo, lock = make_lock_repo(Path(tmp))
        args = lockmod.validate_lock(lock, repo, 5, 10)
        assert len(args) == 16

        mutated = json.loads(json.dumps(lock))
        mutated["campaigns"][0]["files"][0]["csv_sha256"] = "0" * 64
        try:
            lockmod.validate_lock(mutated, repo, 5, 10)
        except ValueError:
            pass
        else:
            raise AssertionError("Manipulierter Rohdatenhash wurde akzeptiert")

        missing = json.loads(json.dumps(lock))
        missing["campaigns"] = missing["campaigns"][:-1]
        try:
            lockmod.validate_lock(missing, repo, 5, 10)
        except ValueError:
            pass
        else:
            raise AssertionError("Unvollständiger Lock wurde akzeptiert")


def test_in_range_requires_five_times_ten() -> None:
    rows = []
    for session in range(1, 6):
        repetitions = 2 if session == 4 else 10
        for _ in range(repetitions):
            rows.append(
                {
                    "platform": "amd",
                    "problem_size": "2000000",
                    "threads": "20",
                    "session_number": str(session),
                    "runtime_status": "in_range",
                    "time_e2e_op_s": "1.0",
                }
            )
    configs = robust.in_range_estimates(rows, "time_e2e_op_s", 10)
    assert len(configs) == 1
    assert configs[0]["sessions_covered"] == 5
    assert configs[0]["complete_sessions"] == 4
    assert robust.in_range_envelope(configs, 2_000_000, 5) == []


def test_fixed_vs_oracle_session_support() -> None:
    selected = {
        "a": {"platform": "a", "threads": "1"},
        "b": {"platform": "b", "threads": "-1"},
    }
    rows = [
        {"platform": "a", "problem_size": "1000000", "session_number": "1", "threads": "1", "median_time_e2e_op_s": "1.0"},
        {"platform": "a", "problem_size": "1000000", "session_number": "1", "threads": "2", "median_time_e2e_op_s": "1.1"},
        {"platform": "b", "problem_size": "1000000", "session_number": "1", "threads": "-1", "median_time_e2e_op_s": "1.2"},
        {"platform": "a", "problem_size": "1000000", "session_number": "2", "threads": "1", "median_time_e2e_op_s": "1.3"},
        {"platform": "a", "problem_size": "1000000", "session_number": "2", "threads": "2", "median_time_e2e_op_s": "0.8"},
        {"platform": "b", "problem_size": "1000000", "session_number": "2", "threads": "-1", "median_time_e2e_op_s": "1.0"},
    ]
    fixed, _, _ = robust.session_winner_support(
        rows, "time_e2e_op_s", 1_000_000, 0.0, selected=selected
    )
    oracle, _, _ = robust.session_winner_support(
        rows, "time_e2e_op_s", 1_000_000, 0.0, selected=None
    )
    assert fixed["a"] == 1 and fixed["b"] == 1
    assert oracle["a"] == 2 and oracle["b"] == 0




def test_cpu_throttle_sentinel_historical_blank() -> None:
    assert validate.cpu_throttle_sentinel_ok("")
    assert validate.cpu_throttle_sentinel_ok("-1")
    assert validate.cpu_throttle_sentinel_ok("-1.000000")
    assert not validate.cpu_throttle_sentinel_ok("0")
    assert not validate.cpu_throttle_sentinel_ok("nonsense")


def main() -> int:
    test_cpu_throttle_sentinel_historical_blank()
    test_gpu_quickcheck_direct_evidence()
    test_gpu_quickcheck_pass_marker_cannot_override_failure()
    test_lock_verification_is_byte_exact_and_fail_closed()
    test_in_range_requires_five_times_ten()
    test_fixed_vs_oracle_session_support()
    print("AXPY analysis v5 selftest: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

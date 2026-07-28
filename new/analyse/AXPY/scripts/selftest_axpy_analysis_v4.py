#!/usr/bin/env python3
"""Kleine Regressionstests für die v4-Audit-Patches."""
from __future__ import annotations

import importlib.util
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


common = load("common_v4", "axpy_analysis_common.py")
prov = load("prov_v4", "00_validate_axpy_provenance.py")
robust = load("robust_v4", "03_assess_axpy_robustness.py")


def test_quickcheck_without_shell_marker() -> None:
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


def test_explicit_campaign_lock_selection() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        for session in range(1, 6):
            (directory / f"frozen_campaign_session{session}.csv").write_text("x\n", encoding="utf-8")
        prefix, entries, complete = common.select_campaign(
            directory, 5, "frozen_campaign"
        )
        assert prefix == "frozen_campaign"
        assert complete
        assert len(entries) == 5


def test_fixed_vs_oracle_session_support() -> None:
    # Global selected configs: A/T1 und B/GPU. In Session 2 kann A mit T2
    # oracle gewinnen, während die feste T1-Konfiguration verliert.
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


def main() -> int:
    test_quickcheck_without_shell_marker()
    test_explicit_campaign_lock_selection()
    test_fixed_vs_oracle_session_support()
    print("AXPY analysis v4 selftest: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Kleiner Vorabcheck für Intel-/AMD-AXPY-CSV-Dateien (keine Endanalyse)."""

from __future__ import annotations

import argparse
import csv
import math
import re
import statistics as stats
import sys
from collections import Counter, defaultdict
from pathlib import Path

SIZES = {
    1_000_000, 2_000_000, 4_000_000, 8_000_000, 16_000_000,
    32_000_000, 64_000_000, 128_000_000, 256_000_000,
}
THREADS = {
    "Intel": {1, 2, 4, 8, 10, 16, 20},
    "AMD": {1, 2, 4, 8, 10, 16, 20, 32, 64},
}
REQUIRED = {
    "schema_version", "session_id", "sequence_index", "repetition", "workload",
    "implementation", "execution_mode", "device_name", "num_threads",
    "problem_size", "batches", "e2e_time_s", "kernel_time_s", "wall_time_s",
    "device_energy_j", "total_energy_j", "dram_energy_j", "energy_per_op_j",
    "time_per_op_ms_e2e", "flops_total", "logical_bytes_per_op", "avg_power_w",
    "runtime_status", "checksum_ok",
}
SESSION_RE = re.compile(r"^(.*)_session([1-9][0-9]*)$")


def args():
    p = argparse.ArgumentParser(
        description="Vorläufige Integritäts-, Plausibilitäts- und Vergleichsanalyse für CPU-AXPY."
    )
    p.add_argument("--repo", type=Path, default=Path("."))
    p.add_argument("--intel-dir", type=Path, default=Path("new/INTEL/runs/AXPY"))
    p.add_argument("--amd-dir", type=Path, default=Path("new/AMD/runs/AXPY"))
    p.add_argument("--report", type=Path,
                   default=Path("new/analyse/AXPY/axpy_cpu_precheck.md"))
    p.add_argument("--sessions", type=int, default=5)
    p.add_argument("--reps", type=int, default=10)
    return p.parse_args()


def truth(v):
    return str(v).strip().lower() in {"1", "t", "true", "yes", "ok"}


def cv(xs):
    return stats.stdev(xs) / stats.fmean(xs) if len(xs) > 1 and stats.fmean(xs) else math.nan


def pct(x):
    return "n/a" if not math.isfinite(x) else f"{100*x:.2f} %"


def ratio(x):
    return "n/a" if not math.isfinite(x) else f"{x:.3f}×"


def latest_campaign(directory: Path, expected_sessions: int):
    groups = defaultdict(list)
    for f in directory.glob("*.csv"):
        if "quickcheck" in f.name.lower():
            continue
        m = SESSION_RE.match(f.stem)
        if m:
            groups[m.group(1)].append((int(m.group(2)), f))
    if not groups:
        raise RuntimeError(f"Keine *_sessionN.csv in {directory}")

    wanted = set(range(1, expected_sessions + 1))
    candidates = []
    for prefix, entries in groups.items():
        sessions = {n for n, _ in entries}
        newest = max(f.stat().st_mtime for _, f in entries)
        candidates.append((sessions == wanted, len(sessions), newest, prefix, entries))
    complete, _, _, prefix, entries = max(candidates)
    return prefix, sorted(entries), complete


def load_platform(platform, directory, sessions, reps):
    prefix, files, complete = latest_campaign(directory, sessions)
    issues = []
    rows = []
    expected_rows_file = len(SIZES) * len(THREADS[platform]) * reps

    if not complete:
        issues.append(("FAIL", f"{platform}: ausgewählte Kampagne ist nicht vollständig."))

    for session_no, path in files:
        with path.open(newline="", encoding="utf-8") as f:
            rd = csv.DictReader(f)
            header = rd.fieldnames or []
            part = list(rd)

        missing = REQUIRED - set(header)
        if missing:
            issues.append(("FAIL", f"{path.name}: Pflichtspalten fehlen: {sorted(missing)}"))
            continue
        if len(header) != 45:
            issues.append(("WARN", f"{path.name}: {len(header)} statt 45 Spalten."))
        if len(part) != expected_rows_file:
            issues.append((
                "FAIL",
                f"{path.name}: {len(part)} statt {expected_rows_file} Zeilen."
            ))
        if not path.with_suffix(".log").exists():
            issues.append(("WARN", f"{path.name}: zugehörige Logdatei fehlt."))

        seen = Counter()
        seqs = []
        for line, r in enumerate(part, 2):
            where = f"{path.name}:{line}"
            try:
                n = int(r["problem_size"])
                t = int(r["num_threads"])
                rep = int(r["repetition"])
                b = int(r["batches"])
                seq = int(r["sequence_index"])
                e2e = float(r["e2e_time_s"])
                kernel = float(r["kernel_time_s"])
                wall = float(r["wall_time_s"])
                pkg = float(r["device_energy_j"])
                total = float(r["total_energy_j"])
                dram = float(r["dram_energy_j"])
                eop = float(r["energy_per_op_j"])
                top = float(r["time_per_op_ms_e2e"])
                flops = int(r["flops_total"])
                logical_bytes = int(r["logical_bytes_per_op"])
                power = float(r["avg_power_w"])
            except (KeyError, ValueError):
                issues.append(("FAIL", f"{where}: ungültiger Zahlenwert."))
                continue

            seqs.append(seq)
            seen[(n, t, rep)] += 1

            if r["session_id"] != path.stem:
                issues.append(("FAIL", f"{where}: session_id passt nicht zum Dateinamen."))
            if r["schema_version"] != "cpu-gpu-v2":
                issues.append(("FAIL", f"{where}: falsche schema_version."))
            if r["workload"] != "AXPY" or r["execution_mode"] != "cpu_native":
                issues.append(("FAIL", f"{where}: falscher Workload/Modus."))
            if r["implementation"] != "openmp_axpy_inplace_fp32":
                issues.append(("FAIL", f"{where}: unerwartete Implementierung."))
            if not truth(r["checksum_ok"]):
                issues.append(("FAIL", f"{where}: Checksumme fehlgeschlagen."))
            if n not in SIZES or t not in THREADS[platform] or not 1 <= rep <= reps:
                issues.append(("FAIL", f"{where}: unerwartete Konfiguration."))
            if b <= 0 or min(e2e, kernel, wall, pkg, total, eop, top, power) <= 0:
                issues.append(("FAIL", f"{where}: nichtpositiver Messwert."))
            if not (math.isclose(e2e, kernel, rel_tol=2e-9) and
                    math.isclose(e2e, wall, rel_tol=2e-9)):
                issues.append(("FAIL", f"{where}: e2e/kernel/wall nicht identisch."))

            expected_total = pkg if dram == -1 else pkg + dram
            checks = [
                math.isclose(total, expected_total, rel_tol=2e-9, abs_tol=1e-12),
                math.isclose(eop, pkg / b, rel_tol=2e-9, abs_tol=1e-12),
                math.isclose(top, 1000 * e2e / b, rel_tol=2e-9, abs_tol=1e-12),
                math.isclose(power, pkg / wall, rel_tol=2e-9, abs_tol=1e-12),
                flops == 2 * n * b,
                logical_bytes == 12 * n,
            ]
            if not all(checks):
                issues.append(("FAIL", f"{where}: mindestens ein Formelanker stimmt nicht."))

            expected_status = "below" if e2e < .75 else ("in_range" if e2e <= 1.25 else "above")
            if r["runtime_status"] != expected_status:
                issues.append(("FAIL", f"{where}: runtime_status stimmt rechnerisch nicht."))
            elif expected_status == "below":
                issues.append(("FAIL", f"{where}: offizielle Messung unter 0,75 s."))
            elif expected_status == "above":
                issues.append(("WARN", f"{where}: Messung über 1,25 s."))

            rows.append(r)

        if seqs != list(range(1, len(part) + 1)):
            issues.append(("FAIL", f"{path.name}: sequence_index nicht lückenlos."))
        if any(v != 1 for v in seen.values()):
            issues.append(("FAIL", f"{path.name}: doppelte Konfiguration/Repetition."))

        expected_keys = {
            (n, t, rep)
            for n in SIZES for t in THREADS[platform] for rep in range(1, reps + 1)
        }
        if set(seen) != expected_keys:
            issues.append(("FAIL", f"{path.name}: Konfigurationsraster unvollständig."))

    return {
        "platform": platform,
        "prefix": prefix,
        "files": [f for _, f in files],
        "rows": rows,
        "issues": issues,
        "devices": sorted({r["device_name"] for r in rows if r.get("device_name")}),
    }


def summarize(data):
    groups = defaultdict(list)
    for r in data["rows"]:
        groups[(int(r["problem_size"]), int(r["num_threads"]))].append(r)

    out = {}
    for key, rs in groups.items():
        time = [float(r["time_per_op_ms_e2e"]) for r in rs]
        pkg_energy = [float(r["energy_per_op_j"]) for r in rs]
        power = [float(r["avg_power_w"]) for r in rs]
        out[key] = {
            "count": len(rs),
            "time": stats.median(time),
            "energy": stats.median(pkg_energy),
            "power": stats.median(power),
            "cv_time": cv(time),
            "cv_energy": cv(pkg_energy),
        }
    return out


def winners(intel, amd):
    common = sorted(set(intel) & set(amd))
    tw, ew, dominance = Counter(), Counter(), Counter()
    tr, er = [], []

    for key in common:
        rt = amd[key]["time"] / intel[key]["time"]
        re = amd[key]["energy"] / intel[key]["energy"]
        tr.append(rt)
        er.append(re)

        tw["AMD" if rt < .98 else "Intel" if rt > 1.02 else "gleich"] += 1
        ew["AMD" if re < .98 else "Intel" if re > 1.02 else "gleich"] += 1

        if rt < .98 and re < .98:
            dominance["AMD dominiert"] += 1
        elif rt > 1.02 and re > 1.02:
            dominance["Intel dominiert"] += 1
        else:
            dominance["Trade-off/gleich"] += 1

    return {
        "common": common,
        "time_ratio": stats.median(tr) if tr else math.nan,
        "energy_ratio": stats.median(er) if er else math.nan,
        "time_winners": tw,
        "energy_winners": ew,
        "dominance": dominance,
    }


def stability(summary):
    t = [x["cv_time"] for x in summary.values()]
    e = [x["cv_energy"] for x in summary.values()]
    return stats.median(t), max(t), stats.median(e), max(e)


def counter_text(c):
    return ", ".join(f"{k}: {v}" for k, v in c.items()) or "keine"


def main():
    a = args()
    repo = a.repo.expanduser().resolve()
    try:
        intel = load_platform("Intel", repo / a.intel_dir, a.sessions, a.reps)
        amd = load_platform("AMD", repo / a.amd_dir, a.sessions, a.reps)
    except (OSError, RuntimeError) as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2

    si, sa = summarize(intel), summarize(amd)
    comp = winners(si, sa)
    sti, sta = stability(si), stability(sa)
    issues = intel["issues"] + amd["issues"]
    status = "FAIL" if any(s == "FAIL" for s, _ in issues) else (
        "PASS MIT WARNUNGEN" if issues else "PASS"
    )

    unstable = []
    for p, summary in (("Intel", si), ("AMD", sa)):
        for (n, t), x in summary.items():
            unstable.append((max(x["cv_time"], x["cv_energy"]), p, n, t,
                             x["cv_time"], x["cv_energy"]))
    unstable.sort(reverse=True)

    lines = [
        "# AXPY CPU – kleiner Vorabcheck",
        "",
        f"**Status: {status}**",
        "",
        "| Plattform | Kampagne | Sessions | Zeilen | Gerät |",
        "|---|---|---:|---:|---|",
        f"| Intel | `{intel['prefix']}` | {len(intel['files'])} | {len(intel['rows'])} | "
        f"{', '.join(intel['devices']) or 'n/a'} |",
        f"| AMD | `{amd['prefix']}` | {len(amd['files'])} | {len(amd['rows'])} | "
        f"{', '.join(amd['devices']) or 'n/a'} |",
        "",
        "## Streuung",
        "",
        "| Plattform | Median CV Zeit/Op | Max CV Zeit/Op | Median CV Package-Energie/Op | Max CV Package-Energie/Op |",
        "|---|---:|---:|---:|---:|",
        f"| Intel | {pct(sti[0])} | {pct(sti[1])} | {pct(sti[2])} | {pct(sti[3])} |",
        f"| AMD | {pct(sta[0])} | {pct(sta[1])} | {pct(sta[2])} | {pct(sta[3])} |",
        "",
        "Faustregel: CV ≤ 5 % unauffällig, 5–10 % erhöht, > 10 % später gezielt prüfen.",
        "",
        "## Intel–AMD, deskriptiv",
        "",
        f"- Gemeinsame Konfigurationen: **{len(comp['common'])}**",
        f"- Median AMD/Intel Laufzeit pro Operation: **{ratio(comp['time_ratio'])}**",
        f"- Median AMD/Intel Package-Energie pro Operation: **{ratio(comp['energy_ratio'])}**",
        f"- Laufzeitgewinner (±2 % Totzone): {counter_text(comp['time_winners'])}",
        f"- Package-Energiegewinner (±2 % Totzone): {counter_text(comp['energy_winners'])}",
        f"- Dominanz/Trade-off: {counter_text(comp['dominance'])}",
        "",
        "## Höchste Streuungen",
        "",
        "| Plattform | N | Threads | CV Zeit/Op | CV Package-Energie/Op |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, p, n, t, ct, ce in unstable[:8]:
        lines.append(f"| {p} | {n} | {t} | {pct(ct)} | {pct(ce)} |")

    lines += [
        "",
        "## Wissenschaftliche Einordnung",
        "",
        "- Der Vergleich verwendet `energy_per_op_j`, also Package-/Device-Energie. "
        "Da AMD kein DRAM-RAPL liefert, ist dies kein vollständiger Systemenergievergleich.",
        "- Der Check prüft Integrität, Vollständigkeit, Formelanker, Streuung und grobe Plausibilität. "
        "Konfidenzintervalle und session-gepaarte Endanalyse folgen später.",
        "- Source-/Header-/Runner-Hashes müssen zusammen mit den Logs archiviert bleiben; "
        "sie stehen nicht in jeder CSV-Zeile.",
    ]

    if issues:
        lines += ["", "## Meldungen", ""]
        for severity, message in issues[:30]:
            lines.append(f"- **{severity}:** {message}")
        if len(issues) > 30:
            lines.append(f"- … {len(issues)-30} weitere Meldungen ausgelassen.")
    else:
        lines += [
            "",
            "Keine strukturellen oder numerischen Probleme gefunden. "
            "Die Daten sind grundsätzlich für die spätere wissenschaftliche Analyse verwendbar.",
        ]

    report = repo / a.report
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("=" * 68)
    print(f"AXPY CPU VORABCHECK: {status}")
    print(f"Intel: {intel['prefix']} ({len(intel['rows'])} Zeilen)")
    print(f"AMD:   {amd['prefix']} ({len(amd['rows'])} Zeilen)")
    print(f"AMD/Intel Zeit:    {ratio(comp['time_ratio'])}")
    print(f"AMD/Intel Energie: {ratio(comp['energy_ratio'])} (Package)")
    print(f"Bericht: {report}")
    print("=" * 68)
    return 1 if status == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())

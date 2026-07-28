# AXPY analysis v5 — audit closure

Behoben:

1. Campaign-Lock fail-open und fehlende Rohdaten-Hashprüfung.
2. GPU-Quickcheck-`PASS` konnte `FATAL`/Checksumfehler überstimmen.
3. Rekalibrierungsereignisse in offiziellen Logs wurden nicht ausgewiesen.
4. In-Range-Vollständigkeit prüfte nur Sessionabdeckung statt 5×10 Repetitionen.
5. Telemetriebegriff war stärker als die gemessene within-window Evidenz.
6. GPU-CPU-Counter- und CPU-Throttle-Sentinels fehlten.
7. Quickcheck-Logs und Analyseparameter fehlten im Freeze/Handover.

Regressionstests:

- Python-Syntax: PASS
- Shell-Syntax: PASS
- v5-Selftest: PASS
- synthetischer 8100-Zeilen-Vierplattformlauf: PASS_WITH_WARNINGS
- gültiger Lock-Rerun: PASS
- manipulierter Rohdatenhash: harter Abbruch vor Analyse
- unvollständiger Lock: harter Abbruch
- GPU-PASS plus FATAL: abgelehnt
- In-Range 5 Sessions, aber 2/10 in einer Session: aus 5×10-Primäranalyse ausgeschlossen

# AXPY-Auswertung v5 – gehärtete Referenzanalyse

Dieses Paket validiert und vergleicht die vollständigen AXPY-Kampagnen von:

- Intel i9-7900X
- AMD Threadripper 3970X
- RTX 3090
- RTX 5060 Ti

## Wesentliche v5-Korrekturen

- Campaign-Lock ist jetzt ein echter Byte-Freeze (`axpy-campaign-lock-v2`).
- Lock-Reruns prüfen fail-closed Schema, Plattformen, Kampagnen, Rohdateien und SHA-256.
- GPU-Quickchecks können durch `PASS` plus Anti-Collapse/Messnachweis oder durch direkten Messnachweis bestehen; `FATAL` und Checksumfehler können nie überstimmt werden.
- Ausgewählte Quickcheck-Logs und deren Hashes werden im Lock festgehalten und ins Handover kopiert.
- Offizielle Logs werden auf Rekalibrierungs- und Below-Retry-Ereignisse geprüft.
- Plattform-Sentinels werden vollständig geprüft.
- Die In-Range-Primärsensitivität verlangt exakt 5 Sessions mit jeweils 10 In-Range-Repetitionen.
- Telemetrie meldet nur `within_window_clock_drop_detected`; ein bereits vor Messbeginn reduzierter Takt wird dadurch ausdrücklich nicht ausgeschlossen.
- `ANALYSIS_COMPLETE.json` enthält Bootstrap-, Totzonen-, Session- und Repetitionsparameter.

## Installation

```bash
cd ~/projects/energy

mkdir -p new/analyse/AXPY/scripts
unzip -o ~/Downloads/AXPY_ANALYSIS_ALL_PLATFORMS_PATCHED_v5.zip \
  -d new/analyse/AXPY/scripts

chmod +x new/analyse/AXPY/scripts/run_all_axpy_analysis.sh
```

## Normaler Analyselauf

```bash
cd ~/projects/energy
bash new/analyse/AXPY/scripts/run_all_axpy_analysis.sh
```

## Bytegenauer Wiederholungslauf mit Campaign-Lock

Nach einem erfolgreichen ersten v5-Lauf:

```bash
cd ~/projects/energy

AXPY_CAMPAIGN_LOCK="$PWD/new/analyse/AXPY/all_platforms/axpy_campaign_lock.json" \
  bash new/analyse/AXPY/scripts/run_all_axpy_analysis.sh
```

Jede Abweichung an einer gesperrten CSV, einem Session-Log, einem Manifest oder einem bestandenen Quickcheck-Log führt vor Beginn der Analyse zum Abbruch.

Ein alter v1-Lock wird bewusst nicht akzeptiert. Einmal ohne Lock starten, um einen v2-Lock zu erzeugen.

## Parameter

Standardwerte:

```text
BOOTSTRAP_RESAMPLES=5000
TIE_PERCENT=2.0
REQUIRED_SESSIONS=5
REQUIRED_REPETITIONS=10
```

Beispiel:

```bash
BOOTSTRAP_RESAMPLES=5000 TIE_PERCENT=2.0 \
  bash new/analyse/AXPY/scripts/run_all_axpy_analysis.sh
```

## Wichtigste Ausgaben

```text
new/analyse/AXPY/all_platforms/
├── axpy_campaign_lock.json
├── axpy_provenance_report.md
├── axpy_recalibration_summary.csv
├── axpy_validation_report.md
├── axpy_telemetry_report.md
├── axpy_comparison_report.md
├── axpy_robustness_report.md
├── axpy_in_range_sensitivity.csv
├── ANALYSIS_COMPLETE.json
└── AXPY_ANALYSIS_HANDOVER.zip
```

## Wissenschaftliche Grenze

Die primäre Energiegröße bleibt eine Device-Domain-Messung:

- CPU: RAPL-Package
- GPU: NVML-Board

Das ist insbesondere bei AXPY kein vollständiger Systemenergievergleich, weil externe CPU-DRAM-Energie fehlt, während GPU-VRAM im Board-Zähler enthalten ist.


## Patch v5.1

Der CPU-Writer der eingefrorenen Intel-/AMD-AXPY-Reihe schreibt das GPU-only-Feld
`throttle_reasons` absichtlich leer. Der v5-Validator verlangte irrtümlich `-1` und
verwarf deshalb sämtliche CPU-Zeilen. v5.1 akzeptiert für CPU-Zeilen exakt leer
oder numerisch `-1`; GPU-Regeln bleiben unverändert.

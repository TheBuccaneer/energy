# AXPY – Validierung und Aggregation

**Status: PASS MIT WARNUNGEN**

## Ausgewählte Kampagnen

| Plattform | Kampagne | Sessions | Manifest | Rohzeilen |
|---|---|---:|---|---:|
| Intel i9-7900X | `axpy_intel_20260725_163341` | 5 | nein | 3150 |
| AMD Threadripper 3970X | `axpy_amd_20260725_215406` | 5 | nein | 4050 |
| RTX 3090 | `axpy_3090_20260727_133600` | 5 | ja | 450 |
| RTX 5060 Ti | `axpy_5060ti_20260727_141707` | 5 | ja | 450 |

## Laufzeitfenster

| Plattform | Zeilen | über 1,25 s | Anteil | Median der Warnungen | Maximum |
|---|---:|---:|---:|---:|---:|
| Intel i9-7900X | 3150 | 0 | 0 % | n/a s | n/a s |
| AMD Threadripper 3970X | 4050 | 258 | 6.37 % | 1.3169 s | 2.509 s |
| RTX 3090 | 450 | 0 | 0 % | n/a s | n/a s |
| RTX 5060 Ti | 450 | 0 | 0 % | n/a s | n/a s |

## Eingefrorener AXPY-Vertrag

Jede Zeile wurde positionsgenau gegen `problem_spec` mit `alpha=3.0`, den eingefrorenen Periodenmustern, Reset außerhalb des Messfensters und `max_batches=250000` geprüft. Fehlerhafte Zeilen werden verworfen und in `axpy_rejected_rows.csv` dokumentiert.

## Aggregationsmethode

Für jede Konfiguration wird zuerst innerhalb jeder Session der Median der zehn Wiederholungen berechnet. Der finale Punktschätzer ist der Median der fünf Session-Mediane. Die 95-%-Intervalle sind ein deterministischer Percentile-Bootstrap über die Session-Mediane.

Die primäre Laufzeitmetrik ist E2E-Zeit pro logischer AXPY-Operation. Die primäre Energiegröße ist `device_energy_j / batches`: auf CPUs Package-Energie, auf GPUs Board-Energie.

## Stabilität

| Plattform | Konfigurationen | Median Zeit-CV | Max Zeit-CV | Median Energie-CV | Max Energie-CV |
|---|---:|---:|---:|---:|---:|
| Intel i9-7900X | 63 | 0.3572 % | 17.69 % | 2.169 % | 12.96 % |
| AMD Threadripper 3970X | 81 | 0.605 % | 34.48 % | 1.69 % | 32.17 % |
| RTX 3090 | 9 | 0.04888 % | 3.378 % | 1.56 % | 4.923 % |
| RTX 5060 Ti | 9 | 0.03638 % | 0.2477 % | 1.575 % | 2.124 % |

## Temperatur- und Takttelemetrie

Die vollständige Zusammenfassung steht in `axpy_telemetry_report.md`; die Felder `clock_before_mhz`, `clock_after_mhz`, `temp_before_c` und `temp_after_c` bleiben in den normalisierten Daten erhalten.

## GPU-Throttle-Masken

Die konkreten NVML-Bitmasken bleiben in den normalisierten Zeilen erhalten und werden nicht auf ein bloßes Ja/Nein reduziert.

| Plattform | Maske | Dekodierung | Zeilen |
|---|---|---|---:|
| RTX 3090 | `0x4` | software_power_cap | 450 |
| RTX 5060 Ti | `0x0` | none | 450 |

## Wissenschaftliche Grenzen

- Cross-device-Energie ist eine Device-Domain-Größe: CPU-Package gegen GPU-Board. Bei AXPY fehlt auf der CPU-Seite insbesondere externer DDR4-Verbrauch, während GPU-VRAM im Board-Zähler enthalten ist.
- `logical_bytes_per_op=12N` ist ein semantischer Anker, kein gemessener physischer Speicherverkehr.
- Fünf Sessions erlauben robuste deskriptive Aussagen, aber die Bootstrap-Intervalle ersetzen keine umfassende inferenzstatistische Modellierung.

## Meldungen

**FAIL: 0; WARN: 3**

- **WARN [intel] Kampagne:** Provenienzmanifest fehlt: axpy_intel_20260725_163341_manifest.txt.
- **WARN [amd] Kampagne:** Provenienzmanifest fehlt: axpy_amd_20260725_215406_manifest.txt.
- **WARN [intel] Kampagne:** CPU-Telemetrie erreicht 100.0 °C; 102 Zeilen liegen bei mindestens 95 °C. Siehe axpy_telemetry_report.md.

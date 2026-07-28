# AXPY – Provenienzprüfung

**Status: PASS MIT WARNUNGEN**

| Plattform | Kampagne | Auswahl | Manifest | CSV-Hashes | Log-Hashes | Loginhalt | Quickcheck | Status |
|---|---|---|---|---:|---:|---:|---|---|
| Intel i9-7900X | `axpy_intel_20260725_163341` | latest_complete | nein | 0/5 | 0/5 | 5/5 | pass (measurement_and_anti_collapse_evidence) | partial |
| AMD Threadripper 3970X | `axpy_amd_20260725_215406` | latest_complete | nein | 0/5 | 0/5 | 5/5 | missing (none) | partial |
| RTX 3090 | `axpy_3090_20260727_133600` | latest_complete | ja | 5/5 | 5/5 | 5/5 | pass (measurement_and_anti_collapse_evidence) | verified |
| RTX 5060 Ti | `axpy_5060ti_20260727_141707` | latest_complete | ja | 5/5 | 5/5 | 5/5 | pass (measurement_and_anti_collapse_evidence) | verified |

`manifest_parsed` bedeutet nur, dass ein Manifest gelesen wurde. Source-, Runner- und Binary-Status werden separat ausgewiesen; dadurch wird kein nicht geprüfter Hash als verifiziert bezeichnet.

GPU-Quickchecks dürfen durch Messnachweis statt Shell-PASS-Marker bestätigt werden: mindestens sechs AXPY-Zeilen, alle Checksummen OK, Anti-Collapse PASS und kein FATAL.

## Rekalibrierungsereignisse in offiziellen Logs

| Plattform | Rekalibrierungen | Below-Retries |
|---|---:|---:|
| Intel i9-7900X | 0 | 0 |
| AMD Threadripper 3970X | 0 | 0 |
| RTX 3090 | 0 | 0 |
| RTX 5060 Ti | 0 | 0 |

FAIL: 0; WARN: 3
- **WARN [intel/manifest]:** Manifest fehlt: axpy_intel_20260725_163341_manifest.txt
- **WARN [amd/manifest]:** Manifest fehlt: axpy_amd_20260725_215406_manifest.txt
- **WARN [amd/quickcheck]:** Kein Quickcheck-Log gefunden; Kampagne bleibt analysierbar, Provenienzstatus ist partial.

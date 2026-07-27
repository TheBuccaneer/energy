# AXPY CPU – kleiner Vorabcheck

**Status: PASS MIT WARNUNGEN**

| Plattform | Kampagne | Sessions | Zeilen | Gerät |
|---|---|---:|---:|---|
| Intel | `axpy_intel_20260725_163341` | 5 | 3150 | Intel(R) Core(TM) i9-7900X CPU @ 3.30GHz |
| AMD | `axpy_amd_20260725_215406` | 5 | 4050 | AMD Ryzen Threadripper 3970X 32-Core Processor |

## Streuung

| Plattform | Median CV Zeit/Op | Max CV Zeit/Op | Median CV Package-Energie/Op | Max CV Package-Energie/Op |
|---|---:|---:|---:|---:|
| Intel | 0.36 % | 17.69 % | 2.17 % | 12.96 % |
| AMD | 0.60 % | 34.48 % | 1.69 % | 32.17 % |

Faustregel: CV ≤ 5 % unauffällig, 5–10 % erhöht, > 10 % später gezielt prüfen.

## Intel–AMD, deskriptiv

- Gemeinsame Konfigurationen: **63**
- Median AMD/Intel Laufzeit pro Operation: **0.659×**
- Median AMD/Intel Package-Energie pro Operation: **0.776×**
- Laufzeitgewinner (±2 % Totzone): AMD: 61, Intel: 2
- Package-Energiegewinner (±2 % Totzone): AMD: 46, Intel: 14, gleich: 3
- Dominanz/Trade-off: AMD dominiert: 46, Intel dominiert: 2, Trade-off/gleich: 15

## Höchste Streuungen

| Plattform | N | Threads | CV Zeit/Op | CV Package-Energie/Op |
|---|---:|---:|---:|---:|
| AMD | 2000000 | 64 | 34.48 % | 32.17 % |
| AMD | 4000000 | 32 | 28.03 % | 25.16 % |
| Intel | 1000000 | 20 | 17.69 % | 12.96 % |
| AMD | 8000000 | 32 | 17.63 % | 16.10 % |
| AMD | 1000000 | 16 | 16.20 % | 12.97 % |
| Intel | 1000000 | 10 | 11.73 % | 9.81 % |
| AMD | 2000000 | 20 | 11.42 % | 6.75 % |
| AMD | 1000000 | 10 | 10.87 % | 5.67 % |

## Wissenschaftliche Einordnung

- Der Vergleich verwendet `energy_per_op_j`, also Package-/Device-Energie. Da AMD kein DRAM-RAPL liefert, ist dies kein vollständiger Systemenergievergleich.
- Der Check prüft Integrität, Vollständigkeit, Formelanker, Streuung und grobe Plausibilität. Konfidenzintervalle und session-gepaarte Endanalyse folgen später.
- Source-/Header-/Runner-Hashes müssen zusammen mit den Logs archiviert bleiben; sie stehen nicht in jeder CSV-Zeile.

## Meldungen

- **WARN:** axpy_amd_20260725_215406_session1.csv:12: Messung über 1,25 s.
- **WARN:** axpy_amd_20260725_215406_session1.csv:13: Messung über 1,25 s.
- **WARN:** axpy_amd_20260725_215406_session1.csv:14: Messung über 1,25 s.
- **WARN:** axpy_amd_20260725_215406_session1.csv:15: Messung über 1,25 s.
- **WARN:** axpy_amd_20260725_215406_session1.csv:16: Messung über 1,25 s.
- **WARN:** axpy_amd_20260725_215406_session1.csv:17: Messung über 1,25 s.
- **WARN:** axpy_amd_20260725_215406_session1.csv:18: Messung über 1,25 s.
- **WARN:** axpy_amd_20260725_215406_session1.csv:19: Messung über 1,25 s.
- **WARN:** axpy_amd_20260725_215406_session1.csv:20: Messung über 1,25 s.
- **WARN:** axpy_amd_20260725_215406_session1.csv:21: Messung über 1,25 s.
- **WARN:** axpy_amd_20260725_215406_session1.csv:62: Messung über 1,25 s.
- **WARN:** axpy_amd_20260725_215406_session1.csv:63: Messung über 1,25 s.
- **WARN:** axpy_amd_20260725_215406_session1.csv:64: Messung über 1,25 s.
- **WARN:** axpy_amd_20260725_215406_session1.csv:65: Messung über 1,25 s.
- **WARN:** axpy_amd_20260725_215406_session1.csv:66: Messung über 1,25 s.
- **WARN:** axpy_amd_20260725_215406_session1.csv:67: Messung über 1,25 s.
- **WARN:** axpy_amd_20260725_215406_session1.csv:68: Messung über 1,25 s.
- **WARN:** axpy_amd_20260725_215406_session1.csv:69: Messung über 1,25 s.
- **WARN:** axpy_amd_20260725_215406_session1.csv:70: Messung über 1,25 s.
- **WARN:** axpy_amd_20260725_215406_session1.csv:71: Messung über 1,25 s.
- **WARN:** axpy_amd_20260725_215406_session1.csv:312: Messung über 1,25 s.
- **WARN:** axpy_amd_20260725_215406_session1.csv:313: Messung über 1,25 s.
- **WARN:** axpy_amd_20260725_215406_session1.csv:314: Messung über 1,25 s.
- **WARN:** axpy_amd_20260725_215406_session1.csv:315: Messung über 1,25 s.
- **WARN:** axpy_amd_20260725_215406_session1.csv:316: Messung über 1,25 s.
- **WARN:** axpy_amd_20260725_215406_session1.csv:317: Messung über 1,25 s.
- **WARN:** axpy_amd_20260725_215406_session1.csv:318: Messung über 1,25 s.
- **WARN:** axpy_amd_20260725_215406_session1.csv:319: Messung über 1,25 s.
- **WARN:** axpy_amd_20260725_215406_session1.csv:320: Messung über 1,25 s.
- **WARN:** axpy_amd_20260725_215406_session1.csv:321: Messung über 1,25 s.
- … 228 weitere Meldungen ausgelassen.

# Audit der RTX-3090-GEMM-Kampagne

## Gesamturteil

**PASS WITH WARNINGS — wissenschaftlich gut verwendbar.**

Die Kampagne `20260719_152731` ist vollständig und intern konsistent:

- 5 Session-Dateien;
- 450 Messungen;
- 9 Größen × 10 Wiederholungen × 5 Sessions;
- keine fehlenden oder doppelten Messgruppen;
- keine Checksum-Fehler;
- alle Zeit-, Energie-, FLOP-, Leistungs- und Durchsatzformeln reproduzierbar;
- `gpu_resident`;
- striktes FP32 über `CUBLAS_COMPUTE_32F_PEDANTIC`;
- TF32 über `NVIDIA_TF32_OVERRIDE=0` deaktiviert;
- Energie über den direkten NVML-Gesamtenergiezähler;
- keine ernsthaften Throttle-Gründe;
- keine instabilen Größen bei Laufzeit, Energie oder Durchsatz.

## Bedeutung der Warnungen

### 1. Abweichende Spaltenreihenfolge

Im GPU-CSV steht `device_energy_j` vor `total_energy_j`, während die CPU-v2-Dateien
die umgekehrte Reihenfolge verwenden. Die Namen und Werte sind korrekt und beide
Energiefelder sind auf der GPU identisch. Die Analyse arbeitet spaltennamenbasiert.

**Bewertung:** reines Schema-/Kompatibilitätsproblem, kein Messfehler. Für zukünftige
gemeinsame Parser sollte die Reihenfolge vereinheitlicht werden. Die bestehenden
3090-Daten müssen deswegen nicht neu gemessen werden.

### 2. Zielzeitfenster: 88,89 %

400 von 450 Messungen liegen im Zielbereich von 0,75–1,25 Sekunden. Sämtliche
50 Abweichungen gehören zu `N=16384`. Dort wurden 4 GEMMs pro Messfenster
ausgeführt, bei einer medianen Laufzeit von etwa 0,3325 Sekunden pro GEMM.
Das ergibt ungefähr 1,33 Sekunden pro Messfenster.

Drei GEMMs hätten rechnerisch etwa 1,00 Sekunde ergeben. Die Abweichung ist daher
nicht unvermeidbar, sondern zeigt eine leichte Überkalibrierung bei der größten
Matrix.

**Bewertung:** kein Gültigkeitsfehler. Die per-GEMM-Werte sind stabil, das Fenster
liegt nur rund 6,5 % oberhalb der festgelegten Obergrenze. Die Kalibrierungslogik
sollte erst geändert werden, wenn anschließend beide GPUs mit derselben
korrigierten Version neu gemessen werden. Für einen direkten 3090–5060-Ti-Vergleich
ist identischer Code wichtiger als eine nur einseitig korrigierte Kalibrierung.

### 3. Robuste Ausreißerquote: 14,44 %

65 von 450 Zeilen wurden vom robusten Diagnoseverfahren markiert. Das ist keine
Fehlerrate und es wurden keine Zeilen verworfen. Gleichzeitig beträgt:

- die Run-Level-Laufzeitstreuung je Größe höchstens 0,44 %;
- die Run-Level-Energiestreuung je Größe 1,71–3,21 %;
- die Session-Level-Laufzeitstreuung höchstens 0,48 %;
- die Session-Level-Energiestreuung höchstens 0,89 %;
- die Spannweite der globalen Session-Median-Durchsätze nur 0,22 %.

Die hohe Markierungsquote entsteht damit wahrscheinlich durch eine empfindliche
MAD-basierte Regel bei sehr engen Verteilungen und durch die gröbere
NVML-Energieauflösung.

**Bewertung:** Die markierten Zeilen dürfen nicht automatisch gelöscht werden.
Die Kampagne ist auf Größen- und Session-Ebene sehr stabil. Eine spätere Version
des Analyseskripts sollte für jede Markierung die auslösende Metrik und den
robusten Score ausgeben.

## Thermik, Takt und Leistungsgrenze

- maximale Temperatur: 67 °C;
- Session-Mediane: 62–65 °C;
- medianer Taktabfall im Messfenster: 0,00 %;
- keine ernsthaften Throttle-Zeilen;
- 300 Zeilen tragen `0x4 = software_power_cap`.

`software_power_cap` ist keine thermische Drosselung. Es zeigt, dass die GPU bei
größeren GEMMs die konfigurierte Software-Leistungsgrenze erreicht. Das ist ein
normaler und reproduzierbarer Betriebszustand, muss aber als Teil der
Messkonfiguration dokumentiert werden.

## Wissenschaftliche Kernergebnisse

|     N |   Runtime/GEMM (ms) |   Energy/GEMM (J) |   Throughput (TFLOP/s) |   Board power (W) |   GFLOP/J |   Temp. (°C) |
|------:|--------------------:|------------------:|-----------------------:|------------------:|----------:|-------------:|
|    64 |          0.00521703 |       0.000841273 |                  0.1   |             161.3 |      0.62 |         57   |
|   128 |          0.0051611  |       0.000922996 |                  0.813 |             178.8 |      4.54 |         58   |
|   256 |          0.00718051 |       0.0019916   |                  4.673 |             277.2 |     16.86 |         62   |
|   512 |          0.0226917  |       0.00721341  |                 11.83  |             317.9 |     37.22 |         65.5 |
|  1024 |          0.120991   |       0.0399125   |                 17.749 |             329.9 |     53.8  |         64   |
|  2048 |          0.752203   |       0.250474    |                 22.84  |             332.7 |     68.64 |         64   |
|  4096 |          5.55136    |       1.71844     |                 24.758 |             309.7 |     79.95 |         64   |
|  8192 |         41.739      |      13.1432      |                 26.343 |             315.2 |     83.58 |         64   |
| 16384 |        332.494      |     103.624       |                 26.455 |             311.2 |     85.01 |         63   |

Der Durchsatz wächst stark bis zu mittleren und großen Matrizen und sättigt
anschließend:

- 17,75 TFLOP/s bei `N=1024`;
- 22,84 TFLOP/s bei `N=2048`;
- 24,76 TFLOP/s bei `N=4096`;
- 26,34 TFLOP/s bei `N=8192`;
- 26,46 TFLOP/s bei `N=16384`.

Von `N=8192` zu `N=16384` steigt der Durchsatz nur noch um etwa 0,43 %. Die
RTX 3090 erreicht für diese strikt-pedantische FP32-GEMM damit ein Plateau von
ungefähr 26,4 TFLOP/s.

Die Board-Energieeffizienz steigt mit der Problemgröße deutlich:

- 0,62 GFLOP/J bei `N=64`;
- 53,80 GFLOP/J bei `N=1024`;
- 79,95 GFLOP/J bei `N=4096`;
- 85,01 GFLOP/J bei `N=16384`.

Kleine Matrizen amortisieren Launch- und Geräte-Overheads schlecht. Große
Matrizen nutzen die GPU deutlich effizienter.

## Korrektur zur automatisch erzeugten Ergebnisprosa

Die Aussage „lowest median EDP at N=64“ ist rechnerisch richtig, aber
wissenschaftlich nicht als Optimierungsbefund brauchbar: Die Größen führen
unterschiedlich viel Arbeit aus, und absolute Energie, Laufzeit sowie EDP müssen
bei kleineren Jobs zwangsläufig geringer sein. Sinnvoll sind stattdessen:

- Durchsatz;
- GFLOP/J beziehungsweise Energie pro FLOP;
- Skalierungs- und Sättigungsverhalten;
- später der Vergleich derselben Größe zwischen Geräten.

## Messdomäne und Vergleichbarkeit

Diese Kampagne misst ausschließlich `gpu_resident`:

- Allokation und Initialisierung außerhalb des Messfensters;
- keine PCIe-Transfers im Messfenster;
- NVML-Boardenergie einschließlich Gerätespeicher.

CPU-RAPL-Package und GPU-NVML-Boardenergie sind unterschiedliche Messdomänen.
Ein CPU–GPU-Vergleich ist als Vergleich der jeweils gemessenen Gerätedomänen
zulässig, muss diese Asymmetrie aber offen benennen. Für eine reale
Placement-Entscheidung sollte später zusätzlich `gpu_e2e` mit Transfers als
Sensitivitätsanalyse betrachtet werden.

## Endgültige Freigabe

Die RTX-3090-GEMM-Kampagne kann als offizieller residenter FP32-GPU-Datensatz
verwendet werden. Eine Wiederholung ist aufgrund der vorliegenden Daten nicht
erforderlich. Die drei Warnungen müssen dokumentiert, aber nicht als
Messfehler behandelt werden.

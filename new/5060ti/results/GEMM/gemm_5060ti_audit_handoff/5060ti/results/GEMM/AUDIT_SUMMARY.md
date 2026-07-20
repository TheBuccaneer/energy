# Audit der RTX-5060-Ti-GEMM-Kampagne

## Gesamturteil

**PASS WITH WARNINGS — wissenschaftlich gut verwendbar.**

Die Kampagne `20260719_172746` ist vollständig und intern konsistent:

- 5 Session-Dateien;
- 450 Messungen;
- 9 Größen × 10 Wiederholungen × 5 Sessions;
- keine fehlenden oder doppelten Messgruppen;
- keine Checksum-Fehler;
- alle Zeit-, Energie-, FLOP-, Leistungs- und Durchsatzformeln reproduzierbar;
- `gpu_resident`;
- striktes FP32 über `CUBLAS_COMPUTE_32F_PEDANTIC`;
- TF32 deaktiviert;
- Energie über den direkten NVML-Gesamtenergiezähler;
- keine ernsthaften Throttle-Gründe;
- keine instabilen Größen bei Laufzeit, Energie oder Durchsatz.

## Bedeutung der Warnungen

### 1. Abweichende Spaltenreihenfolge

Im GPU-CSV steht `device_energy_j` vor `total_energy_j`, während die CPU-v2-Dateien
die umgekehrte Reihenfolge verwenden. Namen und Werte sind korrekt. Die Analyse
arbeitet spaltennamenbasiert.

**Bewertung:** reines Schema-/Kompatibilitätsproblem, kein Messfehler und kein
Grund für eine Wiederholung.

### 2. Zielzeitfenster: 88,89 %

400 von 450 Messungen liegen im Zielbereich von 0,75–1,25 Sekunden. Alle
50 Abweichungen gehören zu `N=16384`. Dort gilt `batches=1`, und eine einzelne
GEMM benötigt im Median etwa 1,793 Sekunden.

Die Analyse weist deshalb eine **actionable target-window share von 100 %** aus:
Es existiert keine kleinere zulässige Batchzahl. Anders als bei einer
Überkalibrierung kann die Laufzeit hier nicht weiter reduziert werden.

**Bewertung:** unvermeidbare Mindestbatch-Abweichung, kein Kalibrierungs- oder
Gültigkeitsfehler.

## Robuste Ausreißerquote: 24 %

108 von 450 Zeilen werden vom robusten MAD-Verfahren markiert. Das ist keine
Fehlerrate und keine Zeile wurde verworfen. Gleichzeitig gilt:

- maximale Run-Level-Laufzeitstreuung: 1.57 %;
- maximale Run-Level-Energiestreuung: 4.50 %;
- maximale Session-Level-Laufzeitstreuung: 1.35 %;
- maximale Session-Level-Energiestreuung: 3.33 %;
- Spannweite der globalen Session-Median-Durchsätze: 0.46 %.

Die Markierungen verteilen sich über alle fünf Sessions und alle neun Größen.
Damit gibt es keinen Hinweis auf eine einzelne defekte Session oder Größe. Die
hohe Quote entsteht wahrscheinlich durch die empfindliche MAD-Regel bei sehr
engen Verteilungen und diskreter NVML-Energieauflösung.

**Bewertung:** keine automatischen Ausschlüsse. Für eine spätere Analyseversion
sollten auslösende Metrik und robuster Score je Markierung gespeichert werden.

## Thermik, Takt und Throttling

- maximale Temperatur: 75 °C;
- Session-Mediane: 66–68 °C;
- medianer Taktabfall im Messfenster: 0,00 %;
- ernsthafte Throttle-Zeilen: 0;
- alle 450 Zeilen melden `throttle_mask=0x0`;
- beobachteter Leistungsbereich: 32,4–169,6 W bei einem 180-W-Limit.

Die Kampagne zeigt weder thermische noch Power-Cap-Drosselung. Das thermische
Verhalten ist stabil und unkritisch.

## Wissenschaftliche Kernergebnisse

|     N |   Runtime/GEMM (ms) |   Energy/GEMM (J) |   Throughput (TFLOP/s) |   Board power (W) |   GFLOP/J |   Temp. (°C) |
|------:|--------------------:|------------------:|-----------------------:|------------------:|----------:|-------------:|
|    64 |          0.0099198  |       0.000332818 |                  0.053 |              33.6 |      1.57 |         48.5 |
|   128 |          0.00992057 |       0.000418624 |                  0.423 |              41.7 |     10.13 |         50   |
|   256 |          0.0163733  |       0.00124799  |                  2.049 |              76.2 |     26.9  |         57   |
|   512 |          0.0695993  |       0.00748716  |                  3.857 |             107.6 |     35.85 |         66   |
|  1024 |          0.448735   |       0.0553448   |                  4.786 |             123.3 |     38.8  |         67   |
|  2048 |          3.51052    |       0.471474    |                  4.894 |             133.7 |     36.59 |         68   |
|  4096 |         28.1354     |       3.8981      |                  4.885 |             138.6 |     35.26 |         70   |
|  8192 |        224.475      |      31.1611      |                  4.898 |             138.8 |     35.28 |         70   |
| 16384 |       1793.22       |     298.952       |                  4.905 |             166.7 |     29.42 |         72.5 |

## Durchsatz und Sättigung

Der Durchsatz steigt bis `N=1024` stark an und erreicht dort bereits
4.786 TFLOP/s. Danach entsteht ein ausgeprägtes Plateau:

- `N=2048`: 4.894 TFLOP/s;
- `N=4096`: 4.885 TFLOP/s;
- `N=8192`: 4.898 TFLOP/s;
- `N=16384`: 4.905 TFLOP/s.

Von `N=1024` bis `N=16384` steigt der Durchsatz nur noch um
2.50 %. Von `N=8192` zu `N=16384` beträgt der Zuwachs nur
0.14 %. Das resident-pedantische FP32-GEMM-Plateau liegt
damit bei ungefähr 4,9 TFLOP/s.

## Energieeffizienz

Die höchste Board-Effizienz wird nicht bei der größten Matrix erreicht, sondern
bei `N=1024` mit 38.80 GFLOP/J.

Danach sinkt die Effizienz:

- `N=2048`: 36.59 GFLOP/J;
- `N=4096`: 35.26 GFLOP/J;
- `N=8192`: 35.28 GFLOP/J;
- `N=16384`: 29.42 GFLOP/J.

Von `N=8192` zu `N=16384` steigt die mediane Board-Leistung um
20.10 %, während der Durchsatz praktisch konstant
bleibt. Dadurch sinkt die Effizienz um 16.61 %.
Gegenüber dem Effizienzmaximum bei `N=1024` beträgt der Rückgang bis `N=16384`
24.17 %.

Dieser Befund ist für die spätere Placement-Analyse relevant: maximale
Auslastung beziehungsweise größte Problemgröße bedeutet auf der RTX 5060 Ti
nicht automatisch maximale Energieeffizienz.

## Korrektur zur automatisch erzeugten Ergebnisprosa

Die Aussage „lowest median EDP at N=64“ ist rechnerisch korrekt, aber als
Optimierungsbefund wissenschaftlich trivial, weil die Größen unterschiedlich
viel Arbeit enthalten. Über Größen hinweg sind stattdessen relevant:

- Durchsatz;
- GFLOP/J beziehungsweise Energie pro FLOP;
- Sättigungsverhalten;
- Vergleiche derselben Größe zwischen Geräten.

## Messdomäne und Grenzen

Diese Kampagne misst ausschließlich `gpu_resident`:

- Allokation und Initialisierung außerhalb des Messfensters;
- keine PCIe-Transfers im Messfenster;
- NVML-Boardenergie einschließlich Gerätespeicher.

Die Prüfung bestätigt Reproduzierbarkeit und methodische Konsistenz des gewählten
pedantischen FP32-Pfads. Sie beweist nicht, dass dieser Modus die maximal mögliche
Leistung der Architektur erreicht. Für die Studie ist er dennoch vergleichbar,
sofern auf RTX 3090 und RTX 5060 Ti exakt derselbe CUDA-Code und dieselbe
Compute-Semantik verwendet werden.

## Endgültige Freigabe

Die RTX-5060-Ti-GEMM-Kampagne kann als offizieller residenter FP32-GPU-Datensatz
verwendet werden. Eine Wiederholung ist nicht erforderlich. Die beiden
Validatorwarnungen und die diagnostische Ausreißerquote müssen dokumentiert,
aber nicht als Messfehler behandelt werden.

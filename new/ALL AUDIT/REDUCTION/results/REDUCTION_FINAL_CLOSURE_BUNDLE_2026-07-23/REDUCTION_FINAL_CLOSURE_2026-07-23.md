# REDUCTION — finales Abschluss-, Freeze- und Handover-Dokument

**Freeze-Datum:** 23. Juli 2026  
**Status:** **FINAL — PASS WITH REPORTING QUALIFICATIONS**  
**Projektpfad:** `~/projects/energy/new`  
**Finale Analysepipeline:** `REDUCTION_analysis_all_platforms_v1_4.zip`  
**Vollständiger Rerun erforderlich:** **nein**

---

## 0. Zweck und verbindlicher Status

Dieses Dokument ist die verbindliche Abschlussreferenz für den vollständig
gemessenen und analysierten Workload **REDUCTION**. Es bewahrt:

- den finalen Messvertrag;
- die akzeptierten Kampagnen;
- die finale Analyseversion;
- die Validierungs- und Auditentscheidungen;
- die belastbaren Zahlen;
- die zulässigen und unzulässigen Paper-Claims;
- die noch erforderlichen Reportingpatches;
- optionale Zusatzexperimente;
- alle relevanten Datei- und Hashreferenzen.

Spätere Diskussionen sollen von diesem Stand ausgehen. Bereits gelöste
Validatorprobleme dürfen nicht erneut als Messfehler interpretiert werden,
solange keine neue Rohdatenevidenz vorliegt.

### Endgültiges Urteil

```text
Messkampagnen:       PASS WITH WARNINGS
Einzelanalysen:      PASS WITH WARNINGS
Vier-Plattform-Audit: PASS WITH WARNINGS
Unabhängiges Audit:  PASS WITH REPORTING AND ARCHIVAL QUALIFICATIONS
Rerun:               nicht erforderlich
```

Die verbleibenden Qualifikationen betreffen Reporting, Stabilitätskennzeichnung,
Messgrenzen und eine noch nicht mechanistisch bestätigte Regimeinterpretation.
Sie invalidieren weder die Messungen noch die zentralen Placement-Befunde.

---

## 1. Source-of-Truth-Hierarchie

Für REDUCTION gilt künftig diese Reihenfolge:

1. **Dieses Abschlussdokument**
2. die fünf eingefrorenen Ergebnisarchive samt Kampagnenmanifesten;
3. die finale Analysepipeline **v1.4**;
4. das unabhängige Audit-Bundle vom 23. Juli 2026;
5. die gebundenen Source- und Runnerdateien in den Kampagnenmanifesten;
6. ältere Projektpläne und alte CSV-Schemadokumente nur als historische
   Projektentwicklung.

### Explizit überholte ältere Angaben

`Plan-v8.2-FINAL-POLISHED.md` und `CSV_SCHEMA_FINAL(2).md` sind für den
finalen REDUCTION-Vertrag **nicht** maßgeblich, soweit sie widersprechen.
Insbesondere sind folgende alte Angaben überholt:

| Alter Stand | Finaler REDUCTION-Stand |
|---|---|
| `cpu-gpu-v1`, 42 Spalten | `cpu-gpu-v2`, 45 Spalten |
| Dot-with-ones beziehungsweise zwei Eingabearrays | Ein-Array-Summe `sum(x[0:N])` |
| logische Bytes `8N` | logische Bytes `4N+4` |
| FLOP-Näherung `N` | exakt `N-1` Additionen |
| `total_energy_j` als einzige zentrale Energiegröße | primäre `device_energy_j`; `total_energy_j` nur Zusatzgröße |
| 50 Wiederholungen als flache Menge | 5 Sitzungen × 10 technische Wiederholungen; Sitzungsmedian ist Einheit |

Die älteren Dokumente bleiben für die allgemeine Entwicklung der
Workload-Suite nützlich, sind aber keine REDUCTION-Freeze-Spezifikation.

---

## 2. Finaler Mess- und Implementierungsvertrag

### 2.1 Mathematische Operation

```text
Eine Operation = vollständige FP32-Summe von x[0:N]
Ausgabe         = ein FP32-Skalar
FLOPs/Operation = N-1 FP32-Additionen
logische Bytes  = 4*N + 4
```

`4*N+4` ist ein **semantischer Datenanker**. Es ist kein gemessener
DRAM-/VRAM-Verkehr und schließt unter anderem CPU-Partials und
CUB-Workspace-Verkehr aus.

### 2.2 Größen

```text
N = 1M, 2M, 4M, 8M, 16M, 32M, 64M, 128M, 256M
```

Dabei sind `M` dezimale Millionen. Die semantische Ein-/Ausgabemenge reicht
damit ungefähr von 4 MB bis 1.024 MB.

### 2.3 CPU

```text
Implementierung: openmp_blocked_sum_fp32
Blockgröße:      4096 Elemente
Hierarchie:      SIMD-Blocksumme → explizite serielle Finalsumme
Akkumulation:    FP32
Referenz:        O(1), long double
Checksum-Grenze: relativer Fehler <= 1e-4
```

Threadraster:

```text
Intel: 1, 2, 4, 8, 10, 16, 20 Threads
AMD:   1, 2, 4, 8, 10, 16, 20, 32, 64 Threads
```

### 2.4 GPU

```text
Implementierung: cub_device_reduce_sum_fp32
Primitive:       CUB DeviceReduce::Sum
Modus:           gpu_resident
```

Allokationen und PCIe-Transfers liegen außerhalb des gemessenen Intervalls.
Beide GPUs verwenden byteidentischen CUDA-Sourcecode.

### 2.5 Messfenster und Wiederholung

```text
Adaptive Batchkalibrierung: Ziel ungefähr 1 s pro Rohmessung
Sitzungen:                  5
technische Reps/Sitzung:   10 pro Konfiguration
statistische Einheit:       Sitzungsmedian
inferenzielle Stichprobe:   n=5, nicht n=50
```

Ausreißer werden markiert, aber nicht still entfernt.

### 2.6 Primäre Energiedomänen

```text
Intel:      CPU Package RAPL
AMD:        CPU Package RAPL
RTX 3090:   NVML Board Energy einschließlich VRAM
RTX 5060 Ti: NVML Board Energy einschließlich VRAM
```

Die separate Intel-DRAM-RAPL-Zone ist bewusst keine primäre
Cross-Platform-Metrik. Sie bleibt eine Intel-interne Sensitivitätsgröße.

Die verbleibende Messgrenzenasymmetrie muss im Paper offen benannt werden:

```text
CPU: Package, externes DRAM nicht enthalten
GPU: Board, VRAM enthalten
```

Das ist eine eingefrorene Studienentscheidung, kein Analysefehler.

### 2.7 Ableitung von Leistung und EDP

`power_w` ist der Median der bereits pro Rohwiederholung berechneten
Verhältnisse `energy/time`.

Daher gilt allgemein:

```text
median(E/T) != median(E) / median(T)
```

Das ist kein Rechenfehler. `edp_j_s` wird dagegen aus den zusammengefassten
Energie- und Laufzeitwerten gebildet. Diese unterschiedliche
Aggregationsreihenfolge muss im Methodenteil dokumentiert werden.

---

## 3. Eingefrorene Kampagnen und Provenienz

| Plattform   |        Kampagne |   Rohzeilen | Source                                                                          | Source-SHA-256                                                   | Runner                                                                            | Runner-SHA-256                                                   |
|:------------|----------------:|------------:|:--------------------------------------------------------------------------------|:-----------------------------------------------------------------|:----------------------------------------------------------------------------------|:-----------------------------------------------------------------|
| RTX 3090    | 20260722_191930 |         450 | /home/rock/projects/energy/new/3090/scripts/REDUCTION/main_reduction.cu         | 2791f28e575f34e0f802b08b22e3df6f069e98b5b9a2eaff7f651108225f892c | /home/rock/projects/energy/new/3090/02_run_GPU_3090_REDUCTION_only.sh             | 482ff73c0203128bac58277c3da374fa70f4123e3c4ba079da8c9710adbc773a |
| RTX 5060 Ti | 20260722_050953 |         450 | /home/rock/projects/energy/new/5060ti/scripts/REDUCTION/main_reduction.cu       | 2791f28e575f34e0f802b08b22e3df6f069e98b5b9a2eaff7f651108225f892c | /home/rock/projects/energy/new/5060ti/scripts/02_run_GPU_5060Ti_REDUCTION_only.sh | 2ea114b6d6139888dacf824821dc79c48f3f4d7871584d9750346812de052044 |
| AMD         | 20260722_174206 |        4050 | /home/rock/projects/energy/new/AMD/scripts/REDUCTION/main_reduction_amd.cpp     | 7138ae0caac769b5cfeaa6a722da22df94e49e94cfe3f095979ada3bdd01eea2 | /home/rock/projects/energy/new/AMD/scripts/02_run_CPU_AMD_REDUCTION_only.sh       | 0d01466078b1aeb497657d2c0091965b77a316998ee602bf11bf4e6aa07cc699 |
| Intel       | 20260721_164732 |        3150 | /home/rock/projects/energy/new/INTEL/scripts/REDUCTION/main_reduction_intel.cpp | b0ece22c5eda5ca42ba9dadfb87bc42d73ec070c55b75e2d1a7b1082132ab468 | /home/rock/projects/energy/new/INTEL/scripts/02_run_CPU_Intel_REDUCTION_only.sh   | fc83509a1a4517deb118f3679cdf479873431327c1137de5be0e97dc7bcf2b84 |

### Ergebnisarchive

| Rolle                                | Datei             | SHA-256                                                          |
|:-------------------------------------|:------------------|:-----------------------------------------------------------------|
| Gemeinsame Vier-Plattform-Ergebnisse | REDUCTION(1).zip  | 3ab54ebdd2d35a2a9de7fd77a3f6004310cf8e7351754e6843500d176a6acd23 |
| RTX-3090-Plattformergebnisse         | REDUCTION2(1).zip | f445217354e16497fd568933aa19b7cd74323223c0447b950eb3879590059146 |
| RTX-5060-Ti-Plattformergebnisse      | REDUCTION3(1).zip | 9f0ed4a7bb9ae1576baba96cc021a9c0fc2ffcd7cff649cf6a60e27a143b43fc |
| AMD-Plattformergebnisse              | REDUCTION4(1).zip | 7a20df01e603d665698e1f67380c75348d28b88cbc81c08f1f1b0702ef6784db |
| Intel-Plattformergebnisse            | REDUCTION5(1).zip | 8fba579ef8518b03a6be1e2ce1d5a53e831b11e1ba5a7b2a925eac46010a60a7 |

Die Kampagnenmanifeste binden jede Sitzungsdatei zusätzlich über SHA-256.
Die hochgeladenen Ergebnisarchive enthalten die Analyseoutputs und
Manifeste, jedoch nicht die 8.100 Roh-CSV-Zeilen selbst. Für ein öffentliches
Replikationspaket müssen die Rohdateien und Source-/Runner-Snapshots
zusätzlich beigefügt werden.

---

## 4. Coverage und statistische Struktur

| Plattform | Rohzeilen | Sitzungsmediane | Konfigurationen |
|---|---:|---:|---:|
| Intel | 3.150 | 315 | 63 |
| AMD | 4.050 | 405 | 81 |
| RTX 3090 | 450 | 45 | 9 |
| RTX 5060 Ti | 450 | 45 | 9 |
| **Gesamt** | **8.100** | **810** | **162** |

Zusätzliche finale Outputdimensionen:

```text
Native Policy-Leader:                 180
ausgewählte Policy-Sitzungsmediane:   900
Paarvergleiche:                       270
globale Gewinnerzeilen:                45
Exact-Winner-Regret-Zeilen:            36
```

Das unabhängige Audit reproduzierte:

- sämtliche Konfigurationsmediane;
- alle exakt enumerierten Bootstrapintervalle (`5^5 = 3125`);
- sämtliche Sitzungs-CVs;
- globale Gewinner;
- strikte Paretofronten;
- praktische Paretofronten mit 2-%-Schwelle;
- FLOP-, EDP-, logische-Rate- und GB/J-Identitäten.

Es wurden keine numerischen Hard Failures gefunden.

---

## 5. Finale Validierung und Warnungen

| Ebene       | Warnung                                                            | Bedeutung                                                                     |
|:------------|:-------------------------------------------------------------------|:------------------------------------------------------------------------------|
| Intel       | 3124 in_range, 26 above                                            | Zeitfensterwarnung; alle Checksummen und Formeln PASS                         |
| AMD         | 4022 in_range, 28 above                                            | Zeitfensterwarnung; alle Checksummen und Formeln PASS                         |
| RTX 3090    | 176 minimale kernel_time > e2e_time Abweichungen; max. 0,007246 ms | Cross-clock-Rundung; weit unter harter Toleranz                               |
| RTX 5060 Ti | 27 minimale Cross-clock-Abweichungen; max. 0,003018 ms             | weit unter harter Toleranz                                                    |
| RTX 5060 Ti | Legacy-Headerreihenfolge                                           | nur total_energy_j/device_energy_j positional vertauscht; Spalten vollständig |
| Gesamtaudit | native best ist post-selection/deskriptiv                          | Gewinnerauswahl und Intervall verwenden dieselben fünf Sessions               |
| Gesamtaudit | logische Nutzdatenrate ist kein physischer Traffic                 | 4N+4 schließt interne Partials/CUB-Workspace aus                              |
| Gesamtaudit | Energiedomänen asymmetrisch                                        | CPU Package RAPL vs. GPU Board NVML einschließlich VRAM                       |

Alle Checksummen bestanden. Insbesondere ist der zuvor diskutierte
FP32-Grenzfall bei `N=256M` praktisch erledigt: Die tatsächlichen Kampagnen
bestehen den eingefrorenen Checksum-Gate.

### Endgültige Warnungsinterpretation

- `above` bedeutet nicht fehlerhaft, sondern nur länger als das bevorzugte
  Kalibrierfenster.
- minimale GPU-Abweichungen zwischen CUDA-Event- und Host-Zeit sind
  Cross-clock-Effekte weit unter der materiellen Toleranz
  `max(0,5 ms, 0,5 % von e2e)`.
- der RTX-5060-Ti-Header enthält alle 45 eindeutigen Spalten; nur zwei
  Energiepositionen sind vertauscht.
- die Post-selection-Warnung begrenzt inferenzielle Sprache, nicht die
  deskriptive Gültigkeit.
- logische Rate darf niemals als gemessene physische Bandbreite bezeichnet
  werden.

---

## 6. Gelöste Analysefehler und finale Pipelineversion

Die finale Analyse ist **v1.4**. Die vorherigen Fehler betrafen ausschließlich
Validator- beziehungsweise Provenienzlogik. Keine Rohmessung wurde geändert,
korrigiert, gelöscht oder neu erzeugt.

| Version | Problem | Lösung |
|---|---|---|
| v1.0 | CPU-Formelwerte wurden trotz Scientific-6-Serialisierung gegen ungerundete Werte geprüft | v1.1 reproduziert die Writer-Serialisierung |
| v1.1 | GPU speichert exakte Formelwerte; Parser verlangte CPU-Darstellung und `flops_per_op` | v1.2 akzeptiert strikt exakten oder Scientific-6-Wert und direkte GPU-`flops_total`-Form |
| v1.2 | 5060-Ti-Runner-Großschreibung und reale Headerreihenfolge nicht erkannt | v1.3 akzeptiert exakt den dokumentierten Legacy-Tausch als Warnung |
| v1.3 | CPU-Provenienznormalisierung erkannte reale Bannerunterschiede nicht | v1.4 normalisiert nur Dateiname und Plattformbanner |

Finales Pipelinepaket:

```text
REDUCTION_analysis_all_platforms_v1_4.zip
SHA-256:
33e7742e5e2da7f82d821c9419645fc71d70118e2ae9c28d3488ecc212976d40
```

Erneuter Analyseaufruf:

```bash
cd ~/projects/energy/new
./run_reduction_analysis_all.sh
```

Ergebnisorte:

```text
AMD/results/REDUCTION
INTEL/results/REDUCTION
3090/results/REDUCTION
5060ti/results/REDUCTION
ALL AUDIT/REDUCTION/results
```

---

## 7. Finale nüchterne Platzierungsergebnisse

† = tie-aware nicht eindeutig.

| N    | semantische Daten   | Laufzeit             | Energie             | EDP    | klarer Laufzeit-/Energiekonflikt   |
|:-----|:--------------------|:---------------------|:--------------------|:-------|:-----------------------------------|
| 1M   | 4 MB                | 3090 (5.658 µs)      | 5060ti (0.406 mJ)   | 5060ti | ja                                 |
| 2M   | 8 MB                | 5060ti (8.196 µs)    | 5060ti (0.730 mJ)   | 5060ti | nein                               |
| 4M   | 16 MB               | 5060ti (12.285 µs) † | 5060ti (1.019 mJ)   | 5060ti | nein                               |
| 8M   | 32 MB               | 5060ti (20.508 µs)   | 5060ti (1.820 mJ)   | 5060ti | nein                               |
| 16M  | 64 MB               | 3090 (76.895 µs)     | 5060ti (13.406 mJ)  | 3090 † | ja                                 |
| 32M  | 128 MB              | 3090 (148.772 µs)    | 5060ti (25.730 mJ)  | 3090   | ja                                 |
| 64M  | 256 MB              | 3090 (292.511 µs)    | 5060ti (51.055 mJ)  | 3090   | ja                                 |
| 128M | 512 MB              | 3090 (579.977 µs)    | 5060ti (101.797 mJ) | 3090   | ja                                 |
| 256M | 1024 MB             | 3090 (1.155 ms)      | 5060ti (202.513 mJ) | 3090   | ja                                 |

### Zentrale Zählung

Bei **6 von 9 Größen** existiert ein klarer Geräte-Zielkonflikt zwischen
Laufzeit- und Energieoptimum:

```text
1M, 16M, 32M, 64M, 128M, 256M
```

Die RTX 5060 Ti ist bei **9 von 9 Größen** klarer Energiegewinner in der
gemessenen Device-domain energy.

Diese 9/9-Aussage muss im Paper nach Größenregimen aufgeteilt werden:

```text
N <= 8M:
sehr große, heterogene Energieabstände; effektives Ausführungsregime
unterscheidet sich stark zwischen Plattformen.

N >= 16M:
sehr stabiles großes GPU-Verhältnis von ungefähr 2:1.
```

---

## 8. Stärkster Cross-Device-Befund

Für `N >= 16M`:

| N    |   Laufzeit 5060Ti / 3090 |   Energie 3090 / 5060Ti |   EDP 3090 / 5060Ti |
|:-----|-------------------------:|------------------------:|--------------------:|
| 16M  |                    2.053 |                   1.983 |               0.966 |
| 32M  |                    2.07  |                   2.02  |               0.975 |
| 64M  |                    2.08  |                   2.026 |               0.974 |
| 128M |                    2.085 |                   2.018 |               0.968 |
| 256M |                    2.088 |                   2.024 |               0.97  |

Belastbare Zusammenfassung:

```text
RTX 3090:      2,053–2,088× schneller
RTX 3090:      1,983–2,026× mehr Boardenergie
```

Laufzeit und Energie kompensieren sich fast, aber nicht vollständig.

### Finale EDP-Sprache

- bei 16M: RTX 3090 und RTX 5060 Ti tie-aware nicht eindeutig;
- bei 32M–256M: RTX 3090 besitzt einen kleinen, aber konsistenten
  EDP-Vorteil von ungefähr 2,5–3,4 %;
- wegen der eingefrorenen 2-%-Regel darf ab 32M nicht von praktischer
  EDP-Gleichheit gesprochen werden.

Zulässige Formulierung:

> Runtime and energy nearly, but not completely, compensate: from 32M
> onward, the RTX 3090 retains a small 2.5–3.4% EDP advantage.

---

## 9. Überraschendster Regimebefund

Auf der RTX 5060 Ti zwischen 8M und 16M:

```text
semantische Problemgröße: 2×
Laufzeit:                  7,70×
Energie:                   7,37×
logische Nutzdatenrate:    ungefähr 1560 → 405 GB/s
```

Das ist ein direkt beobachteter Regimewechsel.

### Was beobachtet und was nur vermutet ist

Direkt gemessen:

> The RTX 5060 Ti exhibits a 7.7× runtime increase between 8M and 16M
> for a twofold increase in semantic problem size.

Nur als post-hoc Mechanismushypothese zulässig:

> This discontinuity is consistent with a change in the effective
> memory-hierarchy regime.

Nicht zulässig:

- der Cachegrenzwert sei bewiesen;
- das Array sei nachweislich vollständig cache-resident;
- die logische Rate sei physische VRAM-Bandbreite.

Der residente Wiederholungsmodus war vorab festgelegt. Die konkrete
Cache-/Kapazitätsdeutung wurde jedoch nicht präregistriert und muss als
explorative Interpretation kenntlich bleiben.

### Vergleichbarkeit von N

`N` ist plattformübergreifend semantisch vergleichbar:

```text
gleiche Elementzahl
gleiche Operation
gleiche FLOP-Definition
gleicher logischer Datenanker
```

Dass dasselbe `N` auf verschiedenen Plattformen unterschiedliche
mikroarchitektonische Regime erzeugt, ist kein Messfehler, sondern ein
zentraler Placement-Befund.

---

## 10. Paretofront bei N=16M

| Konfiguration   |   Laufzeit [µs] |   Energie [mJ] |   Leistung [W] |   Runtime-CV [%] |   Energie-CV [%] |
|:----------------|----------------:|---------------:|---------------:|-----------------:|-----------------:|
| RTX 3090        |          76.895 |         26.59  |          345.9 |            0.019 |            0.567 |
| AMD CPU 32T     |          97.506 |         22.392 |          230   |            0.336 |            0.247 |
| AMD CPU 16T     |         120.186 |         22.059 |          183   |            0.544 |            0.403 |
| RTX 5060 Ti     |         157.843 |         13.406 |           84.9 |            0.021 |            3.19  |

Interpretation:

```text
RTX 3090:    Laufzeitpol
AMD 32T:     schneller CPU-Kompromiss
AMD 16T:     energieärmerer CPU-Kompromiss
RTX 5060 Ti: Energiepol
```

Alle vier Punkte sind sitzungsstabil. Dies ist die einzige gemessene Größe,
bei der CPU- und GPU-Konfigurationen gemeinsam eine robuste mehrgliedrige
Front bilden.

---

## 11. CPU-Threadkonflikte und Race-to-idle

| Plattform   | N    | Laufzeitoptimum   | Energieoptimum   |   mehr Energie [%] |   Laufzeitgewinn [%] | Energieklassifikation               | Laufzeitklassifikation              |
|:------------|:-----|:------------------|:-----------------|-------------------:|---------------------:|:------------------------------------|:------------------------------------|
| INTEL       | 16M  | 8T                | 4T               |              31.62 |                 5.14 | clear_energy_opt                    | clear_runtime_opt                   |
| INTEL       | 32M  | 8T                | 4T               |              30.04 |                 3.01 | clear_energy_opt                    | clear_runtime_opt                   |
| INTEL       | 64M  | 8T                | 4T               |              36.8  |                 1.33 | clear_energy_opt                    | practically_equivalent_or_uncertain |
| INTEL       | 128M | 8T                | 4T               |              36.91 |                 0.82 | clear_energy_opt                    | practically_equivalent_or_uncertain |
| INTEL       | 256M | 8T                | 4T               |              39.45 |                 0.24 | clear_energy_opt                    | practically_equivalent_or_uncertain |
| AMD         | 16M  | 32T               | 16T              |               1.51 |                18.87 | practically_equivalent_or_uncertain | clear_runtime_opt                   |
| AMD         | 128M | 16T               | 4T               |              25.48 |                13.67 | clear_energy_opt                    | clear_runtime_opt                   |
| AMD         | 256M | 16T               | 4T               |              30.99 |                12.05 | clear_energy_opt                    | clear_runtime_opt                   |

### Stärkster Within-platform-Claim

Bei Intel `N=256M`:

```text
Laufzeitoptimum: 8T
Energieoptimum:  4T
Laufzeitgewinn:  0,24 %
mehr Package-Energie: 39,45 %
gepaarte Laufzeitratio: 0,9976
95-%-Intervall: ungefähr [0,9956; 1,0001]
gepaarte Energieratio: 1,3945
95-%-Intervall: ungefähr [1,357; 1,553]
```

Die Laufzeit ist praktisch beziehungsweise statistisch nicht
unterscheidbar, die Energie dagegen klar schlechter.

Zulässige Kernaussage:

> Once bandwidth saturation is reached, additional CPU threads can increase
> package energy without producing a statistically distinguishable runtime
> improvement.

### AMD

Bei AMD 128M und 256M besteht dagegen ein echter Trade-off:

```text
128M: 16T statt 4T → 13,67 % schneller, 25,48 % mehr Energie
256M: 16T statt 4T → 12,05 % schneller, 30,99 % mehr Energie
```

Bei AMD 16M ist Race-to-idle nahezu günstig:

```text
32T statt 16T → 18,87 % schneller, nur 1,51 % mehr Energie
```

Finale Interpretation:

> Race-to-idle besitzt bei REDUCTION kein universelles Vorzeichen. Es hängt
> von Plattform, Working Set und Parallelitätsregime ab.

---

## 12. Stabilität, Ausreißer und AMD 4M

| Plattform   | Konfigurationen mit CV >5 %   |   Anteil [%] | markierte Rohzeilen, nicht entfernt   |
|:------------|:------------------------------|-------------:|:--------------------------------------|
| RTX 3090    | 0/9                           |          0   | 31/450 (6.89 %)                       |
| RTX 5060 Ti | 0/9                           |          0   | 43/450 (9.56 %)                       |
| AMD         | 27/81                         |         33.3 | 343/4050 (8.47 %)                     |
| Intel       | 30/63                         |         47.6 | 309/3150 (9.81 %)                     |

Die GPUs sind über Sitzungen sehr stabil. Dabei ist zu beachten, dass pro
GPU und Größe nur eine Konfiguration existiert; die CPUs besitzen ein
deutlich größeres Threadraster.

Markierte Rohzeilen wurden nicht entfernt.

### Kritischer Sonderfall: AMD 4M/64T

|   Session |   Laufzeit [µs] |   Energie [mJ] |   Leistung [W] |   Temperatur [°C] |
|----------:|----------------:|---------------:|---------------:|------------------:|
|         1 |         114.128 |         18.351 |          160.9 |              53.5 |
|         2 |          10.793 |          3.071 |          284.3 |              71.5 |
|         3 |          11.482 |          3.194 |          278.2 |              68   |
|         4 |          16.054 |          3.917 |          244.1 |              66   |
|         5 |         116.961 |         18.751 |          160.3 |              55.5 |

```text
Runtime-CV: ungefähr 104,5 %
Energie-CV: ungefähr 87,9 %
```

Die Konfiguration ist bimodal. Der Median ist rechnerisch korrekt, aber kein
stabiler Hardwarecharakterisierungswert.

### Verbindliche Propagation

AMD `4M/64T` darf:

- nicht als stabiler Sieger formuliert werden;
- nicht als Beleg verwendet werden, AMD sei bei 4M eindeutig GPU-konkurrenzfähig;
- nicht gelöscht oder durch Sessionauswahl „repariert“ werden.

Vor dem Paper müssen mindestens folgende abgeleitete Darstellungen ein
Instabilitätsflag beziehungsweise eine Fußnote erhalten:

```text
best_cpu_vs_best_gpu.csv, N=4M
placement_by_size.csv, N=4M
jede Hauptfigur oder Tabelle mit AMD 4M/64T
```

Zulässige Formulierung:

> At 4M, AMD enters the uncertainty set only through a bimodal 64-thread
> configuration; its point median remains 30.7% slower than the RTX 5060 Ti
> and is not treated as a stable placement option.

---

## 13. Endgültig geklärte Auditstreitpunkte

| Streitpunkt | Finale Entscheidung |
|---|---|
| Package-only auf Intel und AMD | bewusst und korrekt |
| Intel-DRAM | nur Sensitivität; weder primäre Metrik noch als defekt bewiesen |
| GPU Board Energy inkl. VRAM | korrekt; Scope offen benennen |
| `power_w` | kein Rechenfehler; Medianisierungsreihenfolge dokumentieren |
| N-Vergleich | semantisch vergleichbar |
| identisches mikroarchitektonisches Regime | nicht garantiert; gerade deshalb Placement-relevant |
| Cachemechanismus | plausible post-hoc Hypothese, nicht bewiesen |
| EDP „praktisch gleich“ | bei 16M unsicher; ab 32M nach 2-%-Regel falsch |
| AMD 4M/64T | gültige, aber instabile Daten; Flag muss propagiert werden |
| Runtime-/Checksum-Acceptance | durch vollständige Validatorarchive belegt |
| vollständiger Rerun | nicht erforderlich |
| Multi-Buffer-Kontrolle | optionaler Mechanismustest, keine Rettungsmessung |

---

## 14. Belastbare Paper-Claims

### 14.1 Cross-device

> Across six of nine problem sizes, the runtime and energy leaders are
> distinct devices.

> The RTX 5060 Ti is the clear measured board-energy winner at all nine
> sizes.

> For N≥16M, the RTX 3090 is 2.05–2.09× faster than the RTX 5060 Ti,
> whereas the RTX 5060 Ti uses approximately half the board energy.

> At N=16M, the strict runtime–energy Pareto front contains four stable
> configurations across three hardware platforms.

### 14.2 Within-platform

> At N=256M on Intel, the runtime-optimal 8-thread configuration improves
> runtime by only 0.24% over the energy-optimal 4-thread configuration while
> increasing package energy by 39.45%.

> AMD exhibits robust runtime–energy thread-count trade-offs at 128M and
> 256M.

> Race-to-idle changes character across working-set and platform regimes.

### 14.3 Regimebeobachtung

> The RTX 5060 Ti exhibits a 7.7× runtime increase between 8M and 16M for
> only a twofold increase in semantic problem size.

Mechanismussprache nur mit:

```text
consistent with
suggests
hypothesis
requires a multi-buffer or counter-based control
```

---

## 15. Nicht zulässige oder überzogene Claims

Nicht behaupten:

- `4N+4` sei gemessener DRAM-/VRAM-Verkehr;
- logische GB/s seien physische Speicherbandbreite;
- Cachekausalität sei nachgewiesen;
- kleine N seien wertlos oder nicht vergleichbar;
- GPU-Energie umfasse Host, PCIe oder Whole-system;
- CPU-Package und GPU-Board seien identische Grenzen;
- 50 technische Wiederholungen seien 50 unabhängige Beobachtungen;
- AMD 4M/64T sei ein stabiler Gewinner;
- EDP sei ab 32M praktisch gleich;
- Durchsatz und Laufzeit seien unabhängige Evidenz;
- logische GB/J und Energie seien unabhängige Evidenz;
- Intel-DRAM sei nachweislich defekt;
- ein einheitlicher Mechanismus erkläre den 5060-Ti-Energievorteil über
  alle neun Größen.

---

## 16. Verbindliche Reportingpatches vor einer Einreichung

1. **EDP-Text korrigieren:** ab 32M kleiner 3090-Vorteil statt praktischer
   Gleichheit.
2. **AMD 4M propagieren:** Instabilitätsflag in Vergleichstabellen und
   Hauptfiguren.
3. **Energiegewinn nach Regimen trennen:** N≤8M und N≥16M nicht mit einem
   einheitlichen Mechanismus erklären.
4. **`power_w` dokumentieren:** Median der Rohverhältnisse, nicht Quotient
   der Mediane.
5. **Energiedomänen offen nennen:** CPU Package versus GPU Board.
6. **logische Rate umbenennen:** „logical useful-data rate“ beziehungsweise
   „logische Nutzdatenrate“.
7. **Unsicherheit sichtbar machen:** CIs oder Stabilitätsmarker in
   Hauptfiguren.
8. **x-Achse korrekt beschriften:** `1M ... 256M` oder `4 MB ... 1024 MB`,
   nicht `2^20 ... 2^28`.
9. **Post-selection kenntlich machen:** Exact-winner-Regret ist
   deskriptiv; gepaarte Sitzungsanalyse als Sensitivität.
10. **Ausreißerpolitik nennen:** markiert, nicht entfernt.

Diese Punkte erfordern keinen Rerun und keine Änderung der Rohdaten.

---

## 17. Paperwert und empfohlene Story

### Finale Kernthese

> Hardware selection for reduction is not a ranking problem; it is a
> working-set-, objective-, and configuration-dependent regime problem.

### Stärkste Ergebnisreihenfolge

1. **Intel-Sättigung:** nahezu gleiche Laufzeit, bis zu stark unterschiedliche
   Energie — zusätzliche Threads kaufen keine Geschwindigkeit.
2. **Große GPU-Front:** ungefähr 2× Geschwindigkeit gegen ungefähr
   2× Energie über fünf Größen hinweg.
3. **16M-Paretofront:** RTX 3090, zwei AMD-Konfigurationen und RTX 5060 Ti.
4. **5060-Ti-Regimebruch:** 7,7× Laufzeitsprung bei doppeltem N.
5. **Race-to-idle kippt:** Intel groß versus AMD 16M.

### Beziehung zu STREAM

STREAM und REDUCTION teilen bei großen Größen die Front:

```text
RTX 3090: schneller
RTX 5060 Ti: energieärmer
```

REDUCTION liefert darüber hinaus:

- hierarchische Aggregation;
- einen abrupten Mittelgrößen-Regimewechsel;
- eine robuste CPU-Zwischenfront bei 16M;
- deutlich wechselnde Threadzahl-Regrets;
- den besonders sauberen Intel-Sättigungsbefund.

Die gemeinsame übergeordnete These kann lauten:

> Arithmetic intensity alone is insufficient. Working-set topology,
> access pattern, reduction hierarchy, and optimization objective jointly
> determine rational CPU/GPU placement.

---

## 18. Optionales Zusatzexperiment mit höchstem Nutzen

### Multi-Buffer-Kontrolle

K unabhängige Arrays mit einer Gesamtgröße deutlich oberhalb der relevanten
Cacheebene anlegen. Batch `i` reduziert Array `i mod K`.

Ziel:

- residente Wiederverwendung gezielt unterbrechen;
- den 8M→16M-Regimebruch mechanistisch testen;
- kleine N nicht „retten“, sondern den Mechanismus eines bereits gültigen
  Placement-Befunds bestätigen oder widerlegen.

Minimalraster:

```text
N = 4M und 8M auf allen vier Plattformen
optional RTX 5060 Ti zusätzlich 6M, 10M, 12M, 14M, 16M, 20M
CPU mit relevantem Threadraster
```

Nützliche Counter:

```text
L2-/LLC-Hitrate
tatsächliche DRAM-/VRAM-Read-Bytes
Affinity-/Topologiezustand
```

### AMD-Bimodalität

Sekundärer kleiner Versuch:

```text
AMD 4M/64T
10 Sessions
Affinity, NUMA-/CCD-/CCX-Zuordnung und Frequenzzustand protokollieren
```

Beide Experimente sind optional. Die vorhandene Kampagne bleibt ohne sie
gültig.

---

## 19. Reproduktions- und Archivhinweise

### Finale Kernartefakte

```text
REDUCTION_analysis_all_platforms_v1_4.zip
REDUCTION_INDEPENDENT_AUDIT_BUNDLE_2026-07-23.zip
REDUCTION(1).zip
REDUCTION2(1).zip
REDUCTION3(1).zip
REDUCTION4(1).zip
REDUCTION5(1).zip
```

### Unabhängiges Audit

```text
REDUCTION_INDEPENDENT_AUDIT_2026-07-23.md
SHA-256:
7134f8348a05eeb069a8b5c9f18f97bae08c6a1e1308b675feafe8fd23a6b481

REDUCTION_INDEPENDENT_AUDIT_BUNDLE_2026-07-23.zip
SHA-256:
3e8960d8d03f2847386550eb6af55b7fbf15d880fcba7aea540037446a362261
```

### Für ein öffentliches Freeze-Paket noch ergänzen

- die 8.100 Roh-CSV-Zeilen;
- gebundene CPU-/GPU-Sourcebytes;
- gebundene Runnerbytes;
- Hardware- und Softwaremetadaten;
- dieses Abschlussdokument;
- finale Analyse v1.4;
- vollständige SHA-256-Datei.

---

## 20. Schlussentscheidung

# **REDUCTION IST ABGENOMMEN UND FÜR DIE PAPERANALYSE FREIGEGEBEN.**

Verbindlich:

```text
keine vollständige Neumessung
keine stille Rohdatenkorrektur
keine Ausreißerentfernung
keine erneute Diskussion der eingefrorenen Energiedomäne
keine Cachekausalität ohne Zusatzmessung
AMD 4M als instabil propagieren
EDP-Sprache korrigieren
```

Der Workload liefert eigenständigen Paperwert und ist nicht nur eine
Bestätigung von STREAM. Besonders stark sind der Intel-Sättigungsbefund,
die stabile große Zwei-GPU-Front, die 16M-Paretofront und der abrupte
RTX-5060-Ti-Regimewechsel.

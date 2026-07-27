# Unabhängiges Gesamtaudit — REDUCTION

**Datum:** 23. Juli 2026  
**Workload:** `sum(x[0:N]) -> FP32-Skalar`  
**Plattformen:** Intel i9-7900X, AMD Threadripper 3970X, RTX 3090, RTX 5060 Ti

## Gesamturteil

# **PASS WITH REPORTING AND ARCHIVAL QUALIFICATIONS**

Die hochgeladenen Ergebnisarchive sind intern konsistent. Alle vier
Plattformvalidatoren sind ohne harten Fehler abgeschlossen; auch der
gemeinsame Preflight und der Integrity-Audit enthalten keinen harten
Fehler. Die zentralen Formeln, Konfigurationsstatistiken, globalen
Gewinner, Paretofronten und Dimensionszählungen wurden unabhängig aus den
gelieferten CSV-Ergebnissen nachgerechnet.

**Eine vollständige Neumessung der REDUCTION-Kampagnen ist durch dieses
Audit nicht begründet.**

Der wichtigste Scope-Vorbehalt: Die fünf Archive enthalten die
8.100 Rohmesszeilen und die gebundenen Source-/Runnerbytes nicht selbst.
Sie enthalten Manifeste und Hashes, Validierungsberichte,
Sitzungsmediane sowie aggregierte Ergebnisse. Dies ist deshalb ein
vollständiges **Ergebnis- und Analyseaudit**, aber kein byteweises
Rohdaten-/Source-Replay.

## 1. Auditbasis

| Plattform | Rohzeilen laut Manifest | Sitzungsmediane | Konfigurationen | Kampagne |
|---|---:|---:|---:|---|
| Intel | 3.150 | 315 | 63 | 20260721_164732 |
| AMD | 4.050 | 405 | 81 | 20260722_174206 |
| RTX 3090 | 450 | 45 | 9 | 20260722_191930 |
| RTX 5060 Ti | 450 | 45 | 9 | 20260722_050953 |
| **Gesamt** | **8.100** | **810** | **162** | |

Die statistische Einheit ist korrekt:

```text
10 technische Wiederholungen
→ ein Sitzungsmedian
→ fünf Sitzungsmediane pro Konfiguration
```

Es wird nicht mit einem künstlichen `n=50` argumentiert.

## 2. Unabhängig nachgerechnete Integrität

- 810 Sitzungsmediane: **PASS**
- 162 Konfigurationszusammenfassungen: **PASS**
- 180 native Policy-Leader: **PASS**
- 900 ausgewählte Policy-Sitzungsmediane: **PASS**
- 270 Paarvergleiche: **PASS**
- 45 globale Gewinnerzeilen: **PASS**
- 36 Exact-Winner-Regret-Zeilen: **PASS**
- exakt fünf Sitzungsmediane je Konfiguration: **PASS**
- alle Mediane, Bootstrapintervalle und CV-Werte: **PASS**
- exakte globale Gewinner: **PASS**
- strikte und praktische 2-%-Paretofronten: **PASS**

Maximale Rechenabweichungen:

| Identität | maximale absolute Abweichung |
|---|---:|
| EDP = Laufzeit × Energie | 9.926e-17 |
| Durchsatz = `(N-1)/t` | 5.684e-14 |
| Effizienz = `(N-1)/E` | 4.441e-16 |
| logische Nutzdatenrate = `(4N+4)/t` | 2.274e-13 |
| logische GB/J = `(4N+4)/E` | 1.776e-15 |

Das sind nur Gleitkomma-Rundungsreste.

## 3. Nüchterne Hauptergebnisse

### 3.1 Globale Platzierung

|         N | Daten   | Laufzeit            | Energie             | EDP    | Konflikt   |
|----------:|:--------|:--------------------|:--------------------|:-------|:-----------|
|   1000000 | 4 MB    | 3090 (5.658 µs)     | 5060ti (0.406 mJ)   | 5060ti | ja         |
|   2000000 | 8 MB    | 5060ti (8.196 µs)   | 5060ti (0.730 mJ)   | 5060ti | nein       |
|   4000000 | 16 MB   | 5060ti (12.285 µs)* | 5060ti (1.019 mJ)   | 5060ti | nein       |
|   8000000 | 32 MB   | 5060ti (20.508 µs)  | 5060ti (1.820 mJ)   | 5060ti | nein       |
|  16000000 | 64 MB   | 3090 (76.895 µs)    | 5060ti (13.406 mJ)  | 3090*  | ja         |
|  32000000 | 128 MB  | 3090 (148.772 µs)   | 5060ti (25.730 mJ)  | 3090   | ja         |
|  64000000 | 256 MB  | 3090 (292.511 µs)   | 5060ti (51.055 mJ)  | 3090   | ja         |
| 128000000 | 512 MB  | 3090 (579.977 µs)   | 5060ti (101.797 mJ) | 3090   | ja         |
| 256000000 | 1024 MB | 3090 (1.155 ms)     | 5060ti (202.513 mJ) | 3090   | ja         |

`*` bezeichnet eine tie-aware unsichere Auswahl. Der Punktgewinner darf
dort nicht als eindeutig überlegen formuliert werden.

Bei **6 von 9 Größen** sind der klare schnellste und der klare
energieärmste Gerätetyp verschieden:

```text
N = 1M sowie N = 16M, 32M, 64M, 128M und 256M
```

Die RTX 5060 Ti ist bei allen neun Größen der klare Energiegewinner. Ab
16M ist die RTX 3090 der klare Laufzeitgewinner. Bei 2M bis 8M dominiert
die RTX 5060 Ti weitgehend alle Ziele; bei 4M ist ihre Laufzeitführung
gegenüber AMD tie-aware unsicher.

### 3.2 Fast perfekte GPU-Laufzeit-/Energie-Kompensation

Für `N >= 16M` gilt im Mittel:

- RTX 3090 gegenüber RTX 5060 Ti: **2.075× schneller**
- RTX 3090 gegenüber RTX 5060 Ti: **2.014× Energie**
- `EDP_3090 / EDP_5060Ti`: **0.966 bis 0.975**

Die RTX 5060 Ti benötigt also ungefähr die halbe Energie, aber ungefähr die
doppelte Laufzeit. Dadurch ist EDP fast ausgeglichen; der Punktwert
bevorzugt die RTX 3090 nur um ungefähr 2,5–3,4 %. Bei 16M ist selbst diese
EDP-Auswahl tie-aware nicht eindeutig.

Das ist ein stärkerer Befund als ein einfacher Gerätesieger: **Je nach
Ziel sind zwei gegensätzliche Platzierungen gleichzeitig rational.**

### 3.3 Abrupter RTX-5060-Ti-Regimewechsel

| Größe | semantische Daten | Laufzeit | Energie | logische Nutzdatenrate |
|---|---:|---:|---:|---:|
| 8M | 32 MB | 20.508 µs | 1.820 mJ | 1560.3 GB/s |
| 16M | 64 MB | 157.843 µs | 13.406 mJ | 405.5 GB/s |

Bei nur doppelter Datenmenge:

- Laufzeit: **7.697×**
- Energie: **7.366×**
- logische Nutzdatenrate: **0.260×**

Das ist ein empirischer Regimebruch. Cachekapazität, CUB-Hierarchie und
interner Speicherverkehr sind plausible Mechanismen, aber ohne
Hardwarecounter noch nicht kausal nachgewiesen.

### 3.4 Dreifache Hardware-Paretofront bei N=16M

| Konfiguration   |   Laufzeit [µs] |   Energie [mJ] |   Leistung [W] |   Runtime-CV [%] |   Energie-CV [%] |
|:----------------|----------------:|---------------:|---------------:|-----------------:|-----------------:|
| RTX 3090        |          76.895 |         26.59  |          345.9 |            0.019 |            0.567 |
| AMD CPU 32T     |          97.506 |         22.392 |          230   |            0.336 |            0.247 |
| AMD CPU 16T     |         120.186 |         22.059 |          183   |            0.544 |            0.403 |
| RTX 5060 Ti     |         157.843 |         13.406 |           84.9 |            0.021 |            3.19  |

- RTX 3090: Laufzeitpol
- RTX 5060 Ti: Energiepol
- AMD 16T/32T: zwei echte Zwischenlösungen

Das ist publizierbarer als ein bloßer CPU-vs.-GPU-Sieger, weil drei
Hardwareklassen auf derselben Front liegen.

### 3.5 Beste CPU gegen beste GPU

`beste CPU-Energie / beste GPU-Energie`:

```text
3,81×; 5,31×; 3,84×; 6,77×; 1,65×; 3,49×; 9,36×; 9,14×; 8,97×
```

Bei der Laufzeit ist die beste CPU:

- bei 4M punktweise nur 1,31× langsamer als die RTX 5060 Ti, aber wegen
  AMD-Instabilität statistisch unsicher;
- bei 16M 1,27× langsamer als die RTX 3090;
- ab 64M etwa 15,3–15,7× langsamer als die RTX 3090.

Die belastbare Aussage lautet: CPU kann an einer Übergangsgröße Teil der
Paretofront sein, wird bei großen residenten Reduktionen aber klar von
GPUs verdrängt.

## 4. Threadzahl und Race-to-idle

### Klare tie-aware AMD-Konflikte

| N | Laufzeitoptimum | Energieoptimum | Energieaufschlag | Laufzeitgewinn |
|---:|---:|---:|---:|---:|
| 128M | 16T | 4T | 25,48 % | 13,67 % |
| 256M | 16T | 4T | 30,99 % | 12,05 % |

### Besonders stark: Intel bei großen Größen

| N | Laufzeitoptimum | Energieoptimum | mehr Energie | Laufzeitgewinn |
|---:|---:|---:|---:|---:|
| 64M | 8T | 4T | 36,80 % | 1,33 % |
| 128M | 8T | 4T | 36,91 % | 0,82 % |
| 256M | 8T | 4T | 39,45 % | 0,24 % |

Das ist ein klarer Gegenbefund zu einer pauschalen Race-to-idle-Heuristik:
Mehr Threads sparen fast keine Zeit, erhöhen aber die Paketenergie stark.

Race-to-idle ist dennoch nicht generell falsch. Bei AMD 16M liefert 32T
gegenüber 16T rund 18,87 % Zeitgewinn bei nur 1,51 % Energieaufschlag.

> Race-to-idle ist bei REDUCTION selbst ein größen- und
> plattformspezifisches Regime.

## 5. Statistische Sensitivität

### Gepaarte Sitzungen

Die vorhandene Regret-Tabelle bootstrapped die beiden Konfigurationen
unabhängig. Ich habe ergänzend die ausgerichteten Sitzungsnummern gepaart
und alle `5^5 = 3125` Resamples enumeriert.

- Kein vorhandener klarer Energieaufschlag wird abgeschwächt.
- Intel 8M wird zusätzlich klar: 11,45 % mehr Energie für 20,16 %
  Zeitgewinn.
- Die großen Intel- und AMD-Konflikte bleiben erhalten.

Für das Paper sollte die gepaarte Analyse die primäre
Within-platform-Sensitivität sein. Sie bleibt wegen der Gewinnerauswahl
aus denselben Daten deskriptiv/post-selection.

### Leave-one-session-out

Robust in allen fünf Folds:

- Intel 16M–256M: Laufzeit 8T, Energie 4T
- AMD 128M/256M: Laufzeit 16T, Energie 4T

Nicht robust bei AMD 4M:

- Laufzeit: 16T einmal, 20T zweimal, 64T zweimal
- Energie: 8T dreimal, 64T zweimal
- EDP: drei verschiedene Konfigurationen

Der exakte AMD-4M-Punktgewinner ist daher keine stabile
Hardwareeigenschaft.

## 6. Stabilität und Ausreißer

| platform   |   configurations |   runtime_unstable_configs |   energy_unstable_configs |   any_unstable_configs |   any_unstable_pct |   flagged_raw_rows |   raw_rows |   flagged_pct | removed   |
|:-----------|-----------------:|---------------------------:|--------------------------:|-----------------------:|-------------------:|-------------------:|-----------:|--------------:|:----------|
| 3090       |                9 |                          0 |                         0 |                      0 |                0   |                 31 |        450 |          6.89 | False     |
| 5060ti     |                9 |                          0 |                         0 |                      0 |                0   |                 43 |        450 |          9.56 | False     |
| AMD        |               81 |                         17 |                        26 |                     27 |               33.3 |                343 |       4050 |          8.47 | False     |
| INTEL      |               63 |                          5 |                        30 |                     30 |               47.6 |                309 |       3150 |          9.81 | False     |

57 von 162 Konfigurationen überschreiten 5 % Sitzungs-CV bei Laufzeit oder
Energie. Die GPU-Konfigurationen sind vollständig stabil; die Instabilität
liegt auf den CPUs.

Der kritischste Fall ist AMD 4M/64T:

|   Session |   Laufzeit [µs] |   Energie [mJ] |   Leistung [W] |   Temperatur [°C] |
|----------:|----------------:|---------------:|---------------:|------------------:|
|         1 |         114.128 |         18.351 |          160.9 |              53.5 |
|         2 |          10.793 |          3.071 |          284.3 |              71.5 |
|         3 |          11.482 |          3.194 |          278.2 |              68   |
|         4 |          16.054 |          3.917 |          244.1 |              66   |
|         5 |         116.961 |         18.751 |          160.3 |              55.5 |

Die Verteilung ist bimodal: drei schnelle und zwei ungefähr zehnmal
langsamere Sitzungen. Der Median ist berechenbar, aber der exakte
64T-Gewinner ist nicht robust.

Die Rohzeilen-Ausreißer wurden nur markiert und nicht entfernt. Die
Flag-Quote liegt zwischen 6,89 % und 9,81 %. Das ist kein Ausschlussgrund,
muss aber mit Definition und No-removal-Regel berichtet werden.

## 7. Energiegrenze

Primär verglichen werden:

```text
Intel/AMD: CPU package RAPL
RTX 3090/5060 Ti: NVML board energy einschließlich VRAM
```

Das ist Device-domain energy, nicht Whole-system-Energie. Bei Intel hebt
Package+DRAM den Median im Mittel nur um
0.746 %, im Median um
0.965 % und maximal um
1.359 % an. Die globale
Energieplatzierung ändert sich dadurch nicht.

## 8. Figuren- und Reporting-Audit

1. Konfidenzintervalle oder Stabilitätsmarker ergänzen.
2. Die x-Achse darf nicht `2^20 ... 2^28` zeigen, weil dezimale Größen
   `1M ... 256M` gemessen wurden.
3. `logical_bandwidth_gb_s` als **logische Nutzdatenrate** bezeichnen.
4. Laufzeit und logische Rate nicht als unabhängige Evidenz zählen;
   ebenso Energie und logische GB/J.
5. Eine Regimekarte und die 16M-Paretofront sollten Hauptfiguren werden.

## 9. Was für das Paper „mega interessant“ ist

### Kernstory

> **Hardware selection for reduction is not a ranking problem; it is a
> working-set-, objective-, and configuration-dependent regime problem.**

Vier Regime:

1. **4 MB:** RTX 3090 schnell, RTX 5060 Ti energiearm.
2. **8–32 MB:** RTX 5060 Ti dominiert nahezu alle Ziele.
3. **64 MB:** RTX 3090, RTX 5060 Ti und AMD bilden eine mehrgliedrige Front.
4. **ab 128 MB:** stabile Zwei-GPU-Front; ungefähr 2× Speed gegen ungefähr
   2× Energie, EDP fast gleich.

### Drei besonders starke Beiträge

**A. Fast exakte Kompensation:** Doppelte Geschwindigkeit kostet fast
doppelte Energie; EDP kann den eigentlichen Zielkonflikt verdecken.

**B. 32→64-MB-Cliff:** Kein sanfter Bandbreitenverlauf, sondern abrupter
Regimewechsel auf der RTX 5060 Ti.

**C. Race-to-idle wechselt das Vorzeichen:** Intel groß ist stark
energieineffizient, AMD 16M dagegen nahezu kostenlos schneller.

### Verbindung zu STREAM

STREAM zeigte bei großen Größen bereits „RTX 3090 schneller, RTX 5060 Ti
energieärmer“. REDUCTION reproduziert diese große Front, fügt aber
hierarchische Aggregation, einen abrupten Mittelgrößen-Cliff, eine
CPU-Zwischenfront und wechselnde Thread-Regrets hinzu.

Die stärkere gemeinsame These lautet:

> Arithmetic intensity alone is insufficient. Working-set topology,
> access pattern, reduction hierarchy, and optimization objective jointly
> determine rational placement.

## 10. Claim-Grenzen

### Belastbar

- sechs klare Geräte-Zielkonflikte;
- RTX 5060 Ti bei allen Größen energieärmster Device-domain-Winner;
- RTX 3090 ab 16M ungefähr 2,05–2,09× schneller;
- große GPU-Energieratio ungefähr 1,98–2,03×;
- 16M-Paretofront über GPU- und CPU-Konfigurationen;
- große Intel-Energieaufschläge bei minimalem Zeitgewinn;
- robuste AMD-Konflikte bei 128M/256M.

### Nur Hypothese

- Cache-/Kapazitätswechsel verursacht den 5060-Ti-Cliff;
- CPU-Knickpunkte werden durch LLC-Grenzen verursacht;
- thermischer Zustand erklärt Intel-Energievarianz.

### Nicht behaupten

- `4N+4` sei gemessener DRAM-/VRAM-Verkehr;
- logische Nutzdatenrate sei physische Speicherbandbreite;
- Cachekausalität sei bewiesen;
- GPU-Energie enthalte PCIe, Host oder Whole-system;
- CPU-Package und GPU-Board seien identische Systemgrenzen;
- 50 technische Reps seien 50 unabhängige Beobachtungen;
- AMD 4M/64T sei ein stabiler Gewinner;
- EDP, Bandbreite und GB/J seien unabhängige Messdimensionen.

## 11. Kleinstes Zusatzexperiment mit größtem Wert

Kein kompletter Rerun. Stattdessen:

- RTX 5060 Ti: `N = 6M, 8M, 10M, 12M, 14M, 16M, 20M`
- AMD: zusätzliche Punkte um 16M–64M
- Intel: zusätzliche Punkte zwischen 2M und 4M
- L2-/LLC-Hitrate und tatsächliche DRAM-/VRAM-Read-Bytes
- AMD 4M/8M/16M mit 64T gezielt wiederholen und Affinität/Topologie loggen

Damit wird aus einer plausiblen Cacheinterpretation ein mechanistisch
gestützter Beitrag.

## 12. Endbewertung

**Messkampagne:** PASS.  
**Analyse:** PASS WITH REPORTING QUALIFICATIONS.  
**Publikationswert:** hoch, besonders zusammen mit STREAM.  
**Wichtigster Vorbehalt:** öffentliches Replikationspaket um Roh-CSV und
Source-/Runner-Snapshots ergänzen; Unsicherheit in Hauptfiguren sichtbar
machen.

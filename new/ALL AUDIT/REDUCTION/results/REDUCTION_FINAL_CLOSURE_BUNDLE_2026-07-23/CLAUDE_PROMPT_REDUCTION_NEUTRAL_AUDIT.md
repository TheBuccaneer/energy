# Auftrag: Nüchternes unabhängiges Zahlen-Audit der REDUCTION-Ergebnisse

Du erhältst 20 Dateien zu einer CPU/GPU-Messkampagne für den Workload:

```text
sum(x[0:N]) -> ein FP32-Skalar
```

Plattformen:

```text
Intel Core i9-7900X
AMD Ryzen Threadripper 3970X
NVIDIA GeForce RTX 3090
NVIDIA GeForce RTX 5060 Ti
```

## Deine Rolle

Arbeite als unabhängiger Auditor eines empirischen Systems-Papers. Untersuche
die Zahlen nüchtern und rechne zentrale Ergebnisse selbst nach. Übernimm
weder die vorhandene Ergebnisstory noch die Klassifikationen ungeprüft.

Es geht ausdrücklich nicht darum, freundlich zu bestätigen, was die Autoren
gern zeigen möchten. Finde Rechenfehler, unstabile Gewinner, überzogene
Interpretationen, versteckte Zielkonflikte und tatsächlich starke Befunde.

## Verbindlicher Messvertrag

Diese Punkte sind eingefroren und nicht als Fehler zu beanstanden:

### Statistische Einheit

```text
10 technische Wiederholungen innerhalb einer Sitzung
-> ein Sitzungsmedian
-> fünf Sitzungsmediane je Konfiguration
```

Die inferenzielle Einheit ist `n=5`, nicht `n=50`.

### Primäre Energiedomäne

```text
Intel CPU: Package RAPL
AMD CPU:   Package RAPL
RTX 3090:  NVML Board Energy einschließlich VRAM
RTX 5060:  NVML Board Energy einschließlich VRAM
```

Die separate Intel-DRAM-RAPL-Zone ist bewusst **nicht** Bestandteil der
primären plattformübergreifenden Energiegröße. Sie darf nur als
Intel-interne Sensitivität betrachtet werden. Beanstande diese Entscheidung
nicht als Pipelinefehler.

Die verbleibende, offen zu benennende Grenzasymmetrie ist:

```text
CPU: Package ohne externes DRAM
GPU: Board einschließlich VRAM
```

Das ist eine Limitation der Messgrenzen, keine fehlerhafte Datenaufbereitung.

### Arbeits- und Datenmodell

```text
FLOPs pro Operation:          N - 1
logische Bytes pro Operation: 4*N + 4
GPU-Modus:                    gpu_resident
```

`4*N+4` ist ein semantischer Datenanker und **kein gemessener physischer
DRAM-/VRAM-Verkehr**. Die daraus gebildete Größe ist eine logische
Nutzdatenrate, keine physische Speicherbandbreite.

## Priorität der Dateien

Nutze als primäre Zahlenbasis:

1. `unified_session_medians.csv`
2. `unified_configuration_summary.csv`

Nutze die folgenden Dateien als abgeleitete Ergebnisse, die du kontrollieren
sollst, nicht als unfehlbare Wahrheit:

- `native_policy_leaders.csv`
- `native_policy_session_medians.csv`
- `pairwise_native_best_comparisons.csv`
- `all_configuration_pareto.csv`
- `all_platform_metric_winners.csv`
- `placement_by_size.csv`
- `within_platform_exact_winner_regret.csv`
- `within_platform_energy_runtime_tradeoffs.csv`
- `best_cpu_vs_best_gpu.csv`

Nutze diese Dateien für Validierung, Methodik und Sensitivität:

- `input_manifest.csv`
- `all_platform_stability.csv`
- `preflight_checks.csv`
- `integrity_checks.csv`
- `METHODS_AND_LIMITATIONS.md`
- `independent_audit_checks.csv`
- `paired_regret_sensitivity.csv`
- `looso_selection_frequency.csv`
- `worst_stability_configurations.csv`

Die 8.100 Rohmesszeilen liegen in diesem Paket nicht vor. Mache daher klar,
welche Aussagen du aus Sitzungsmedianen und Aggregaten verifizieren kannst
und welche ein Rohdaten-Replay erfordern würden.

## Pflichtaufgaben

### 1. Struktur und Integrität

Prüfe mindestens:

- erwartete Plattformen, Größen, Threadkonfigurationen und fünf Sitzungen;
- Anzahl der Sitzungsmediane und Konfigurationen;
- keine offensichtlichen Duplikate oder fehlenden Konfigurationen;
- Übereinstimmung von Sitzungsmedianen und Konfigurationszusammenfassungen;
- Formeln für Laufzeit, Energie, EDP, Durchsatz, logische Rate und GB/J;
- ob vorhandene Validator-, Preflight- und Integrity-Urteile durch die
  gelieferten Tabellen gestützt werden.

### 2. Zahlen unabhängig nachrechnen

Rechne aus den fünf Sitzungsmedianen mindestens nach:

- Median je Plattform, Größe und Konfiguration;
- Sitzungs-CV;
- Laufzeit-, Energie- und EDP-Optimum je Plattform und Größe;
- globale Laufzeit-, Energie- und EDP-Gewinner;
- strikte Laufzeit-/Energie-Paretofronten;
- praktische Paretofronten mit 2-%-Schwelle;
- Verhältnisse und Penalties zwischen Laufzeit- und Energieoptimum;
- CPU-vs.-GPU-Verhältnisse.

Falls du Konfidenzintervalle neu berechnest, verwende eine exakt
enumerierte Bootstrap-Verteilung der Mediane aus fünf Sitzungen oder
erkläre präzise, warum du eine andere Methode verwendest.

### 3. Folgende Behauptungen gezielt verifizieren oder falsifizieren

Prüfe mit konkreten Zahlen:

1. Bei sechs von neun Größen existiert ein klarer Geräte-Zielkonflikt
   zwischen schnellster und energieärmster Plattform.
2. Die RTX 5060 Ti ist bei allen neun Größen der klare Energiegewinner.
3. Ab `N=16M` ist die RTX 3090 ungefähr zweimal schneller als die
   RTX 5060 Ti, benötigt aber ungefähr zweimal deren Energie.
4. Das EDP dieser beiden GPUs ist bei großen Größen fast gleich.
5. Zwischen `N=8M` und `N=16M` zeigt die RTX 5060 Ti einen abrupten
   Laufzeit-/Energie-Regimewechsel.
6. Bei `N=16M` gibt es eine Paretofront mit RTX 3090, RTX 5060 Ti und
   AMD-Konfigurationen.
7. Intel zeigt bei großen Größen hohe Energieaufschläge für sehr kleine
   Laufzeitgewinne durch zusätzliche Threads.
8. AMD zeigt bei `128M` und `256M` einen robusten Konflikt zwischen
   Laufzeit- und Energieoptimum.
9. AMD `4M/64T` ist stark instabil oder bimodal und darf nicht als stabiler
   Sieger interpretiert werden.
10. GPU-Konfigurationen sind über Sitzungen deutlich stabiler als die
    CPU-Threadkonfigurationen.

Erstelle für jede Behauptung:

```text
Status: bestätigt / teilweise bestätigt / nicht bestätigt
reproduzierte Kennzahl
Unsicherheit oder Einschränkung
zulässige Paper-Formulierung
```

### 4. Gewinnerlogik streng behandeln

Unterscheide konsequent:

- exakter Punktgewinner;
- tie-aware Leader-Set;
- klare Auswahl;
- unsichere Auswahl;
- praktische Äquivalenz innerhalb 2 %;
- post-selection/deskriptive Gewinneranalyse.

Ein Punktgewinner mit überlappender Unsicherheit ist kein klarer
Hardwaregewinner.

Prüfe besonders, ob `within_platform_exact_winner_regret.csv` zu starke
Aussagen macht. Vergleiche sie mit:

- `paired_regret_sensitivity.csv`
- `looso_selection_frequency.csv`

### 5. Stabilität und Ausreißer

Berichte:

- Anzahl und Anteil instabiler Konfigurationen je Plattform;
- die zehn auffälligsten Konfigurationen;
- ob Instabilität die globalen oder internen Gewinner verändert;
- welche Resultate trotz korrektem Median nicht paperfest sind.

Ausreißer dürfen nicht still entfernt werden.

### 6. Paper-Interpretation

Erst nach der Zahlenprüfung:

- Nenne die drei bis fünf wissenschaftlich stärksten Befunde.
- Ordne sie nach Evidenzstärke und potenzieller Neuheit.
- Trenne strikt:
  - direkt gemessener Befund;
  - mathematisch abgeleitete Größe;
  - plausible Mechanismushypothese;
  - nicht belegte Kausalbehauptung.
- Beurteile, ob REDUCTION nur STREAM bestätigt oder einen eigenständigen
  Beitrag liefert.
- Formuliere eine belastbare zentrale Paper-These.
- Nenne den kleinsten Zusatzversuch mit dem größten Erkenntnisgewinn.

## Verbotene Abkürzungen

- Keine Hardware- oder Cacheerklärung allein aus Modellnamen ableiten.
- Keine externen Webquellen oder Hersteller-Peaks verwenden.
- Keine Kausalität aus einem Knick in einer Kurve behaupten.
- Logische Rate nicht als physische Speicherbandbreite bezeichnen.
- EDP, Laufzeit und Energie nicht als drei unabhängige Messungen behandeln.
- Bestehende Markdown-Zusammenfassungen nicht einfach paraphrasieren.
- Keine Resultate erfinden, wenn eine Datei oder Spalte fehlt.

## Gewünschtes Ausgabeformat

### A. Auditurteil

```text
PASS
PASS WITH QUALIFICATIONS
oder FAIL
```

Mit höchstens fünf wichtigsten Gründen.

### B. Reproduktionsmatrix

Eine Tabelle:

```text
Behauptung | reproduzierte Zahl | Status | Einschränkung
```

### C. Nüchterne Ergebnisdarstellung

Tabellen für:

- globale Placement-Entscheidung je Größe;
- GPU-3090/5060-Ti-Verhältnisse;
- wichtigste CPU-Threadkonflikte;
- Stabilität je Plattform.

### D. Fehler und Schwachstellen

Getrennt nach:

- Daten-/Rechenfehler;
- statistische Schwäche;
- Reportingproblem;
- nicht belegte Interpretation.

### E. Paperwert

- stärkster Kernclaim;
- zweitstärkster Claim;
- überraschendster Befund;
- Resultate, die nicht ins Abstract gehören;
- kleinster sinnvoller Zusatzversuch.

### F. Claim-Sprache

Gib konkrete Formulierungen für:

- zulässige Claims;
- nur vorsichtig zulässige Hypothesen;
- verbotene oder überzogene Claims.

Schreibe auf Deutsch. Sei detailliert, numerisch und kritisch. Zeige bei
wichtigen Zahlen die verwendete Formel oder den Rechenweg. Sage ausdrücklich,
wenn dein Ergebnis von einer vorhandenen Auswertung abweicht.

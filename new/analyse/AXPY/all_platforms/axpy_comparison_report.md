# AXPY – Vergleich aller Plattformen

## Methodik

Die Vergleichspunkte stammen aus dem Median der Session-Mediane. Für CPUs bleibt jede Threadkonfiguration zunächst ein eigener Punkt; die GPUs besitzen je Größe einen Punkt. Die globale Pareto-Analyse verwendet E2E-Zeit und Device-Domain-Energie pro Operation.

Gewinnergleichstände verwenden eine Totzone von ±2.00 %. Die Plattform-Envelopes wählen pro Größe separat die schnellste, energieärmste und EDP-minimale Konfiguration einer Plattform. Die Linienabbildungen zeigen Bootstrap-Unsicherheitsbalken und beschriften die tatsächlich gemessenen Dezimalgrößen 1M bis 256M.

## Punktgewinner über die neun Problemgrößen

| Kriterium | Intel | AMD | RTX 3090 | RTX 5060 Ti |
|---|---:|---:|---:|---:|
| Laufzeit-Envelope | 0 | 1 | 5 | 3 |
| Energie-Envelope | 0 | 1 | 0 | 8 |
| EDP-Envelope | 0 | 1 | 5 | 3 |

Mehrfachzählungen sind bei Gleichständen innerhalb der Totzone möglich. Diese Tabelle verwendet nur Punktschätzer; die nachgeschaltete Robustheitsanalyse prüft Session-Konsistenz, Streuung, Intervalle und In-Range-Sensitivität.

## Globale Pareto-Struktur

| Plattform | Zahl globaler Pareto-Punkte |
|---|---:|
| Intel i9-7900X | 0 |
| AMD Threadripper 3970X | 1 |
| RTX 3090 | 5 |
| RTX 5060 Ti | 8 |

Klassifikationen aller Plattform-/Thread-/Größenpunkte:

- `joint_winner`: 4
- `runtime_winner`: 5
- `energy_winner`: 5
- `pareto_tradeoff`: 0
- `dominated`: 148

## Auffälligste Streuungen

| Plattform | N | Threads | CV Zeit | CV Energie |
|---|---:|---:|---:|---:|
| AMD Threadripper 3970X | 2000000 | 64 | 34.48 % | 32.17 % |
| AMD Threadripper 3970X | 4000000 | 32 | 28.03 % | 25.16 % |
| Intel i9-7900X | 1000000 | 20 | 17.69 % | 12.96 % |
| AMD Threadripper 3970X | 8000000 | 32 | 17.63 % | 16.10 % |
| AMD Threadripper 3970X | 1000000 | 16 | 16.20 % | 12.97 % |
| Intel i9-7900X | 1000000 | 10 | 11.73 % | 9.81 % |
| AMD Threadripper 3970X | 2000000 | 20 | 11.42 % | 6.75 % |
| AMD Threadripper 3970X | 1000000 | 10 | 10.87 % | 5.67 % |
| Intel i9-7900X | 1000000 | 16 | 10.28 % | 8.94 % |
| AMD Threadripper 3970X | 32000000 | 8 | 0.33 % | 7.98 % |
| AMD Threadripper 3970X | 2000000 | 2 | 6.62 % | 5.53 % |
| AMD Threadripper 3970X | 32000000 | 10 | 0.29 % | 6.60 % |

## Wissenschaftliche Interpretation

- `device_energy` bedeutet auf CPUs RAPL-Package und auf GPUs NVML-Board. Der Vergleich ist deshalb kein vollständiger Systemenergievergleich. Bei AXPY begünstigt die fehlende externe CPU-DRAM-Energie tendenziell die CPU-Seite.
- Runtime-, Energie- und EDP-Envelopes dürfen nicht als derselbe Betriebspunkt interpretiert werden: Eine Plattform kann für jedes Kriterium eine andere Threadzahl wählen.
- Die globale Pareto-Tabelle ist deshalb die zentrale Datei für Trade-off- und Crossover-Aussagen.
- Winner Counts sind deskriptive Orientierung. Die spätere Paper-Analyse sollte Session-Unsicherheit, Effektgrößen und Crossover-Stabilität explizit modellieren.

## Ergebnisdateien

- `axpy_global_pareto.csv`: alle Konfigurationen mit Pareto-Status
- `axpy_platform_envelopes.csv`: beste Laufzeit/Energie/EDP je Plattform und N
- `axpy_cross_platform_by_size.csv`: Gewinner je Problemgröße
- `axpy_platform_penalties.csv`: Penalty-Faktoren relativ zum Besten

Abbildungen: erstellt

## Robustheitsstatus der Gewinner

Winner Counts oberhalb sind Punktschätzer. Die primäre Session-Aussage hält die ausgewählte Threadkonfiguration fest. Richtungssicherheit und Stabilität der Effektgröße stehen getrennt in `axpy_robustness_report.md`.

# AXPY – Robustheit und Sensitivität

Die Analyse trennt nun zwei unterschiedliche Session-Fragen: `fixed_config_session_support` hält die global ausgewählte Threadkonfiguration fest; `oracle_envelope_session_support` darf die Threadzahl in jeder Session neu wählen. Die primäre robuste Winner-Aussage verwendet die feste Konfiguration.

`directionally_robust` bedeutet: Punktgewinner, 5/5 Sessions mit fester Konfiguration und konservativ getrennte Bootstrap-Intervalle. `fully_robust` verlangt zusätzlich 5/5 Konsistenz des per-Session-Oracle-Envelopes und CV ≤ 10 %. Damit bleiben ein einzelner alternativer CPU-Betriebspunkt oder eine instabile Effektgröße sichtbar.

| N | Laufzeit: Punkt → Richtung → voll | Energie: Punkt → Richtung → voll | EDP: Punkt → Richtung → voll | In-Range 5×10 |
|---:|---|---|---|---|
| 1000000 | 5060ti → 5060ti → 5060ti | 5060ti → 5060ti → 5060ti | 5060ti → 5060ti → 5060ti | gleich |
| 2000000 | 5060ti → 5060ti → 5060ti | 5060ti → 5060ti → 5060ti | 5060ti → 5060ti → 5060ti | gleich |
| 4000000 | 5060ti → 5060ti → – | 5060ti → 5060ti → – | 5060ti → 5060ti → – | gleich |
| 8000000 | amd → amd → – | amd → amd → – | amd → amd → – | gleich |
| 16000000 | 3090 → 3090 → 3090 | 5060ti → 5060ti → 5060ti | 3090 → 3090 → 3090 | gleich |
| 32000000 | 3090 → 3090 → 3090 | 5060ti → 5060ti → 5060ti | 3090 → 3090 → 3090 | gleich |
| 64000000 | 3090 → 3090 → 3090 | 5060ti → 5060ti → 5060ti | 3090 → 3090 → 3090 | gleich |
| 128000000 | 3090 → 3090 → 3090 | 5060ti → 5060ti → 5060ti | 3090 → 3090 → 3090 | gleich |
| 256000000 | 3090 → 3090 → 3090 | 5060ti → 5060ti → 5060ti | 3090 → 3090 → 3090 | gleich |

## GPU-Throttle-Masken

| Plattform | Maske | Dekodierung | Zeilen |
|---|---|---|---:|
| 3090 | `0x4` | software_power_cap | 450 |
| 5060ti | `0x0` | none | 450 |

## Einordnung

- In-Range-5×10-Sensitivitätsabweichungen: **0** von 27 Metrik×Größe-Fällen.
- Eindeutige unvollständige In-Range-Konfigurationen: **7**; sie sind aus der primären 5×10-Analyse ausgeschlossen und nur in der Any-Coverage-Sensitivität sichtbar.
- `directionally_robust=yes`, aber `fully_robust=no` bedeutet: Richtung des Gewinners stabil, genaue Effektgröße jedoch variabel.
- Bootstrap-Intervalle beruhen auf fünf Session-Zusammenfassungen und sind Unsicherheitsintervalle, kein alleiniger Signifikanznachweis.

# AXPY – Temperatur- und Takttelemetrie

Die Telemetrie ist ein Diagnoseanker. Hohe Temperatur allein verwirft keine Messung; ein möglicher thermischer Effekt wird separat über den Taktvergleich heißer (≥95 °C) und kühler (<90 °C) CPU-Zeilen markiert.

| Plattform | Median Temp. | Max Temp. | ≥95 °C | ≥100 °C | Median Takt nachher | heiß: nachher/vorher | heiß/kühl Takt | Taktabfall im Messfenster |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Intel i9-7900X | 77 °C | 100 °C | 102 | 9 | 2324 MHz | 0.98168 | 0.57659 | no |
| AMD Threadripper 3970X | 70 °C | 88 °C | 0 | 0 | 2167 MHz | n/a | n/a | not_assessable |
| RTX 3090 | 63 °C | 65 °C | 0 | 0 | 1935 MHz | n/a | n/a | not_assessable |
| RTX 5060 Ti | 55 °C | 72 °C | 0 | 0 | 2790 MHz | n/a | n/a | not_assessable |

Konfigurationsdetails stehen in `axpy_telemetry_by_config.csv`. Die Kennzahl `within_window_clock_drop_detected` verwendet primär `clock_after/clock_before` innerhalb der heißen Zeilen. Sie erkennt nur zusätzlichen Taktabfall im Messfenster und schließt keinen bereits vor Messbeginn reduzierten Taktzustand aus. `not_assessable` bedeutet, dass weniger als fünf heiße CPU-Zeilen vorliegen.

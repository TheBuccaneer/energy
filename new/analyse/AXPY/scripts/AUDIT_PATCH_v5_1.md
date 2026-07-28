# AXPY Analysis Patch v5.1

## Behobener Fehler

Der eingefrorene CPU-AXPY-Writer schreibt `throttle_reasons` als leeres CSV-Feld.
Der v5-Validator verlangte irrtümlich den numerischen Sentinel `-1`, wodurch jede
Intel- und AMD-Zeile verworfen wurde.

## Neue Regel

- CPU `throttle_reasons`: leer **oder** numerisch `-1`
- GPU `throttle_reasons`: weiterhin verpflichtende hexadezimale NVML-Maske

Keine Mess- oder Aggregationssemantik wurde geändert.

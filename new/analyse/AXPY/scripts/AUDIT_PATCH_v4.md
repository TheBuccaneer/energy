# AXPY analysis v4 — audit closure

## Addressed findings

1. GPU quickchecks without a Shell-PASS marker are accepted only through direct
   evidence: at least six AXPY result lines, all checksums OK, Anti-Collapse
   `gate=PASS`, and no fatal/checksum failure.
2. CPU clock/temperature fields remain in normalized data and are summarized by
   platform and configuration. Hot-row `clock_after/clock_before` is the primary
   frequency-drop diagnostic.
3. Session robustness distinguishes the globally fixed selected configuration
   from a per-session oracle thread envelope.
4. Directional robustness and magnitude stability are separate. Full robustness
   additionally requires oracle-envelope consistency.
5. Primary in-range sensitivity requires all 5 sessions; partial configurations
   are reported separately.
6. Validation report table order and plot x-axis labels were corrected. Envelope
   plots now include bootstrap uncertainty bars.
7. Global completion status propagates component warnings.
8. Manifest parsing, session hashes, source, runner, and binary provenance are
   reported as separate states.
9. Campaign IDs and raw CSV/log hashes are frozen in `axpy_campaign_lock.json`.
10. A self-contained handover archive is produced after successful analysis.

## Tests executed

- Python compilation of every script: PASS
- Shell syntax: PASS
- v4 regression selftest: PASS
- GPU quickcheck without Shell-PASS marker: PASS
- explicit campaign-lock selection: PASS
- fixed-config versus oracle-session distinction: PASS
- comparison/robustness regression on the supplied 8,100-row derived result: PASS
- expected actual-data classifications:
  - 4M RTX 5060 Ti: directionally robust, not fully robust
  - 8M AMD: directionally robust, not fully robust
  - six unique partial in-range configurations
- full synthetic 8,100-row, four-platform pipeline including provenance,
  validation, telemetry, plots, robustness, atomic output, lock, and handover
  archive: PASS

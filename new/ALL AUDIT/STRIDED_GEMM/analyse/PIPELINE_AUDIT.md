# Pipeline audit summary

## Verdict

**PASS WITH WARNINGS expected on valid data.**

Warnings are methodological disclosures, not automatic data failures:

- CPU package RAPL versus GPU board NVML;
- native-best post-selection;
- five-session scope;
- AMD DRAM RAPL unavailable;
- dense comparison optional if the prior GEMM unified outputs are absent.

## Independent checks implemented

- exact campaign coverage;
- checksum and formula reproduction;
- source and runner provenance;
- `ld=2N` and `problem_spec` semantics;
- logical bytes versus allocated footprint separation;
- five-session bootstrap statistics;
- pairwise ratio-CI classification;
- metric identity checks;
- Pareto recomputation;
- dense-versus-strided ratio identity;
- dense-versus-strided classification-rule reproduction;
- order-invariant Dense-vs-Strided leader-set comparison;
- separation of point-estimate and decisive placement changes;
- stability exposure for selected configurations.

## Known limitations

- No whole-system energy.
- GPU PCIe transfers excluded in resident mode.
- Native-best intervals are descriptive.
- Logical bytes are not measured traffic.
- Results characterize the measured systems, including Intel's thermal regime.

## v3 regression guard

The integrity audit fails if comma-separated leader lists are compared by string order,
or if a point-estimate change under `tie_or_uncertain` is labeled as a decisive
placement change.

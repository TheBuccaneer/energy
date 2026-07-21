# STREAM pipeline audit design

## Intended verdict on valid data

**PASS WITH WARNINGS** is expected because the warnings are methodological disclosures:

- CPU package RAPL versus GPU board NVML;
- GPU resident mode excludes PCIe transfers;
- logical bandwidth is not measured physical traffic;
- native-best selection is descriptive;
- five-session scope.

## Regression checks

- exact campaign coverage;
- 45-column `cpu-gpu-v2` schema;
- formula reproduction with explicit CSV rounding tolerances;
- checksum and thread/GPU semantics;
- source/runner SHA-256 provenance;
- exact session-median row counts;
- runtime↔bandwidth and energy↔GB/J leader consistency;
- pairwise ratio classification;
- strict and 2%-practical Pareto recomputation;
- no pseudoreplication.

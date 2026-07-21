# STREAM analysis package audit

## Verdict

**PASS FOR INSTALLATION; REAL-DATA EXECUTION PENDING COMPLETE CAMPAIGNS.**

## Scope

The package installs individual STREAM validation/analysis pipelines for:

- AMD CPU
- Intel CPU
- RTX 3090
- RTX 5060 Ti

It also installs an all-platform audit pipeline under `ALL AUDIT/STREAM/`.

## Static checks completed

- all Python files pass `python3 -m compileall`;
- all shell scripts pass `bash -n`;
- no existing GEMM or STRIDED_GEMM path is modified;
- installation paths match the supplied project tree;
- the package contains no `__pycache__` artifacts;
- SHA-256 manifest is included.

## Synthetic regression test

A complete synthetic campaign was generated with the real 45-column `cpu-gpu-v2`
schema, five sessions, ten repetitions, all nine STREAM sizes, both CPU thread grids,
and one resident configuration per GPU.

The following stages completed successfully:

- all four individual validators;
- all four individual analyses;
- combined preflight;
- unified statistics;
- all-platform pairwise comparison;
- report generation;
- independent integrity audit.

The synthetic combined integrity verdict was `PASS WITH WARNINGS`, where the warnings
are the expected methodological disclosures rather than calculation failures.

## Regression guards

- exact campaign/session/configuration coverage;
- 45-column schema and workload semantics;
- formula reproduction with CSV rounding allowances;
- checksum validation;
- CPU thread and GPU sentinel checks;
- package/board energy-domain semantics;
- exact metric identities;
- runtime↔logical-bandwidth leader consistency;
- energy↔logical-GB/J leader consistency;
- tie-aware native-best selection;
- pairwise ratio classification;
- strict and 2%-practical Pareto recomputation;
- stability exposure without silent row deletion.

## Real-data prerequisite

Each platform must have a complete five-session campaign named:

- `stream_amd_YYYYMMDD_HHMMSS_session1..5.csv`
- `stream_intel_YYYYMMDD_HHMMSS_session1..5.csv`
- `stream_3090_YYYYMMDD_HHMMSS_session1..5.csv`
- `stream_5060ti_YYYYMMDD_HHMMSS_session1..5.csv`

Partial campaigns are rejected and are never silently combined.

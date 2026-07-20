# Audit of the all-platform GEMM analysis pipeline — v2

## Test inputs

The full pipeline was run against the current validated campaigns:

- Intel: `20260719_085511`
- AMD: `20260719_085402`
- RTX 3090: `20260719_152731`
- RTX 5060 Ti: `20260719_172746`

Combined scope:

- 8,100 raw measurements
- 20 session files
- 162 platform/size/configuration combinations
- 810 independent session-level summary rows
- 9 common matrix sizes

## Corrections introduced after independent audit

### 1. Direct work-normalized metrics

The first version formed GFLOP/J from separately aggregated throughput and power.
That can differ from logical FLOP divided by measured energy because ratios of
medians are not medians of ratios. It produced one contradictory classification
at N=512: energy favored RTX 3090 clearly while efficiency was marked uncertain.

Version 2 derives:

- throughput = `2*N^3 / e2e_runtime`
- GFLOP/J = `2*N^3 / energy`

Energy and GFLOP/J leader sets are now mathematically consistent, as are runtime
and throughput leader sets. The maximum throughput normalization adjustment was
only 0.04119%, so runtime conclusions did not change.

### 2. Coherent all-platform leader rule

Version 1 used marginal CI non-overlap in the winner table but ratio-CI inference
in pairwise tables. At N=512 this yielded a contradictory label. Version 2 uses
the pairwise practical-ratio CI rule against every alternative for the final
all-platform leader table. N=512 energy is therefore correctly reported as
RTX 3090 / RTX 5060 Ti uncertain, with a point advantage for RTX 3090.

### 3. Single GPU configuration semantics

Each GPU has one measured resident configuration. Version 1 labeled that
`tie_or_uncertain`; version 2 labels it `single_configuration`.

### 4. Independent integrity stage

`05_integrity_audit.py` independently recomputes central identities, leader
coherence, effect ranges, Pareto flags, and trade-off classes. The tested output
verdict is `PASS WITH WARNINGS` with no hard failures.

## Remaining warnings

- 33 configurations exceed 5% conventional session CV in at least one central
  metric: 29 Intel, 4 AMD, 0 RTX 3090, 0 RTX 5060 Ti.
- Native-best configuration selection and comparison use the same five sessions.
- CPU package RAPL and GPU board NVML are different measurement domains.
- GPU execution is resident and excludes PCIe transfers.
- Five sessions establish repeatability on these systems, not hardware-population
  generality.
- The three copied CPU-only convenience tables originate in the separately
  audited CPU analysis. They are not required for the central four-platform
  conclusions and should not be the primary Claude inputs.

## Final tested verdict

**PASS WITH WARNINGS.** The corrected pipeline is suitable for descriptive,
same-size, resident GEMM comparison under the stated energy-domain and
post-selection limitations.

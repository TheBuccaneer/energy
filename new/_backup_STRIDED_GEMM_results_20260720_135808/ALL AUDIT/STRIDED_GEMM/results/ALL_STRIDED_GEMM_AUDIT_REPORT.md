# Combined audit of all STRIDED_GEMM campaigns

## Overall verdict

**PASS WITH WARNINGS**

The combined analysis includes 8,100 raw measurements from four validated
campaigns and 162 platform/size/configuration combinations:

| platform_label   |        campaign |   raw_measurement_rows |   configurations |   validation_warnings | energy_domain    |
|:-----------------|----------------:|-----------------------:|-----------------:|----------------------:|:-----------------|
| Intel CPU        | 20260719_175326 |                   3150 |               63 |                     1 | CPU package RAPL |
| AMD CPU          | 20260719_175453 |                   4050 |               81 |                     2 | CPU package RAPL |
| RTX 3090         | 20260720_120846 |                    450 |                9 |                     1 | GPU board NVML   |
| RTX 5060 Ti      | 20260720_114819 |                    450 |                9 |                     2 | GPU board NVML   |

No raw row is treated as an independent statistical experiment. Ten within-session
repetitions are technical repetitions; the primary statistical units are the five
session medians per configuration.

## What this pipeline compares

1. **All configurations:** every Intel and AMD thread count plus one resident
   configuration per GPU.
2. **Native-best per platform:** energy-, runtime-, throughput-, efficiency-, and
   EDP-optimized configurations selected separately.
3. **Best CPU versus best GPU:** the better CPU and better GPU for each size and
   metric, without confusing CPU thread tuning with device placement.
4. **Pairwise device ratios:** all six platform pairs with deterministic
   bootstrap ratio intervals, probability of superiority, and Cliff's delta.

## Statistical interpretation

- Point estimates: median of five session medians.
- Within-configuration uncertainty: exact bootstrap of the five session medians.
- Pairwise ratio uncertainty: deterministic independent bootstrap of the two
  five-session samples.
- Practical equivalence tolerance: ±2%.
- A clear all-platform leader must be clearly favored against every alternative
  by the pairwise ratio-CI rule outside the ±2% practical-equivalence band.
- Native-best ratio intervals are **descriptive**, not confirmatory: configuration
  selection and summarization use the same five sessions, so no post-selection
  p-values are reported.

## Metric identities

At a fixed matrix size, every configuration performs exactly `2*N^3` logical
FLOP. The combined pipeline therefore normalizes:

- throughput as logical FLOP divided by e2e runtime;
- GFLOP/J as logical FLOP divided by measured device-domain energy.

Runtime/throughput and energy/GFLOP-J are inverse views, not four independent
dimensions. EDP is the joint energy-runtime metric.

## Energy-domain limitation

CPU energy is package-only RAPL. GPU energy is NVML board energy and includes
on-board memory. The comparison is valid as a comparison of the measured device
domains, but it is not whole-system energy and must not be described as a pure
architecture-only comparison.

## Metric normalization

For fixed `N`, throughput is derived as `2*N^3 / runtime` and GFLOP/J as
`2*N^3 / energy`. Hence runtime and throughput are one dimension, while energy
per STRIDED_GEMM and GFLOP/J are one dimension. Do not count their winner tables as
independent replications of the same finding.

## Tie-aware all-platform leaders

|     N | Energy leaders        | Runtime leaders   | EDP leaders   | Throughput leaders   | Efficiency leaders    |
|------:|:----------------------|:------------------|:--------------|:---------------------|:----------------------|
|    64 | Intel CPU             | Intel CPU         | Intel CPU     | Intel CPU            | Intel CPU             |
|   128 | RTX 5060 Ti           | RTX 3090          | RTX 5060 Ti   | RTX 3090             | RTX 5060 Ti           |
|   256 | RTX 5060 Ti           | RTX 3090          | RTX 3090      | RTX 3090             | RTX 5060 Ti           |
|   512 | RTX 5060 Ti, RTX 3090 | RTX 3090          | RTX 3090      | RTX 3090             | RTX 5060 Ti, RTX 3090 |
|  1024 | RTX 3090              | RTX 3090          | RTX 3090      | RTX 3090             | RTX 3090              |
|  2048 | RTX 3090              | RTX 3090          | RTX 3090      | RTX 3090             | RTX 3090              |
|  4096 | RTX 3090              | RTX 3090          | RTX 3090      | RTX 3090             | RTX 3090              |
|  8192 | RTX 3090              | RTX 3090          | RTX 3090      | RTX 3090             | RTX 3090              |
| 16384 | RTX 3090              | RTX 3090          | RTX 3090      | RTX 3090             | RTX 3090              |

## Best CPU versus best GPU

|     N | Energy best CPU/GPU   |   CPU/GPU energy ratio | Energy classification   | Runtime best CPU/GPU   |   CPU/GPU runtime ratio | Runtime classification   |   EDP ratio |
|------:|:----------------------|-----------------------:|:------------------------|:-----------------------|------------------------:|:-------------------------|------------:|
|    64 | INTEL/5060ti          |                 0.3985 | clear_CPU               | INTEL/3090             |                  0.4702 | clear_CPU                |     0.09964 |
|   128 | INTEL/5060ti          |                 2.785  | clear_GPU               | INTEL/3090             |                  2.273  | clear_GPU                |     3.503   |
|   256 | INTEL/5060ti          |                 4.419  | clear_GPU               | INTEL/3090             |                  5.392  | clear_GPU                |    13.83    |
|   512 | INTEL/5060ti          |                 5.44   | clear_GPU               | INTEL/3090             |                 11.06   | clear_GPU                |    59.94    |
|  1024 | INTEL/3090            |                 6.483  | clear_GPU               | AMD/3090               |                 13.15   | clear_GPU                |    97.36    |
|  2048 | AMD/3090              |                 7.762  | clear_GPU               | AMD/3090               |                 10.89   | clear_GPU                |    84.44    |
|  4096 | AMD/3090              |                 7.881  | clear_GPU               | AMD/3090               |                  9.769  | clear_GPU                |    82.64    |
|  8192 | AMD/3090              |                 8.497  | clear_GPU               | AMD/3090               |                 11      | clear_GPU                |    97.6     |
| 16384 | AMD/3090              |                 8.246  | clear_GPU               | AMD/3090               |                 10.43   | clear_GPU                |    90.8     |

## RTX 3090 versus RTX 5060 Ti

|     N |   3090/5060Ti runtime |   3090/5060Ti energy |   3090/5060Ti GFLOP/J | Runtime result   | Energy result              |
|------:|----------------------:|---------------------:|----------------------:|:-----------------|:---------------------------|
|    64 |                0.536  |               2.829  |                0.3534 | clear_3090       | clear_5060ti               |
|   128 |                0.5269 |               2.377  |                0.4207 | clear_3090       | clear_5060ti               |
|   256 |                0.4448 |               1.733  |                0.5772 | clear_3090       | clear_5060ti               |
|   512 |                0.3382 |               1.026  |                0.9747 | clear_3090       | uncertain_5060ti_advantage |
|  1024 |                0.271  |               0.72   |                1.389  | clear_3090       | clear_3090                 |
|  2048 |                0.2179 |               0.5441 |                1.838  | clear_3090       | clear_3090                 |
|  4096 |                0.1975 |               0.4631 |                2.159  | clear_3090       | clear_3090                 |
|  8192 |                0.1869 |               0.4411 |                2.267  | clear_3090       | clear_3090                 |
| 16384 |                0.1856 |               0.3827 |                2.613  | clear_3090       | clear_3090                 |

## Energy-runtime configuration trade-offs

Clear within-CPU configuration conflicts occur in 1
size/platform cases. Their full leader sets and penalties are recorded in
`within_platform_energy_runtime_tradeoffs.csv`.

Across devices, 2 sizes have clear disjoint energy and
runtime leaders, while 0 have disjoint but
statistically/practically uncertain leader sets. The per-size result is in
`placement_by_size.csv`.

## Pareto analysis

`all_configuration_pareto.csv` contains strict and practical 2% Pareto status for
every CPU thread configuration and both GPU configurations at every N.
`configuration_tradeoff_map.csv` assigns each point to one of:

- dominant or practically equivalent;
- energy-efficient compromise;
- runtime-efficient compromise;
- balanced Pareto trade-off;
- dominated.

This classification is preferable to a simple winner count because it exposes
how much runtime is traded for energy and vice versa.

## Stability

There are 40 configurations with more than 5% session-level CV in
runtime, energy, or throughput. They are concentrated on the CPUs and are listed
in `all_platform_stability.csv`; use `stability_breakdown.csv` and
`leader_stability.csv` from the integrity audit to separate general instability
from instability of an exact policy-selected configuration. The individual
platform audits remain the authority for thermal and sensor-specific warnings.

## Audit decision

The combined dataset is suitable for descriptive cross-platform STRIDED_GEMM analysis.
Claims must retain three qualifications: resident GPU execution, asymmetric
energy domains, and descriptive post-selection uncertainty for native-best
comparisons.

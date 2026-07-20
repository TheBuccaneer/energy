# Combined audit of all GEMM campaigns

## Overall verdict

**PASS WITH WARNINGS**

The combined analysis includes 8,100 raw measurements from four validated
campaigns and 162 platform/size/configuration combinations:

| platform_label   |        campaign |   raw_measurement_rows |   configurations |   validation_warnings | energy_domain    |
|:-----------------|----------------:|-----------------------:|-----------------:|----------------------:|:-----------------|
| Intel CPU        | 20260719_085511 |                   3150 |               63 |                     2 | CPU package RAPL |
| AMD CPU          | 20260719_085402 |                   4050 |               81 |                     2 | CPU package RAPL |
| RTX 3090         | 20260719_152731 |                    450 |                9 |                     3 | GPU board NVML   |
| RTX 5060 Ti      | 20260719_172746 |                    450 |                9 |                     2 | GPU board NVML   |

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
per GEMM and GFLOP/J are one dimension. Do not count their winner tables as
independent replications of the same finding.

## Tie-aware all-platform leaders

|     N | Energy leaders        | Runtime leaders   | EDP leaders   | Throughput leaders   | Efficiency leaders    |
|------:|:----------------------|:------------------|:--------------|:---------------------|:----------------------|
|    64 | Intel CPU             | Intel CPU         | Intel CPU     | Intel CPU            | Intel CPU             |
|   128 | RTX 5060 Ti           | RTX 3090          | RTX 5060 Ti   | RTX 3090             | RTX 5060 Ti           |
|   256 | RTX 5060 Ti           | RTX 3090          | RTX 3090      | RTX 3090             | RTX 5060 Ti           |
|   512 | RTX 3090, RTX 5060 Ti | RTX 3090          | RTX 3090      | RTX 3090             | RTX 3090, RTX 5060 Ti |
|  1024 | RTX 3090              | RTX 3090          | RTX 3090      | RTX 3090             | RTX 3090              |
|  2048 | RTX 3090              | RTX 3090          | RTX 3090      | RTX 3090             | RTX 3090              |
|  4096 | RTX 3090              | RTX 3090          | RTX 3090      | RTX 3090             | RTX 3090              |
|  8192 | RTX 3090              | RTX 3090          | RTX 3090      | RTX 3090             | RTX 3090              |
| 16384 | RTX 3090              | RTX 3090          | RTX 3090      | RTX 3090             | RTX 3090              |

## Best CPU versus best GPU

|     N | Energy best CPU/GPU   |   CPU/GPU energy ratio | Energy classification   | Runtime best CPU/GPU   |   CPU/GPU runtime ratio | Runtime classification   |   EDP ratio |
|------:|:----------------------|-----------------------:|:------------------------|:-----------------------|------------------------:|:-------------------------|------------:|
|    64 | INTEL/5060ti          |                 0.3687 | clear_CPU               | INTEL/3090             |                  0.4322 | clear_CPU                |     0.08317 |
|   128 | INTEL/5060ti          |                 2.921  | clear_GPU               | INTEL/3090             |                  2.436  | clear_GPU                |     3.827   |
|   256 | INTEL/5060ti          |                 4.416  | clear_GPU               | INTEL/3090             |                  5.494  | clear_GPU                |    15.16    |
|   512 | INTEL/3090            |                 5.253  | clear_GPU               | INTEL/3090             |                 11.16   | clear_GPU                |    60.32    |
|  1024 | INTEL/3090            |                 6.618  | clear_GPU               | INTEL/3090             |                 14.2    | clear_GPU                |    95.58    |
|  2048 | INTEL/3090            |                 8.421  | clear_GPU               | AMD/3090               |                 11.24   | clear_GPU                |    94.81    |
|  4096 | AMD/3090              |                 8.207  | clear_GPU               | AMD/3090               |                  9.614  | clear_GPU                |    84.39    |
|  8192 | AMD/3090              |                 8.894  | clear_GPU               | AMD/3090               |                 11.14   | clear_GPU                |   104.5     |
| 16384 | AMD/3090              |                 9.207  | clear_GPU               | AMD/3090               |                 10.94   | clear_GPU                |   105.8     |

## RTX 3090 versus RTX 5060 Ti

|     N |   3090/5060Ti runtime |   3090/5060Ti energy |   3090/5060Ti GFLOP/J | Runtime result   | Energy result            |
|------:|----------------------:|---------------------:|----------------------:|:-----------------|:-------------------------|
|    64 |                0.5259 |               2.528  |                0.3956 | clear_3090       | clear_5060ti             |
|   128 |                0.5202 |               2.205  |                0.4535 | clear_3090       | clear_5060ti             |
|   256 |                0.4386 |               1.596  |                0.6266 | clear_3090       | clear_5060ti             |
|   512 |                0.326  |               0.9634 |                1.038  | clear_3090       | uncertain_3090_advantage |
|  1024 |                0.2696 |               0.7212 |                1.387  | clear_3090       | clear_3090               |
|  2048 |                0.2143 |               0.5313 |                1.882  | clear_3090       | clear_3090               |
|  4096 |                0.1973 |               0.4408 |                2.268  | clear_3090       | clear_3090               |
|  8192 |                0.1859 |               0.4218 |                2.371  | clear_3090       | clear_3090               |
| 16384 |                0.1854 |               0.3466 |                2.885  | clear_3090       | clear_3090               |

## Energy-runtime configuration trade-offs

Clear within-CPU configuration conflicts occur in 2
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

There are 33 configurations with more than 5% session-level CV in
runtime, energy, or throughput. They are concentrated on the CPUs and are listed
in `all_platform_stability.csv`; use `stability_breakdown.csv` and
`leader_stability.csv` from the integrity audit to separate general instability
from instability of an exact policy-selected configuration. The individual
platform audits remain the authority for thermal and sensor-specific warnings.

## Audit decision

The combined dataset is suitable for descriptive cross-platform GEMM analysis.
Claims must retain three qualifications: resident GPU execution, asymmetric
energy domains, and descriptive post-selection uncertainty for native-best
comparisons.

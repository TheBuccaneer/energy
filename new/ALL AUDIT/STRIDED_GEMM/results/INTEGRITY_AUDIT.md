# Independent integrity audit of the combined STRIDED_GEMM outputs

## Verdict

**PASS WITH WARNINGS**

The audit recomputed the central identities, leader consistency, pairwise ranges,
and Pareto classifications from the generated CSV files. It did not rely on the
automatically generated prose as evidence.

## Hard failures

_None._

## Warnings

| category   | check                            | severity   | status   | observed                                                                  | expected                                                             |
|:-----------|:---------------------------------|:-----------|:---------|:--------------------------------------------------------------------------|:---------------------------------------------------------------------|
| stability  | all_configurations_below_5pct_cv | WARN       | WARN     | unstable=40; by_platform={'INTEL': 28, 'AMD': 12, '3090': 0, '5060ti': 0} | 0                                                                    |
| semantics  | energy_domain_asymmetry          | WARN       | WARN     | CPU package RAPL versus GPU board NVML                                    | retain in every CPU/GPU energy and Pareto claim                      |
| statistics | native_best_post_selection       | WARN       | WARN     | selection and estimation use the same five sessions                       | descriptive intervals/effects only                                   |
| statistics | five_sessions_limit_inference    | WARN       | WARN     | n=5 session medians per configuration                                     | repeatability on measured systems, not hardware-population inference |

## Corrected metric semantics

For each fixed matrix size, every configuration performs the same logical work,
`2*N^3` FLOP. The normalized views therefore obey exact identities:

- throughput = logical FLOP / e2e runtime;
- GFLOP/J = logical FLOP / measured device-domain energy.

Consequently, runtime and throughput are inverse views of the same dimension,
and energy per STRIDED_GEMM and GFLOP/J are inverse views of the same dimension. They
must not be counted as four independent findings. EDP remains the joint
energy-runtime metric.

The maximum difference between the originally reported throughput and the
normalized e2e-throughput is 0.01005%, so the normalization fix
does not change the substantive runtime winners.

## Stability breakdown

| platform   |   unstable_configurations |   runtime_unstable |   energy_unstable |   throughput_unstable | platform_label   |
|:-----------|--------------------------:|-------------------:|------------------:|----------------------:|:-----------------|
| INTEL      |                        28 |                 11 |                25 |                    11 | Intel CPU        |
| AMD        |                        12 |                 11 |                10 |                    11 | AMD CPU          |
| 3090       |                         0 |                  0 |                 0 |                     0 | RTX 3090         |
| 5060ti     |                         0 |                  0 |                 0 |                     0 | RTX 5060 Ti      |

There are 40 configurations above 5% session-level CV in at least
one central metric. They are concentrated on the CPUs, especially Intel. The
full rows remain in `all_platform_stability.csv`.

21 of the 180 policy-selected platform/size views use an
exact configuration that exceeds the 5% CV rule in at least one metric. This
does not automatically invalidate a platform winner: a device-level gap may be
large even when the exact thread-count identity is uncertain. Interpret the
platform decision separately from the selected CPU thread count.

## Pairwise effect direction

`probability_a_better` is already oriented so larger means platform A is better
for the named metric. `cliffs_delta_a_minus_b`, however, is purely numerical:
positive means A has larger values than B. Therefore positive delta favors A for
throughput/GFLOP-J, but favors B for runtime/energy/EDP. This sign rule must be
stated whenever Cliff's delta is used.

## Remaining interpretation constraints

1. CPU energy is package RAPL; GPU energy is NVML board energy.
2. GPU execution is resident and excludes allocation, initialization and PCIe transfer.
3. Native-best comparisons are descriptive after configuration selection.
4. Five sessions support repeatability claims on these systems, not population claims.
5. Practical Pareto status uses the explicit 2% dominance rule implemented by the pipeline.

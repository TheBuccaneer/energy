# INTEL STRIDED_GEMM scientific analysis

## Campaign and statistical unit

- Campaign: `20260719_175326`
- Raw measurements: 3150
- Sessions: 5
- Repetitions per configuration/session: 10
- Primary statistical unit: session median
- Primary energy: CPU package RAPL (`device_energy_j / batches`)
- Optional sensitivity: package + DRAM (`total_energy_j / batches`) where DRAM RAPL exists
- Layout: N×N logical matrices with `ld=2N`; allocated footprint is 2× dense GEMM

The ten repetitions inside one session are technical repetitions. They characterize local
noise but are not treated as 50 independent experimental sessions.

## Quality indicators

- Maximum temperature: 100.0 C
- Diagnostic robust-outlier share: 7.21%
- Runtime-unstable configurations, between-session CV >5%: 11 / 63
- Package-energy-unstable configurations, between-session CV >5%: 25 / 63
- Clear energy-runtime thread trade-offs: 0 / 9

## Primary package-energy leaders

|   problem_size |   exact_best_threads | leader_threads   | selection_status   |   gap_to_second_pct |   package_energy_per_op_j_median |   runtime_per_op_s_median |
|---------------:|---------------------:|:-----------------|:-------------------|--------------------:|---------------------------------:|--------------------------:|
|             64 |                   16 | 16,2,1,10,8,4,20 | tie_or_uncertain   |           10.9794   |                      0.000131563 |               2.47604e-06 |
|            128 |                    4 | 4                | clear_leader       |           20.8124   |                      0.00118163  |               1.23826e-05 |
|            256 |                    8 | 8                | clear_leader       |           25.7262   |                      0.00551394  |               3.92646e-05 |
|            512 |                    8 | 8,16             | tie_or_uncertain   |            2.96397  |                      0.0398911   |               0.000258237 |
|           1024 |                    8 | 8                | clear_leader       |           14.0329   |                      0.262717    |               0.00177514  |
|           2048 |                    8 | 8                | clear_leader       |           11.6408   |                      2.13146     |               0.0137991   |
|           4096 |                    8 | 8,10             | tie_or_uncertain   |            3.28796  |                     17.4234      |               0.113048    |
|           8192 |                    8 | 8,10             | tie_or_uncertain   |            0.673108 |                    144.808       |               0.936769    |
|          16384 |                    8 | 8                | clear_leader       |            6.28581  |                   1154.61        |               7.70034     |

## Runtime leaders

|   problem_size |   exact_best_threads | leader_threads   | selection_status   |   gap_to_second_pct |   runtime_per_op_s_median |   package_energy_per_op_j_median |
|---------------:|---------------------:|:-----------------|:-------------------|--------------------:|--------------------------:|---------------------------------:|
|             64 |                   16 | 16,2,1,10,20,4,8 | tie_or_uncertain   |            0.220017 |               2.47604e-06 |                      0.000131563 |
|            128 |                    8 | 8,20,16,10,4     | tie_or_uncertain   |            2.36327  |               1.17895e-05 |                      0.00142756  |
|            256 |                    8 | 8                | clear_leader       |           27.473    |               3.92646e-05 |                      0.00551394  |
|            512 |                   16 | 16,8             | tie_or_uncertain   |            2.10103  |               0.000252923 |                      0.0410734   |
|           1024 |                    8 | 8,10,20          | tie_or_uncertain   |            8.04098  |               0.00177514  |                      0.262717    |
|           2048 |                    8 | 8,10             | tie_or_uncertain   |            5.29934  |               0.0137991   |                      2.13146     |
|           4096 |                   10 | 10               | clear_leader       |            6.99529  |               0.105657    |                     17.9962      |
|           8192 |                   10 | 10,8             | tie_or_uncertain   |            0.858613 |               0.928794    |                    145.783       |
|          16384 |                   10 | 10,8             | tie_or_uncertain   |            2.6151   |               7.5041      |                   1227.18        |

## Energy-runtime configuration classification

|   problem_size | tradeoff_class                    |   package_energy_exact_best_threads | package_energy_leader_threads   | package_energy_selection_status   |   runtime_exact_best_threads | runtime_leader_threads   | runtime_selection_status   | shared_leader_threads   |   energy_penalty_using_runtime_best_pct |   runtime_penalty_using_energy_best_pct | total_energy_sensitivity_same_exact_thread   |   total_energy_sensitivity_exact_best_threads | total_energy_sensitivity_leader_threads   |
|---------------:|:----------------------------------|------------------------------------:|:--------------------------------|:----------------------------------|-----------------------------:|:-------------------------|:---------------------------|:------------------------|----------------------------------------:|----------------------------------------:|:---------------------------------------------|----------------------------------------------:|:------------------------------------------|
|             64 | shared_near_optimal_configuration |                                  16 | 16,2,1,10,8,4,20                | tie_or_uncertain                  |                           16 | 16,2,1,10,20,4,8         | tie_or_uncertain           | 1,2,4,8,10,16,20        |                                0        |                                0        | True                                         |                                            16 | 16,2,1,10,8,4,20                          |
|            128 | shared_near_optimal_configuration |                                   4 | 4                               | clear_leader                      |                            8 | 8,20,16,10,4             | tie_or_uncertain           | 4                       |                               20.8124   |                                5.03096  | True                                         |                                             4 | 4                                         |
|            256 | shared_near_optimal_configuration |                                   8 | 8                               | clear_leader                      |                            8 | 8                        | clear_leader               | 8                       |                                0        |                                0        | True                                         |                                             8 | 8                                         |
|            512 | shared_near_optimal_configuration |                                   8 | 8,16                            | tie_or_uncertain                  |                           16 | 16,8                     | tie_or_uncertain           | 8,16                    |                                2.96397  |                                2.10103  | True                                         |                                             8 | 8,16                                      |
|           1024 | shared_near_optimal_configuration |                                   8 | 8                               | clear_leader                      |                            8 | 8,10,20                  | tie_or_uncertain           | 8                       |                                0        |                                0        | True                                         |                                             8 | 8                                         |
|           2048 | shared_near_optimal_configuration |                                   8 | 8                               | clear_leader                      |                            8 | 8,10                     | tie_or_uncertain           | 8                       |                                0        |                                0        | True                                         |                                             8 | 8                                         |
|           4096 | shared_near_optimal_configuration |                                   8 | 8,10                            | tie_or_uncertain                  |                           10 | 10                       | clear_leader               | 10                      |                                3.28796  |                                6.99529  | True                                         |                                             8 | 8,10                                      |
|           8192 | shared_near_optimal_configuration |                                   8 | 8,10                            | tie_or_uncertain                  |                           10 | 10,8                     | tie_or_uncertain           | 8,10                    |                                0.673108 |                                0.858613 | True                                         |                                             8 | 8,10                                      |
|          16384 | shared_near_optimal_configuration |                                   8 | 8                               | clear_leader                      |                           10 | 10,8                     | tie_or_uncertain           | 8                       |                                6.28581  |                                2.6151   | True                                         |                                             8 | 8                                         |

## Interpretation contract

1. Runtime and throughput are inverse views of the same fixed-work axis; energy and GFLOP/J
   are inverse views of the same fixed-work axis. They are not counted as independent votes.
2. EDP is a composite of runtime and energy, not a third independent physical objective.
3. A clear leader requires a >2% median gap and separated 95%
   bootstrap intervals across five session medians.
4. Exact minima are preserved, but unresolved cases are labeled `tie_or_uncertain`.
5. `logical_bytes_per_op=12N²` describes logical operands. The allocated footprint is 24N²
   because all three matrices use N×2N storage. This is not equivalent to measured DRAM traffic.
6. Package-only is primary across CPUs because AMD DRAM RAPL is unavailable. Package+DRAM
   is retained only as an optional within-platform sensitivity where DRAM measurements exist.
7. Confidence intervals describe repeatability on this machine, not a processor population.

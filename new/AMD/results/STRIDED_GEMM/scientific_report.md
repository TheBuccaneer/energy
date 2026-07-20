# AMD STRIDED_GEMM scientific analysis

## Campaign and statistical unit

- Campaign: `20260719_175453`
- Raw measurements: 4050
- Sessions: 5
- Repetitions per configuration/session: 10
- Primary statistical unit: session median
- Primary energy: CPU package RAPL (`device_energy_j / batches`)
- Optional sensitivity: package + DRAM (`total_energy_j / batches`) where DRAM RAPL exists
- Layout: N×N logical matrices with `ld=2N`; allocated footprint is 2× dense GEMM

The ten repetitions inside one session are technical repetitions. They characterize local
noise but are not treated as 50 independent experimental sessions.

## Quality indicators

- Maximum temperature: 95.0 C
- Diagnostic robust-outlier share: 9.70%
- Runtime-unstable configurations, between-session CV >5%: 11 / 81
- Package-energy-unstable configurations, between-session CV >5%: 10 / 81
- Clear energy-runtime thread trade-offs: 1 / 9

## Primary package-energy leaders

|   problem_size |   exact_best_threads | leader_threads         | selection_status   |   gap_to_second_pct |   package_energy_per_op_j_median |   runtime_per_op_s_median |
|---------------:|---------------------:|:-----------------------|:-------------------|--------------------:|---------------------------------:|--------------------------:|
|             64 |                    8 | 8,2,20,10,1,4,32,64,16 | tie_or_uncertain   |            0.538751 |                      0.000439085 |               6.13801e-06 |
|            128 |                    4 | 4,64,16,10             | tie_or_uncertain   |           36.4331   |                      0.00151956  |               1.33583e-05 |
|            256 |                    4 | 4,8                    | tie_or_uncertain   |           13.1057   |                      0.0104299   |               8.74045e-05 |
|            512 |                   16 | 16,8                   | tie_or_uncertain   |           11.1755   |                      0.0675389   |               0.000321064 |
|           1024 |                   16 | 16                     | clear_leader       |            3.68694  |                      0.443459    |               0.00187738  |
|           2048 |                   64 | 64                     | clear_leader       |           20.4556   |                      1.99878     |               0.00833042  |
|           4096 |                   64 | 64                     | clear_leader       |            7.20927  |                     14.1758      |               0.0584261   |
|           8192 |                   64 | 64                     | clear_leader       |           11.6747   |                    116.355       |               0.480382    |
|          16384 |                   64 | 64                     | clear_leader       |            7.26083  |                    921.122       |               3.67175     |

## Runtime leaders

|   problem_size |   exact_best_threads | leader_threads         | selection_status   |   gap_to_second_pct |   runtime_per_op_s_median |   package_energy_per_op_j_median |
|---------------:|---------------------:|:-----------------------|:-------------------|--------------------:|--------------------------:|---------------------------------:|
|             64 |                    8 | 8,20,2,4,10,1,32,64,16 | tie_or_uncertain   |            0.128354 |               6.13801e-06 |                      0.000439085 |
|            128 |                    4 | 4,64,16,10             | tie_or_uncertain   |           28.8561   |               1.33583e-05 |                      0.00151956  |
|            256 |                   16 | 16,8                   | tie_or_uncertain   |           17.6802   |               6.57706e-05 |                      0.0117968   |
|            512 |                   16 | 16,20,32,8             | tie_or_uncertain   |            3.34862  |               0.000321064 |                      0.0675389   |
|           1024 |                   32 | 32,64                  | tie_or_uncertain   |           10.6365   |               0.00160647  |                      0.459809    |
|           2048 |                   64 | 64,32                  | tie_or_uncertain   |            0.702137 |               0.00833042  |                      1.99878     |
|           4096 |                   32 | 32,64                  | tie_or_uncertain   |            7.64318  |               0.0542776   |                     15.1978      |
|           8192 |                   32 | 32                     | clear_leader       |            4.61717  |               0.45918     |                    129.939       |
|          16384 |                   32 | 32,64                  | tie_or_uncertain   |            5.83193  |               3.46942     |                    988.004       |

## Energy-runtime configuration classification

|   problem_size | tradeoff_class                    |   package_energy_exact_best_threads | package_energy_leader_threads   | package_energy_selection_status   |   runtime_exact_best_threads | runtime_leader_threads   | runtime_selection_status   | shared_leader_threads   |   energy_penalty_using_runtime_best_pct |   runtime_penalty_using_energy_best_pct | total_energy_sensitivity_same_exact_thread   |   total_energy_sensitivity_exact_best_threads | total_energy_sensitivity_leader_threads   |
|---------------:|:----------------------------------|------------------------------------:|:--------------------------------|:----------------------------------|-----------------------------:|:-------------------------|:---------------------------|:------------------------|----------------------------------------:|----------------------------------------:|:---------------------------------------------|----------------------------------------------:|:------------------------------------------|
|             64 | shared_near_optimal_configuration |                                   8 | 8,2,20,10,1,4,32,64,16          | tie_or_uncertain                  |                            8 | 8,20,2,4,10,1,32,64,16   | tie_or_uncertain           | 1,2,4,8,10,16,20,32,64  |                                 0       |                                 0       | True                                         |                                             8 | 8,2,20,10,1,4,32,64,16                    |
|            128 | shared_near_optimal_configuration |                                   4 | 4,64,16,10                      | tie_or_uncertain                  |                            4 | 4,64,16,10               | tie_or_uncertain           | 4,10,16,64              |                                 0       |                                 0       | True                                         |                                             4 | 4,64,16,10                                |
|            256 | shared_near_optimal_configuration |                                   4 | 4,8                             | tie_or_uncertain                  |                           16 | 16,8                     | tie_or_uncertain           | 8                       |                                13.1057  |                                32.8929  | True                                         |                                             4 | 4,8                                       |
|            512 | shared_near_optimal_configuration |                                  16 | 16,8                            | tie_or_uncertain                  |                           16 | 16,20,32,8               | tie_or_uncertain           | 8,16                    |                                 0       |                                 0       | True                                         |                                            16 | 16,8                                      |
|           1024 | uncertain_configuration_tradeoff  |                                  16 | 16                              | clear_leader                      |                           32 | 32,64                    | tie_or_uncertain           | none                    |                                 3.68694 |                                16.8637  | True                                         |                                            16 | 16                                        |
|           2048 | shared_near_optimal_configuration |                                  64 | 64                              | clear_leader                      |                           64 | 64,32                    | tie_or_uncertain           | 64                      |                                 0       |                                 0       | True                                         |                                            64 | 64                                        |
|           4096 | shared_near_optimal_configuration |                                  64 | 64                              | clear_leader                      |                           32 | 32,64                    | tie_or_uncertain           | 64                      |                                 7.20927 |                                 7.64318 | True                                         |                                            64 | 64                                        |
|           8192 | clear_configuration_tradeoff      |                                  64 | 64                              | clear_leader                      |                           32 | 32                       | clear_leader               | none                    |                                11.6747  |                                 4.61717 | True                                         |                                            64 | 64                                        |
|          16384 | shared_near_optimal_configuration |                                  64 | 64                              | clear_leader                      |                           32 | 32,64                    | tie_or_uncertain           | 64                      |                                 7.26083 |                                 5.83193 | True                                         |                                            64 | 64                                        |

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

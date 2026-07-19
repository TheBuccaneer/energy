# AMD GEMM scientific analysis

## Campaign and statistical unit

- Campaign: `20260719_085402`
- Raw measurements: 4050
- Sessions: 5
- Repetitions per configuration/session: 10
- Primary inferential unit: **session median**, not each raw repetition
- Confidence intervals: non-parametric bootstrap of five session medians
- Primary energy metric: `device_energy_j / batches` = package energy per GEMM

The ten adjacent repetitions within one configuration are treated as repeated
measurements under one session state. They are useful for noise and outlier
analysis, but are not counted as 50 independent experimental sessions.

## Quality indicators

- Maximum temperature: 96.0 C
- Robust outlier share: 4.99%
- Runtime-unstable configurations (robust CV >5%): 3 / 81
- Energy-unstable configurations (robust CV >8%): 0 / 81

## Package-energy leaders and tie status

|   problem_size |   exact_min_threads | leader_threads         | selection_status   |   gap_to_second_pct |   package_energy_per_op_j_median |   runtime_per_op_s_median |
|---------------:|--------------------:|:-----------------------|:-------------------|--------------------:|---------------------------------:|--------------------------:|
|             64 |                   8 | 8,16,1,64,2,32,4,20,10 | tie_or_uncertain   |            0.380905 |                      0.000445945 |               6.12708e-06 |
|            128 |                   4 | 4                      | clear_leader       |           33.8156   |                      0.0016263   |               1.4477e-05  |
|            256 |                   4 | 4                      | clear_leader       |           15.0426   |                      0.0104992   |               8.78647e-05 |
|            512 |                  16 | 16                     | clear_leader       |           14.9815   |                      0.0648808   |               0.000306523 |
|           1024 |                  16 | 16                     | clear_leader       |            5.89877  |                      0.448892    |               0.00191114  |
|           2048 |                  64 | 64                     | clear_leader       |           18.9106   |                      2.11491     |               0.00845239  |
|           4096 |                  64 | 64                     | clear_leader       |            7.72858  |                     14.104       |               0.0571822   |
|           8192 |                  64 | 64                     | clear_leader       |           13.1947   |                    116.891       |               0.488095    |
|          16384 |                  64 | 64                     | clear_leader       |            8.76481  |                    954.04        |               3.82235     |

## Runtime leaders and tie status

|   problem_size |   exact_min_threads | leader_threads         | selection_status   |   gap_to_second_pct |   runtime_per_op_s_median |   package_energy_per_op_j_median |
|---------------:|--------------------:|:-----------------------|:-------------------|--------------------:|--------------------------:|---------------------------------:|
|             64 |                   8 | 8,1,16,64,4,32,2,10,20 | tie_or_uncertain   |            0.232005 |               6.12708e-06 |                      0.000445945 |
|            128 |                   4 | 4                      | clear_leader       |           25.674    |               1.4477e-05  |                      0.0016263   |
|            256 |                  16 | 16,32,64               | tie_or_uncertain   |            3.62768  |               7.06373e-05 |                      0.0120785   |
|            512 |                  16 | 16,32                  | tie_or_uncertain   |            8.09248  |               0.000306523 |                      0.0648808   |
|           1024 |                  64 | 64,32,20,16            | tie_or_uncertain   |            1.88315  |               0.0017509   |                      0.502409    |
|           2048 |                  64 | 64,32                  | tie_or_uncertain   |            3.6493   |               0.00845239  |                      2.11491     |
|           4096 |                  32 | 32                     | clear_leader       |            7.1392   |               0.0533719   |                     15.194       |
|           8192 |                  32 | 32,64                  | tie_or_uncertain   |            4.97687  |               0.464955    |                    132.315       |
|          16384 |                  32 | 32                     | clear_leader       |            5.047    |               3.6387      |                   1037.66        |

## Interpretation constraints

1. `energy_per_op_j` from the raw CSV is not the cross-platform primary metric,
   because it is based on total RAPL energy and Intel may include a separate DRAM
   domain while AMD may not expose one. The analysis recomputes package-only energy.
2. Exact minima are not automatically called unique winners. A clear leader requires
   both a median advantage greater than 2% and a 95%
   bootstrap interval separated from every competitor. Otherwise the result is
   labeled `tie_or_uncertain`, and all practical/CI-overlapping candidates are listed.
3. Strict and practical-2% Pareto frontiers are both retained.
4. AMD 32/64-thread results characterize native capability. Intel-vs-AMD fairness
   at fixed software parallelism is handled separately by `03_compare_gemm.py`.
5. The confidence intervals quantify between-session repeatability for this
   measurement campaign, not population-wide uncertainty across machines.

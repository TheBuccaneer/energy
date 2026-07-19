# INTEL GEMM scientific analysis

## Campaign and statistical unit

- Campaign: `20260719_085511`
- Raw measurements: 3150
- Sessions: 5
- Repetitions per configuration/session: 10
- Primary inferential unit: **session median**, not each raw repetition
- Confidence intervals: non-parametric bootstrap of five session medians
- Primary energy metric: `device_energy_j / batches` = package energy per GEMM

The ten adjacent repetitions within one configuration are treated as repeated
measurements under one session state. They are useful for noise and outlier
analysis, but are not counted as 50 independent experimental sessions.

## Quality indicators

- Maximum temperature: 100.0 C
- Robust outlier share: 7.05%
- Runtime-unstable configurations (robust CV >5%): 9 / 63
- Energy-unstable configurations (robust CV >8%): 9 / 63

## Package-energy leaders and tie status

|   problem_size |   exact_min_threads | leader_threads   | selection_status   |   gap_to_second_pct |   package_energy_per_op_j_median |   runtime_per_op_s_median |
|---------------:|--------------------:|:-----------------|:-------------------|--------------------:|---------------------------------:|--------------------------:|
|             64 |                  16 | 16,20,10,1,2,8,4 | tie_or_uncertain   |            1.68576  |                      0.000122699 |               2.25466e-06 |
|            128 |                   4 | 4,1              | tie_or_uncertain   |           18.3426   |                      0.00122276  |               1.31993e-05 |
|            256 |                   8 | 8                | clear_leader       |           27.1969   |                      0.00551133  |               3.94522e-05 |
|            512 |                   8 | 8,16             | tie_or_uncertain   |            2.72874  |                      0.03789     |               0.000262924 |
|           1024 |                   8 | 8                | clear_leader       |           14.0606   |                      0.26413     |               0.00171815  |
|           2048 |                   8 | 8                | clear_leader       |           12.7966   |                      2.10921     |               0.0140408   |
|           4096 |                   8 | 8,10             | tie_or_uncertain   |            0.601498 |                     16.9458      |               0.109463    |
|           8192 |                   8 | 8,10             | tie_or_uncertain   |            0.542707 |                    127.09        |               0.867264    |
|          16384 |                   8 | 8,10             | tie_or_uncertain   |            0.566404 |                   1081.96        |               7.07207     |

## Runtime leaders and tie status

|   problem_size |   exact_min_threads | leader_threads   | selection_status   |   gap_to_second_pct |   runtime_per_op_s_median |   package_energy_per_op_j_median |
|---------------:|--------------------:|:-----------------|:-------------------|--------------------:|--------------------------:|---------------------------------:|
|             64 |                  16 | 16,1,10,20,2,8,4 | tie_or_uncertain   |             1.10389 |               2.25466e-06 |                      0.000122699 |
|            128 |                  10 | 10,8,16,20,4     | tie_or_uncertain   |             1.20411 |               1.25711e-05 |                      0.00144705  |
|            256 |                   8 | 8                | clear_leader       |            30.0592  |               3.94522e-05 |                      0.00551133  |
|            512 |                  16 | 16,8,10,20       | tie_or_uncertain   |             3.82889 |               0.000253229 |                      0.0389239   |
|           1024 |                   8 | 8                | clear_leader       |             6.77641 |               0.00171815  |                      0.26413     |
|           2048 |                   8 | 8,10             | tie_or_uncertain   |             2.93851 |               0.0140408   |                      2.10921     |
|           4096 |                  10 | 10,8             | tie_or_uncertain   |             7.81365 |               0.10153     |                     17.0477      |
|           8192 |                  10 | 10               | clear_leader       |            14.8408  |               0.755188    |                    127.78        |
|          16384 |                  10 | 10,8             | tie_or_uncertain   |             8.26432 |               6.53223     |                   1088.09        |

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

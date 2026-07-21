# STREAM scientific report — Intel Core i9-7900X

## Dataset

- Campaign: `20260720_234432`
- Raw rows: 3150
- Session medians: 315
- Configurations: 63
- Primary unit: five session medians per size/configuration
- Primary energy domain: CPU package RAPL
- Execution mode: `cpu_native`

## Immediate findings exposed by the analysis

- Clear fastest-vs-greenest configuration conflicts: **0 of 9 sizes**.
- Configurations above 5% session CV in runtime or energy: **15 of 63**.
- The analysis exposes thread-count saturation, runtime-optimal versus energy-optimal choices,
  logical-bandwidth scaling, EDP, and stability across sessions.

## Fastest-vs-greenest table

| platform   |   problem_size |   working_set_gib | energy_opt_configuration   | runtime_opt_configuration   | energy_leaders   | runtime_leaders   | leader_set_overlap   | leader_sets_disjoint   | energy_selection_status   | runtime_selection_status   |   runtime_opt_energy_penalty_pct |   runtime_opt_runtime_gain_pct | interpretation                      |
|:-----------|---------------:|------------------:|:---------------------------|:----------------------------|:-----------------|:------------------|:---------------------|:-----------------------|:--------------------------|:---------------------------|---------------------------------:|-------------------------------:|:------------------------------------|
| INTEL      |        1000000 |         0.0111759 | 10T                        | 10T                         | 10T,20T          | 10T,20T           | 10T,20T              | False                  | tie_or_uncertain          | tie_or_uncertain           |                          0       |                         0      | no_tie_aware_conflict               |
| INTEL      |        2000000 |         0.0223517 | 10T                        | 10T                         | 10T,16T,20T      | 10T,16T,20T       | 10T,16T,20T          | False                  | tie_or_uncertain          | tie_or_uncertain           |                          0       |                         0      | no_tie_aware_conflict               |
| INTEL      |        4000000 |         0.0447035 | 2T                         | 10T                         | 2T,4T            | 4T,8T,10T,16T     | 4T                   | False                  | tie_or_uncertain          | tie_or_uncertain           |                         48.9984  |                        20.2261 | no_tie_aware_conflict               |
| INTEL      |        8000000 |         0.089407  | 2T                         | 8T                          | 2T,4T            | 4T,8T             | 4T                   | False                  | tie_or_uncertain          | tie_or_uncertain           |                         42.9009  |                        18.2298 | no_tie_aware_conflict               |
| INTEL      |       16000000 |         0.178814  | 2T                         | 4T                          | 2T,4T            | 4T,8T             | 4T                   | False                  | tie_or_uncertain          | tie_or_uncertain           |                          6.35976 |                        15.504  | no_tie_aware_conflict               |
| INTEL      |       32000000 |         0.357628  | 2T                         | 4T                          | 2T               | 4T,8T             |                      | True                   | clear_leader              | tie_or_uncertain           |                          6.85829 |                        14.7901 | disjoint_but_at_least_one_uncertain |
| INTEL      |       64000000 |         0.715256  | 2T                         | 4T                          | 2T               | 4T,8T,10T         |                      | True                   | clear_leader              | tie_or_uncertain           |                          8.2506  |                        14.2009 | disjoint_but_at_least_one_uncertain |
| INTEL      |      128000000 |         1.43051   | 2T                         | 4T                          | 2T               | 4T,8T,10T,16T     |                      | True                   | clear_leader              | tie_or_uncertain           |                          8.65934 |                        13.9079 | disjoint_but_at_least_one_uncertain |
| INTEL      |      256000000 |         2.86102   | 2T                         | 8T                          | 2T,4T            | 4T,8T,10T,16T,20T | 4T                   | False                  | tie_or_uncertain          | tie_or_uncertain           |                         50.3473  |                        13.8071 | no_tie_aware_conflict               |

## Interpretation contract

1. Runtime and logical bandwidth are inverse views of the same primitive axis.
2. Energy per Triad and logical GB/J are inverse views of the same primitive axis.
3. EDP is a composite, not an independent vote.
4. `12*N` bytes are logical STREAM bytes, not measured physical DRAM/VRAM traffic.
5. Small sizes may be cache affected; large sizes are the principal memory-stress regime.
6. GPU results are resident and exclude PCIe transfers.
7. CPU/GPU energy domains are asymmetric and must be named explicitly.
8. Native-best selection is descriptive post-selection on five sessions.

## Stability rows above 5%

|   problem_size | configuration   |   num_threads |   runtime_per_op_s_session_cv_pct |   primary_energy_per_op_j_session_cv_pct |   logical_bandwidth_gb_s_session_cv_pct |   primary_power_w_session_cv_pct | runtime_stable_5pct   | energy_stable_5pct   | bandwidth_stable_5pct   |
|---------------:|:----------------|--------------:|----------------------------------:|-----------------------------------------:|----------------------------------------:|---------------------------------:|:----------------------|:---------------------|:------------------------|
|        1000000 | 10T             |            10 |                          4.70836  |                                  5.86102 |                                4.48033  |                          2.4893  | True                  | False                | True                    |
|        1000000 | 16T             |            16 |                         26.0227   |                                 23.3422  |                               19.3029   |                          2.43335 | False                 | False                | False                   |
|        1000000 | 20T             |            20 |                         13.8988   |                                 13.6825  |                               11.7233   |                          3.60206 | False                 | False                | False                   |
|        2000000 | 8T              |             8 |                          7.94293  |                                  5.9542  |                                7.41395  |                          3.38204 | False                 | False                | False                   |
|        2000000 | 10T             |            10 |                          6.97415  |                                  2.91849 |                                6.38554  |                          3.81827 | False                 | True                 | False                   |
|        2000000 | 16T             |            16 |                         15.3734   |                                 10.405   |                               13.619    |                          6.13655 | False                 | False                | False                   |
|        2000000 | 20T             |            20 |                         24.2833   |                                 18.0909  |                               20.9808   |                          8.33703 | False                 | False                | False                   |
|        4000000 | 8T              |             8 |                          0.520265 |                                  6.6352  |                                0.520353 |                          6.27875 | True                  | False                | True                    |
|        8000000 | 2T              |             2 |                          1.46284  |                                  5.70329 |                                1.47029  |                          6.19279 | True                  | False                | True                    |
|        8000000 | 16T             |            16 |                          1.11258  |                                  5.05689 |                                1.1216   |                          4.90895 | True                  | False                | True                    |
|       16000000 | 2T              |             2 |                          0.899392 |                                  5.15834 |                                0.889802 |                          5.08638 | True                  | False                | True                    |
|       64000000 | 2T              |             2 |                          0.688255 |                                  5.70439 |                                0.68851  |                          5.90214 | True                  | False                | True                    |
|       64000000 | 16T             |            16 |                          0.309876 |                                  6.74616 |                                0.310237 |                          6.58083 | True                  | False                | True                    |
|      128000000 | 20T             |            20 |                          0.513758 |                                  5.87544 |                                0.512307 |                          6.37548 | True                  | False                | True                    |
|      256000000 | 4T              |             4 |                          0.442002 |                                  5.81871 |                                0.441535 |                          6.04155 | True                  | False                | True                    |

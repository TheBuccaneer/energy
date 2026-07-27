# REDUCTION scientific report — Intel Core i9-7900X

## Dataset

- Campaign: `20260721_164732`
- Raw rows: 3150
- Session medians: 315
- Configurations: 63
- Primary unit: five session medians per size/configuration
- Primary energy domain: CPU package RAPL
- Execution mode: `cpu_native`

## Immediate findings exposed by the analysis

- Clear fastest-vs-greenest configuration conflicts: **0 of 9 sizes**.
- Configurations above 5% session CV in runtime or energy: **30 of 63**.
- The analysis exposes thread-count saturation, synchronization/aggregation cost, runtime-optimal versus energy-optimal choices,
  logical useful-data-rate scaling, EDP, and stability across sessions.

## Fastest-vs-greenest table

| platform   |   problem_size |   working_set_gib | energy_opt_configuration   | runtime_opt_configuration   | energy_leaders   | runtime_leaders   | leader_set_overlap   | leader_sets_disjoint   | energy_selection_status   | runtime_selection_status   |   runtime_opt_energy_penalty_pct |   runtime_opt_runtime_gain_pct | interpretation                      |
|:-----------|---------------:|------------------:|:---------------------------|:----------------------------|:-----------------|:------------------|:---------------------|:-----------------------|:--------------------------|:---------------------------|---------------------------------:|-------------------------------:|:------------------------------------|
| INTEL      |        1000000 |        0.00372529 | 20T                        | 20T                         | 8T,10T,16T,20T   | 8T,10T,16T,20T    | 10T,16T,20T,8T       | False                  | tie_or_uncertain          | tie_or_uncertain           |                          0       |                       0        | no_tie_aware_conflict               |
| INTEL      |        2000000 |        0.00745058 | 20T                        | 10T                         | 8T,10T,16T,20T   | 10T,16T,20T       | 10T,16T,20T          | False                  | tie_or_uncertain          | tie_or_uncertain           |                          8.18604 |                       4.77131  | no_tie_aware_conflict               |
| INTEL      |        4000000 |        0.0149012  | 16T                        | 10T                         | 8T,10T,16T,20T   | 10T,20T           | 10T,20T              | False                  | tie_or_uncertain          | tie_or_uncertain           |                         15.7912  |                       5.56219  | no_tie_aware_conflict               |
| INTEL      |        8000000 |        0.0298023  | 4T                         | 16T                         | 4T,16T           | 10T,16T           | 16T                  | False                  | tie_or_uncertain          | tie_or_uncertain           |                         11.4455  |                      20.1579   | no_tie_aware_conflict               |
| INTEL      |       16000000 |        0.0596046  | 4T                         | 8T                          | 4T               | 8T,10T            |                      | True                   | clear_leader              | tie_or_uncertain           |                         31.6217  |                       5.14305  | disjoint_but_at_least_one_uncertain |
| INTEL      |       32000000 |        0.119209   | 4T                         | 8T                          | 2T,4T            | 8T                |                      | True                   | tie_or_uncertain          | clear_leader               |                         30.0419  |                       3.0053   | disjoint_but_at_least_one_uncertain |
| INTEL      |       64000000 |        0.238419   | 4T                         | 8T                          | 2T,4T            | 4T,8T             | 4T                   | False                  | tie_or_uncertain          | tie_or_uncertain           |                         36.8039  |                       1.32626  | no_tie_aware_conflict               |
| INTEL      |      128000000 |        0.476837   | 4T                         | 8T                          | 2T,4T            | 4T,8T             | 4T                   | False                  | tie_or_uncertain          | tie_or_uncertain           |                         36.9115  |                       0.82487  | no_tie_aware_conflict               |
| INTEL      |      256000000 |        0.953674   | 4T                         | 8T                          | 4T               | 4T,8T,10T,16T,20T | 4T                   | False                  | clear_leader              | tie_or_uncertain           |                         39.4493  |                       0.243585 | no_tie_aware_conflict               |

## Interpretation contract

1. Runtime and logical useful-data rate are inverse views of the same primitive axis.
2. Energy per reduction and logical GB/J are inverse views of the same primitive axis.
3. EDP is a composite, not an independent vote.
4. `4*N+4` bytes are logical REDUCTION bytes, not measured physical DRAM/VRAM traffic.
5. Small sizes may be cache and synchronization affected; large sizes are the principal resident memory-plus-aggregation regime.
6. GPU results are resident and exclude PCIe transfers.
7. CPU/GPU energy domains are asymmetric and must be named explicitly.
8. Native-best selection is descriptive post-selection on five sessions.

## Stability rows above 5%

|   problem_size | configuration   |   num_threads |   runtime_per_op_s_session_cv_pct |   primary_energy_per_op_j_session_cv_pct |   logical_bandwidth_gb_s_session_cv_pct |   primary_power_w_session_cv_pct | runtime_stable_5pct   | energy_stable_5pct   | bandwidth_stable_5pct   |
|---------------:|:----------------|--------------:|----------------------------------:|-----------------------------------------:|----------------------------------------:|---------------------------------:|:----------------------|:---------------------|:------------------------|
|        1000000 | 16T             |            16 |                        6.28097    |                                  5.15163 |                              6.42794    |                          3.26268 | False                 | False                | False                   |
|        1000000 | 20T             |            20 |                       21.8357     |                                 23.8191  |                             20.2213     |                          6.8157  | False                 | False                | False                   |
|        2000000 | 4T              |             4 |                        3.65476    |                                  7.17092 |                              3.62215    |                          8.02226 | True                  | False                | True                    |
|        2000000 | 8T              |             8 |                        1.10265    |                                  6.57259 |                              1.10763    |                          5.55383 | True                  | False                | True                    |
|        2000000 | 10T             |            10 |                        1.52156    |                                  5.14075 |                              1.51136    |                          4.06685 | True                  | False                | True                    |
|        2000000 | 16T             |            16 |                       17.1994     |                                 13.9795  |                             22.2632     |                          5.9983  | False                 | False                | False                   |
|        2000000 | 20T             |            20 |                       11.3396     |                                 11.473   |                             10.37       |                          2.21162 | False                 | False                | False                   |
|        4000000 | 1T              |             1 |                        1.50612    |                                  7.15097 |                              1.49886    |                          8.2228  | True                  | False                | True                    |
|        4000000 | 10T             |            10 |                        2.02299    |                                  5.55721 |                              2.03593    |                          4.81949 | True                  | False                | True                    |
|        4000000 | 20T             |            20 |                        6.99265    |                                 11.3639  |                              7.63186    |                          7.21699 | False                 | False                | False                   |
|        8000000 | 1T              |             1 |                        0.765089   |                                  6.1439  |                              0.766269   |                          6.45225 | True                  | False                | True                    |
|        8000000 | 10T             |            10 |                        0.515437   |                                  8.46196 |                              0.515513   |                          8.15274 | True                  | False                | True                    |
|        8000000 | 16T             |            16 |                        0.319444   |                                  5.35491 |                              0.319366   |                          5.45256 | True                  | False                | True                    |
|       16000000 | 4T              |             4 |                        0.748972   |                                  7.46977 |                              0.751403   |                          7.6005  | True                  | False                | True                    |
|       16000000 | 10T             |            10 |                        0.14276    |                                  5.4618  |                              0.142721   |                          5.40714 | True                  | False                | True                    |
|       32000000 | 2T              |             2 |                        0.552125   |                                  8.94792 |                              0.550652   |                          8.87852 | True                  | False                | True                    |
|       32000000 | 10T             |            10 |                        0.0692242  |                                  8.51629 |                              0.0692366  |                          8.49565 | True                  | False                | True                    |
|       32000000 | 16T             |            16 |                        0.0413509  |                                  7.33651 |                              0.0413346  |                          7.35064 | True                  | False                | True                    |
|       64000000 | 1T              |             1 |                        0.0758756  |                                  8.53283 |                              0.0758859  |                          8.4959  | True                  | False                | True                    |
|       64000000 | 2T              |             2 |                        0.468349   |                                  8.42547 |                              0.466267   |                          8.54845 | True                  | False                | True                    |
|       64000000 | 10T             |            10 |                        0.0553848  |                                  7.26761 |                              0.0553505  |                          7.31377 | True                  | False                | True                    |
|       64000000 | 16T             |            16 |                        0.154744   |                                  5.7905  |                              0.154693   |                          5.69513 | True                  | False                | True                    |
|       64000000 | 20T             |            20 |                        0.0695361  |                                  7.55457 |                              0.0694894  |                          7.53922 | True                  | False                | True                    |
|      128000000 | 2T              |             2 |                        0.467833   |                                  5.04062 |                              0.467536   |                          5.61226 | True                  | False                | True                    |
|      128000000 | 4T              |             4 |                        0.0992073  |                                  7.29341 |                              0.0993324  |                          7.38048 | True                  | False                | True                    |
|      128000000 | 16T             |            16 |                        0.0893442  |                                  6.27615 |                              0.0893301  |                          6.30274 | True                  | False                | True                    |
|      128000000 | 20T             |            20 |                        0.00497954 |                                  5.28583 |                              0.00497913 |                          5.29646 | True                  | False                | True                    |
|      256000000 | 4T              |             4 |                        0.183571   |                                  7.0141  |                              0.183718   |                          7.07233 | True                  | False                | True                    |
|      256000000 | 10T             |            10 |                        0.025012   |                                  7.87673 |                              0.0250169  |                          7.89382 | True                  | False                | True                    |
|      256000000 | 16T             |            16 |                        0.0547668  |                                  6.36734 |                              0.0547661  |                          6.3764  | True                  | False                | True                    |

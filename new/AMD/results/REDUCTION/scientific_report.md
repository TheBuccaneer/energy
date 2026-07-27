# REDUCTION scientific report — AMD Threadripper 3970X

## Dataset

- Campaign: `20260722_174206`
- Raw rows: 4050
- Session medians: 405
- Configurations: 81
- Primary unit: five session medians per size/configuration
- Primary energy domain: CPU package RAPL
- Execution mode: `cpu_native`

## Immediate findings exposed by the analysis

- Clear fastest-vs-greenest configuration conflicts: **2 of 9 sizes**.
- Configurations above 5% session CV in runtime or energy: **27 of 81**.
- The analysis exposes thread-count saturation, synchronization/aggregation cost, runtime-optimal versus energy-optimal choices,
  logical useful-data-rate scaling, EDP, and stability across sessions.

## Fastest-vs-greenest table

| platform   |   problem_size |   working_set_gib | energy_opt_configuration   | runtime_opt_configuration   | energy_leaders               | runtime_leaders           | leader_set_overlap        | leader_sets_disjoint   | energy_selection_status   | runtime_selection_status   |   runtime_opt_energy_penalty_pct |   runtime_opt_runtime_gain_pct | interpretation               |
|:-----------|---------------:|------------------:|:---------------------------|:----------------------------|:-----------------------------|:--------------------------|:--------------------------|:-----------------------|:--------------------------|:---------------------------|---------------------------------:|-------------------------------:|:-----------------------------|
| AMD        |        1000000 |        0.00372529 | 8T                         | 8T                          | 8T,10T,16T,20T,32T           | 8T,10T,16T,20T,32T        | 10T,16T,20T,32T,8T        | False                  | tie_or_uncertain          | tie_or_uncertain           |                         0        |                        0       | no_tie_aware_conflict        |
| AMD        |        2000000 |        0.00745058 | 8T                         | 10T                         | 8T,10T,20T,32T               | 10T,16T,20T,32T           | 10T,20T,32T               | False                  | tie_or_uncertain          | tie_or_uncertain           |                         0.654418 |                        4.93208 | no_tie_aware_conflict        |
| AMD        |        4000000 |        0.0149012  | 64T                        | 64T                         | 2T,4T,8T,10T,16T,20T,32T,64T | 4T,8T,10T,16T,20T,32T,64T | 10T,16T,20T,32T,4T,64T,8T | False                  | tie_or_uncertain          | tie_or_uncertain           |                         0        |                        0       | no_tie_aware_conflict        |
| AMD        |        8000000 |        0.0298023  | 16T                        | 20T                         | 8T,16T,20T,64T               | 20T,64T                   | 20T,64T                   | False                  | tie_or_uncertain          | tie_or_uncertain           |                         1.33726  |                        4.9994  | no_tie_aware_conflict        |
| AMD        |       16000000 |        0.0596046  | 16T                        | 32T                         | 16T,32T,64T                  | 32T,64T                   | 32T,64T                   | False                  | tie_or_uncertain          | tie_or_uncertain           |                         1.51018  |                       18.8713  | no_tie_aware_conflict        |
| AMD        |       32000000 |        0.119209   | 32T                        | 32T                         | 16T,32T,64T                  | 32T,64T                   | 32T,64T                   | False                  | tie_or_uncertain          | tie_or_uncertain           |                         0        |                        0       | no_tie_aware_conflict        |
| AMD        |       64000000 |        0.238419   | 4T                         | 10T                         | 4T,10T                       | 10T,16T,32T               | 10T                       | False                  | tie_or_uncertain          | tie_or_uncertain           |                         8.13476  |                       17.2909  | no_tie_aware_conflict        |
| AMD        |      128000000 |        0.476837   | 4T                         | 16T                         | 4T                           | 16T                       |                           | True                   | clear_leader              | clear_leader               |                        25.4799   |                       13.6704  | clear_configuration_tradeoff |
| AMD        |      256000000 |        0.953674   | 4T                         | 16T                         | 4T                           | 16T                       |                           | True                   | clear_leader              | clear_leader               |                        30.9854   |                       12.0455  | clear_configuration_tradeoff |

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
|        1000000 | 2T              |             2 |                         7.38864   |                                  7.53552 |                               8.1164    |                          1.44198 | False                 | False                | False                   |
|        1000000 | 16T             |            16 |                        37.6594    |                                 34.6825  |                              75.8326    |                          7.58558 | False                 | False                | False                   |
|        1000000 | 20T             |            20 |                        40.8397    |                                 38.3002  |                              90.2685    |                          7.88206 | False                 | False                | False                   |
|        1000000 | 32T             |            32 |                        47.8421    |                                 45.5181  |                             133.28      |                         12.6025  | False                 | False                | False                   |
|        2000000 | 20T             |            20 |                        36.1795    |                                 31.814   |                              70.2633    |                         10.3489  | False                 | False                | False                   |
|        2000000 | 32T             |            32 |                        44.6392    |                                 40.7097  |                             111.268     |                         15.335   | False                 | False                | False                   |
|        4000000 | 8T              |             8 |                        13.1579    |                                  9.84311 |                              12.9682    |                          3.35332 | False                 | False                | False                   |
|        4000000 | 10T             |            10 |                        14.6055    |                                 11.2066  |                              18.1538    |                          4.29023 | False                 | False                | False                   |
|        4000000 | 16T             |            16 |                        23.9695    |                                 17.9958  |                              35.2053    |                          9.30417 | False                 | False                | False                   |
|        4000000 | 20T             |            20 |                        29.0141    |                                 22.1442  |                              47.3325    |                         12.1238  | False                 | False                | False                   |
|        4000000 | 32T             |            32 |                        39.8263    |                                 33.3492  |                              84.9561    |                         17.8078  | False                 | False                | False                   |
|        4000000 | 64T             |            64 |                       104.547     |                                 87.8638  |                              79.1784    |                         27.1601  | False                 | False                | False                   |
|        8000000 | 8T              |             8 |                         6.38953   |                                  4.81707 |                               6.9869    |                          1.77621 | False                 | True                 | False                   |
|        8000000 | 20T             |            20 |                        20.3968    |                                 13.8411  |                              28.0402    |                          9.23012 | False                 | False                | False                   |
|        8000000 | 64T             |            64 |                        65.3716    |                                 53.0365  |                              86.0293    |                         25.3804  | False                 | False                | False                   |
|       16000000 | 4T              |             4 |                         2.88384   |                                  7.21702 |                               2.92567   |                          5.63475 | True                  | False                | True                    |
|       16000000 | 10T             |            10 |                         3.91216   |                                  6.87675 |                               3.86425   |                          3.55444 | True                  | False                | True                    |
|       16000000 | 64T             |            64 |                        38.7336    |                                 33.394   |                              80.6174    |                         14.1536  | False                 | False                | False                   |
|       32000000 | 8T              |             8 |                         2.75195   |                                  6.2744  |                               2.69448   |                          5.38107 | True                  | False                | True                    |
|       32000000 | 64T             |            64 |                        11.9897    |                                  8.76125 |                              14.1575    |                          3.92944 | False                 | False                | False                   |
|       64000000 | 8T              |             8 |                         0.251918  |                                  6.21608 |                               0.251774  |                          6.214   | True                  | False                | True                    |
|       64000000 | 10T             |            10 |                         0.205452  |                                  6.52417 |                               0.205495  |                          6.6022  | True                  | False                | True                    |
|      128000000 | 8T              |             8 |                         0.126334  |                                  7.9548  |                               0.126358  |                          7.92307 | True                  | False                | True                    |
|      128000000 | 10T             |            10 |                         0.104078  |                                  8.76212 |                               0.104051  |                          8.85204 | True                  | False                | True                    |
|      256000000 | 8T              |             8 |                         0.164811  |                                  8.93387 |                               0.164639  |                          8.81886 | True                  | False                | True                    |
|      256000000 | 10T             |            10 |                         0.170479  |                                  9.90286 |                               0.170401  |                         10.077   | True                  | False                | True                    |
|      256000000 | 16T             |            16 |                         0.0322167 |                                  5.78134 |                               0.0322194 |                          5.74602 | True                  | False                | True                    |

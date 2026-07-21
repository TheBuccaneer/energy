# STREAM scientific report — AMD Threadripper 3970X

## Dataset

- Campaign: `20260721_095517`
- Raw rows: 4050
- Session medians: 405
- Configurations: 81
- Primary unit: five session medians per size/configuration
- Primary energy domain: CPU package RAPL
- Execution mode: `cpu_native`

## Immediate findings exposed by the analysis

- Clear fastest-vs-greenest configuration conflicts: **0 of 9 sizes**.
- Configurations above 5% session CV in runtime or energy: **24 of 81**.
- The analysis exposes thread-count saturation, runtime-optimal versus energy-optimal choices,
  logical-bandwidth scaling, EDP, and stability across sessions.

## Fastest-vs-greenest table

| platform   |   problem_size |   working_set_gib | energy_opt_configuration   | runtime_opt_configuration   | energy_leaders     | runtime_leaders   | leader_set_overlap   | leader_sets_disjoint   | energy_selection_status   | runtime_selection_status   |   runtime_opt_energy_penalty_pct |   runtime_opt_runtime_gain_pct | interpretation        |
|:-----------|---------------:|------------------:|:---------------------------|:----------------------------|:-------------------|:------------------|:---------------------|:-----------------------|:--------------------------|:---------------------------|---------------------------------:|-------------------------------:|:----------------------|
| AMD        |        1000000 |         0.0111759 | 10T                        | 16T                         | 8T,10T,16T,20T,64T | 16T,20T,64T       | 16T,20T,64T          | False                  | tie_or_uncertain          | tie_or_uncertain           |                         2.23288  |                       3.23505  | no_tie_aware_conflict |
| AMD        |        2000000 |         0.0223517 | 16T                        | 20T                         | 10T,16T,20T,64T    | 16T,20T,64T       | 16T,20T,64T          | False                  | tie_or_uncertain          | tie_or_uncertain           |                         3.00111  |                       0.417935 | no_tie_aware_conflict |
| AMD        |        4000000 |         0.0447035 | 16T                        | 32T                         | 16T,32T            | 32T               | 32T                  | False                  | tie_or_uncertain          | clear_leader               |                         0.265366 |                       7.27134  | no_tie_aware_conflict |
| AMD        |        8000000 |         0.089407  | 32T                        | 32T                         | 32T,64T            | 32T               | 32T                  | False                  | tie_or_uncertain          | clear_leader               |                         0        |                       0        | no_tie_aware_conflict |
| AMD        |       16000000 |         0.178814  | 10T                        | 10T                         | 8T,10T             | 10T               | 10T                  | False                  | tie_or_uncertain          | clear_leader               |                         0        |                       0        | no_tie_aware_conflict |
| AMD        |       32000000 |         0.357628  | 4T                         | 4T                          | 2T,4T              | 4T,8T             | 4T                   | False                  | tie_or_uncertain          | tie_or_uncertain           |                         0        |                       0        | no_tie_aware_conflict |
| AMD        |       64000000 |         0.715256  | 4T                         | 4T                          | 4T                 | 4T,8T             | 4T                   | False                  | clear_leader              | tie_or_uncertain           |                         0        |                       0        | no_tie_aware_conflict |
| AMD        |      128000000 |         1.43051   | 4T                         | 4T                          | 4T                 | 4T,8T             | 4T                   | False                  | clear_leader              | tie_or_uncertain           |                         0        |                       0        | no_tie_aware_conflict |
| AMD        |      256000000 |         2.86102   | 4T                         | 4T                          | 4T                 | 4T,8T             | 4T                   | False                  | clear_leader              | tie_or_uncertain           |                         0        |                       0        | no_tie_aware_conflict |

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
|        1000000 | 20T             |            20 |                        19.8647    |                                 13.2448  |                              26.9219    |                          9.21503 | False                 | False                | False                   |
|        1000000 | 64T             |            64 |                        69.3641    |                                 64.6357  |                              96.3049    |                         12.1786  | False                 | False                | False                   |
|        2000000 | 2T              |             2 |                         5.30841   |                                  5.95741 |                               5.25319   |                          3.4761  | False                 | False                | False                   |
|        2000000 | 64T             |            64 |                        31.1132    |                                 28.6523  |                              53.468     |                          5.34986 | False                 | False                | False                   |
|        4000000 | 4T              |             4 |                         9.39157   |                                 10.2327  |                               9.70252   |                          4.36825 | False                 | False                | False                   |
|        4000000 | 10T             |            10 |                         5.77431   |                                  2.79525 |                               5.41524   |                          2.85334 | False                 | True                 | False                   |
|        4000000 | 32T             |            32 |                        23.9493    |                                 22.6876  |                              35.2776    |                          2.01294 | False                 | False                | False                   |
|        8000000 | 4T              |             4 |                         3.73165   |                                  6.8456  |                               3.88127   |                          3.7437  | True                  | False                | True                    |
|        8000000 | 8T              |             8 |                         7.008     |                                  5.19685 |                               7.01369   |                          6.15221 | False                 | False                | False                   |
|        8000000 | 16T             |            16 |                        26.5483    |                                 14.1648  |                              24.8338    |                         12.0596  | False                 | False                | False                   |
|        8000000 | 32T             |            32 |                         8.98998   |                                  6.27151 |                               8.34107   |                          3.24405 | False                 | False                | False                   |
|        8000000 | 64T             |            64 |                         8.10858   |                                 11.2738  |                               7.76873   |                          6.62234 | False                 | False                | False                   |
|       16000000 | 8T              |             8 |                         0.879197  |                                  5.83682 |                               0.880693  |                          6.64515 | True                  | False                | True                    |
|       16000000 | 10T             |            10 |                         2.23619   |                                  5.36387 |                               2.19635   |                          6.31103 | True                  | False                | True                    |
|       16000000 | 20T             |            20 |                         0.864198  |                                  5.14485 |                               0.869483  |                          4.54125 | True                  | False                | True                    |
|       32000000 | 8T              |             8 |                         0.883287  |                                  9.17586 |                               0.874398  |                          9.64808 | True                  | False                | True                    |
|       32000000 | 10T             |            10 |                         0.122758  |                                  7.83728 |                               0.122934  |                          7.9102  | True                  | False                | True                    |
|       64000000 | 8T              |             8 |                         0.29382   |                                  9.79245 |                               0.29338   |                          9.83819 | True                  | False                | True                    |
|       64000000 | 10T             |            10 |                         0.210466  |                                  8.8908  |                               0.210277  |                          9.03335 | True                  | False                | True                    |
|      128000000 | 8T              |             8 |                         0.2035    |                                  8.82278 |                               0.20331   |                          9.01119 | True                  | False                | True                    |
|      128000000 | 10T             |            10 |                         0.157182  |                                  7.73884 |                               0.157021  |                          7.85137 | True                  | False                | True                    |
|      128000000 | 20T             |            20 |                         0.0749275 |                                  5.16719 |                               0.0749237 |                          5.16624 | True                  | False                | True                    |
|      256000000 | 8T              |             8 |                         0.0790888 |                                  8.45138 |                               0.0790936 |                          8.51965 | True                  | False                | True                    |
|      256000000 | 10T             |            10 |                         0.0701112 |                                  7.12283 |                               0.0701476 |                          7.17706 | True                  | False                | True                    |

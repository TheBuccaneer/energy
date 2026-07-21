# STREAM scientific report — RTX 5060 Ti

## Dataset

- Campaign: `20260721_114533`
- Raw rows: 450
- Session medians: 45
- Configurations: 9
- Primary unit: five session medians per size/configuration
- Primary energy domain: GPU board NVML
- Execution mode: `gpu_resident`

## Immediate findings exposed by the analysis

- Clear fastest-vs-greenest configuration conflicts: **0 of 9 sizes**.
- Configurations above 5% session CV in runtime or energy: **0 of 9**.
- The analysis exposes thread-count saturation, runtime-optimal versus energy-optimal choices,
  logical-bandwidth scaling, EDP, and stability across sessions.

## Fastest-vs-greenest table

| platform   |   problem_size |   working_set_gib | energy_opt_configuration   | runtime_opt_configuration   | energy_leaders   | runtime_leaders   | leader_set_overlap   | leader_sets_disjoint   | energy_selection_status   | runtime_selection_status   |   runtime_opt_energy_penalty_pct |   runtime_opt_runtime_gain_pct | interpretation        |
|:-----------|---------------:|------------------:|:---------------------------|:----------------------------|:-----------------|:------------------|:---------------------|:-----------------------|:--------------------------|:---------------------------|---------------------------------:|-------------------------------:|:----------------------|
| 5060ti     |        1000000 |         0.0111759 | gpu_resident               | gpu_resident                | gpu_resident     | gpu_resident      | gpu_resident         | False                  | clear_leader              | clear_leader               |                                0 |                              0 | no_tie_aware_conflict |
| 5060ti     |        2000000 |         0.0223517 | gpu_resident               | gpu_resident                | gpu_resident     | gpu_resident      | gpu_resident         | False                  | clear_leader              | clear_leader               |                                0 |                              0 | no_tie_aware_conflict |
| 5060ti     |        4000000 |         0.0447035 | gpu_resident               | gpu_resident                | gpu_resident     | gpu_resident      | gpu_resident         | False                  | clear_leader              | clear_leader               |                                0 |                              0 | no_tie_aware_conflict |
| 5060ti     |        8000000 |         0.089407  | gpu_resident               | gpu_resident                | gpu_resident     | gpu_resident      | gpu_resident         | False                  | clear_leader              | clear_leader               |                                0 |                              0 | no_tie_aware_conflict |
| 5060ti     |       16000000 |         0.178814  | gpu_resident               | gpu_resident                | gpu_resident     | gpu_resident      | gpu_resident         | False                  | clear_leader              | clear_leader               |                                0 |                              0 | no_tie_aware_conflict |
| 5060ti     |       32000000 |         0.357628  | gpu_resident               | gpu_resident                | gpu_resident     | gpu_resident      | gpu_resident         | False                  | clear_leader              | clear_leader               |                                0 |                              0 | no_tie_aware_conflict |
| 5060ti     |       64000000 |         0.715256  | gpu_resident               | gpu_resident                | gpu_resident     | gpu_resident      | gpu_resident         | False                  | clear_leader              | clear_leader               |                                0 |                              0 | no_tie_aware_conflict |
| 5060ti     |      128000000 |         1.43051   | gpu_resident               | gpu_resident                | gpu_resident     | gpu_resident      | gpu_resident         | False                  | clear_leader              | clear_leader               |                                0 |                              0 | no_tie_aware_conflict |
| 5060ti     |      256000000 |         2.86102   | gpu_resident               | gpu_resident                | gpu_resident     | gpu_resident      | gpu_resident         | False                  | clear_leader              | clear_leader               |                                0 |                              0 | no_tie_aware_conflict |

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

_None._

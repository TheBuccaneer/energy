# REDUCTION scientific report — RTX 3090

## Dataset

- Campaign: `20260722_191930`
- Raw rows: 450
- Session medians: 45
- Configurations: 9
- Primary unit: five session medians per size/configuration
- Primary energy domain: GPU board NVML
- Execution mode: `gpu_resident`

## Immediate findings exposed by the analysis

- Clear fastest-vs-greenest configuration conflicts: **0 of 9 sizes**.
- Configurations above 5% session CV in runtime or energy: **0 of 9**.
- The analysis exposes thread-count saturation, synchronization/aggregation cost, runtime-optimal versus energy-optimal choices,
  logical useful-data-rate scaling, EDP, and stability across sessions.

## Fastest-vs-greenest table

|   platform |   problem_size |   working_set_gib | energy_opt_configuration   | runtime_opt_configuration   | energy_leaders   | runtime_leaders   | leader_set_overlap   | leader_sets_disjoint   | energy_selection_status   | runtime_selection_status   |   runtime_opt_energy_penalty_pct |   runtime_opt_runtime_gain_pct | interpretation        |
|-----------:|---------------:|------------------:|:---------------------------|:----------------------------|:-----------------|:------------------|:---------------------|:-----------------------|:--------------------------|:---------------------------|---------------------------------:|-------------------------------:|:----------------------|
|       3090 |        1000000 |        0.00372529 | gpu_resident               | gpu_resident                | gpu_resident     | gpu_resident      | gpu_resident         | False                  | clear_leader              | clear_leader               |                                0 |                              0 | no_tie_aware_conflict |
|       3090 |        2000000 |        0.00745058 | gpu_resident               | gpu_resident                | gpu_resident     | gpu_resident      | gpu_resident         | False                  | clear_leader              | clear_leader               |                                0 |                              0 | no_tie_aware_conflict |
|       3090 |        4000000 |        0.0149012  | gpu_resident               | gpu_resident                | gpu_resident     | gpu_resident      | gpu_resident         | False                  | clear_leader              | clear_leader               |                                0 |                              0 | no_tie_aware_conflict |
|       3090 |        8000000 |        0.0298023  | gpu_resident               | gpu_resident                | gpu_resident     | gpu_resident      | gpu_resident         | False                  | clear_leader              | clear_leader               |                                0 |                              0 | no_tie_aware_conflict |
|       3090 |       16000000 |        0.0596046  | gpu_resident               | gpu_resident                | gpu_resident     | gpu_resident      | gpu_resident         | False                  | clear_leader              | clear_leader               |                                0 |                              0 | no_tie_aware_conflict |
|       3090 |       32000000 |        0.119209   | gpu_resident               | gpu_resident                | gpu_resident     | gpu_resident      | gpu_resident         | False                  | clear_leader              | clear_leader               |                                0 |                              0 | no_tie_aware_conflict |
|       3090 |       64000000 |        0.238419   | gpu_resident               | gpu_resident                | gpu_resident     | gpu_resident      | gpu_resident         | False                  | clear_leader              | clear_leader               |                                0 |                              0 | no_tie_aware_conflict |
|       3090 |      128000000 |        0.476837   | gpu_resident               | gpu_resident                | gpu_resident     | gpu_resident      | gpu_resident         | False                  | clear_leader              | clear_leader               |                                0 |                              0 | no_tie_aware_conflict |
|       3090 |      256000000 |        0.953674   | gpu_resident               | gpu_resident                | gpu_resident     | gpu_resident      | gpu_resident         | False                  | clear_leader              | clear_leader               |                                0 |                              0 | no_tie_aware_conflict |

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

_None._

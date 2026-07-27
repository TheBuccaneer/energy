# All-platform REDUCTION audit report

## Input campaigns

| platform   |        campaign |   raw_rows |   session_median_rows |   configurations | energy_domain    | result_directory                                        |
|:-----------|----------------:|-----------:|----------------------:|-----------------:|:-----------------|:--------------------------------------------------------|
| INTEL      | 20260721_164732 |       3150 |                   315 |               63 | CPU package RAPL | /home/rock/projects/energy/new/INTEL/results/REDUCTION  |
| AMD        | 20260722_174206 |       4050 |                   405 |               81 | CPU package RAPL | /home/rock/projects/energy/new/AMD/results/REDUCTION    |
| 3090       | 20260722_191930 |        450 |                    45 |                9 | GPU board NVML   | /home/rock/projects/energy/new/3090/results/REDUCTION   |
| 5060ti     | 20260722_050953 |        450 |                    45 |                9 | GPU board NVML   | /home/rock/projects/energy/new/5060ti/results/REDUCTION |

## Aggregate dimensions

- Unified session medians: 810 rows
- Unified configurations: 162 rows
- Native policy leaders: 180 rows
- Selected policy session medians: 900 rows
- Pairwise native-best comparisons: 270 rows
- Global metric winners: 45 rows
- Exact-winner regret rows: 36 rows

## Immediate indicators

- Clear device-level fastest-vs-greenest conflicts: 6
- Clear tie-aware within-platform configuration conflicts: 2
- Clear descriptive exact-winner energy penalties: 8
- Configurations above 5% CV in runtime or energy: 57

## Interpretation

The primary decision axes are runtime and measured device-domain energy. Logical useful-data rate and logical GB/J are normalized inverse views. EDP is joint. No physical traffic claim is made from `4*N+4`, because implementation-specific CPU partials, CUB workspace and memory-system effects are not hardware-counter measurements.

# Intel–AMD GEMM comparison

- Intel campaign: `20260719_085511`
- AMD campaign: `20260719_085402`
- Common-thread comparison: threads [1, 2, 4, 8, 10, 16, 20]
- Native-best comparison: Intel may use up to 20 threads; AMD may use 32/64
- Equivalence tolerance for qualitative classification: ±2%
- Primary energy metric: package-only `device_energy_j / batches`

## Why there are two comparisons

The **common-thread** view holds software parallelism constant and is the cleaner
hardware comparison at matched thread counts. The **native-best** view asks what
each processor can achieve when allowed to use its available thread grid. These
answer different research questions and must not be merged into one winner count.

## Common-thread trade-off counts

| class                                          |   count |
|:-----------------------------------------------|--------:|
| Intel dominant                                 |      53 |
| Intel energy-efficient / AMD runtime-efficient |       6 |
| AMD dominant                                   |       4 |

## Native-best comparison by size

|   problem_size |   intel_energy_best_threads | intel_energy_leader_threads   | intel_energy_selection_status   |   amd_energy_best_threads | amd_energy_leader_threads   | amd_energy_selection_status   |   intel_over_amd_energy_best_ratio |   intel_runtime_best_threads | intel_runtime_leader_threads   | intel_runtime_selection_status   |   amd_runtime_best_threads | amd_runtime_leader_threads   | amd_runtime_selection_status   |   intel_over_amd_runtime_best_ratio |
|---------------:|----------------------------:|:------------------------------|:--------------------------------|--------------------------:|:----------------------------|:------------------------------|-----------------------------------:|-----------------------------:|:-------------------------------|:---------------------------------|---------------------------:|:-----------------------------|:-------------------------------|------------------------------------:|
|             64 |                          16 | 16,20,10,1,2,8,4              | tie_or_uncertain                |                         8 | 8,16,1,64,2,32,4,20,10      | tie_or_uncertain              |                           0.275143 |                           16 | 16,1,10,20,2,8,4               | tie_or_uncertain                 |                          8 | 8,1,16,64,4,32,2,10,20       | tie_or_uncertain               |                            0.367984 |
|            128 |                           4 | 4,1                           | tie_or_uncertain                |                         4 | 4                           | clear_leader                  |                           0.751868 |                           10 | 10,8,16,20,4                   | tie_or_uncertain                 |                          4 | 4                            | clear_leader                   |                            0.868347 |
|            256 |                           8 | 8                             | clear_leader                    |                         4 | 4                           | clear_leader                  |                           0.52493  |                            8 | 8                              | clear_leader                     |                         16 | 16,32,64                     | tie_or_uncertain               |                            0.558518 |
|            512 |                           8 | 8,16                          | tie_or_uncertain                |                        16 | 16                          | clear_leader                  |                           0.583993 |                           16 | 16,8,10,20                     | tie_or_uncertain                 |                         16 | 16,32                        | tie_or_uncertain               |                            0.826132 |
|           1024 |                           8 | 8                             | clear_leader                    |                        16 | 16                          | clear_leader                  |                           0.588403 |                            8 | 8                              | clear_leader                     |                         64 | 64,32,20,16                  | tie_or_uncertain               |                            0.981298 |
|           2048 |                           8 | 8                             | clear_leader                    |                        64 | 64                          | clear_leader                  |                           0.997306 |                            8 | 8,10                           | tie_or_uncertain                 |                         64 | 64,32                        | tie_or_uncertain               |                            1.66116  |
|           4096 |                           8 | 8,10                          | tie_or_uncertain                |                        64 | 64                          | clear_leader                  |                           1.20149  |                           10 | 10,8                           | tie_or_uncertain                 |                         32 | 32                           | clear_leader                   |                            1.90231  |
|           8192 |                           8 | 8,10                          | tie_or_uncertain                |                        64 | 64                          | clear_leader                  |                           1.08725  |                           10 | 10                             | clear_leader                     |                         32 | 32,64                        | tie_or_uncertain               |                            1.62422  |
|          16384 |                           8 | 8,10                          | tie_or_uncertain                |                        64 | 64                          | clear_leader                  |                           1.13408  |                           10 | 10,8                           | tie_or_uncertain                 |                         32 | 32                           | clear_leader                   |                            1.79521  |

## Interpretation

- A ratio below 1 favors Intel; above 1 favors AMD.
- Dominance requires both lower package energy and lower runtime by more than 2%.
- Native-best thread fields retain the exact observed minimum, but the accompanying
  leader sets/status prevent a sub-2% or CI-overlapping difference from being
  presented as a unique thread-count winner.
- A trade-off class is not a failure: it identifies whether energy or runtime is
  being exchanged and is more informative than a single winner count.
- Package energy domains are not physically identical across CPU vendors and do
  not include the same external memory/system components. Results support
  package-level placement decisions, not whole-system energy claims.
- Five session medians provide repeatability evidence for these machines and
  settings; they do not establish generality across processor samples or systems.

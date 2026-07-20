# Handoff for Claude: combined Intel, AMD, RTX 3090, and RTX 5060 Ti GEMM analysis

## Dataset and audit state

- Raw measurements: 8,100
- Platforms: Intel CPU, AMD CPU, RTX 3090, RTX 5060 Ti
- Problem sizes: [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
- Sessions: 5 per platform
- Technical repetitions: 10 per configuration/session
- Combined preflight verdict: PASS WITH WARNINGS
- Hard preflight failures: 0
- Warnings: 6

Each individual platform validator must pass before the combined pipeline runs.
The pipeline also verifies that each analysis matches the latest complete raw
campaign and checks GPU source provenance.

## Statistical contract

- Unit of analysis: five session medians, not 50 raw repetitions.
- Exact bootstrap CI per fixed configuration.
- Deterministic unpaired bootstrap CI for pairwise median ratios.
- Probability of superiority and Cliff's delta are included.
- Practical tolerance: 2%.
- Clear all-platform leaders require ratio-CI support against every alternative.
- Positive Cliff's delta means numerically larger A values; this favors A only
  for higher-is-better metrics.
- Native-best comparisons are descriptive post-selection analyses; no p-values
  are claimed because selection and estimation use the same five sessions.

## Energy semantics

- Intel/AMD: CPU package RAPL only.
- RTX 3090/5060 Ti: NVML board energy including device memory.
- GPU mode: resident; PCIe transfers excluded.

Do not translate device-domain comparisons into whole-system or architecture-only
claims without an explicit limitation.

## Tie-aware all-platform leaders

|     N | Energy leaders        | Runtime leaders   | EDP leaders   | Throughput leaders   | Efficiency leaders    |
|------:|:----------------------|:------------------|:--------------|:---------------------|:----------------------|
|    64 | Intel CPU             | Intel CPU         | Intel CPU     | Intel CPU            | Intel CPU             |
|   128 | RTX 5060 Ti           | RTX 3090          | RTX 5060 Ti   | RTX 3090             | RTX 5060 Ti           |
|   256 | RTX 5060 Ti           | RTX 3090          | RTX 3090      | RTX 3090             | RTX 5060 Ti           |
|   512 | RTX 3090, RTX 5060 Ti | RTX 3090          | RTX 3090      | RTX 3090             | RTX 3090, RTX 5060 Ti |
|  1024 | RTX 3090              | RTX 3090          | RTX 3090      | RTX 3090             | RTX 3090              |
|  2048 | RTX 3090              | RTX 3090          | RTX 3090      | RTX 3090             | RTX 3090              |
|  4096 | RTX 3090              | RTX 3090          | RTX 3090      | RTX 3090             | RTX 3090              |
|  8192 | RTX 3090              | RTX 3090          | RTX 3090      | RTX 3090             | RTX 3090              |
| 16384 | RTX 3090              | RTX 3090          | RTX 3090      | RTX 3090             | RTX 3090              |

## Best CPU versus best GPU

|     N | Energy best CPU/GPU   |   CPU/GPU energy ratio | Energy classification   | Runtime best CPU/GPU   |   CPU/GPU runtime ratio | Runtime classification   |   EDP ratio |
|------:|:----------------------|-----------------------:|:------------------------|:-----------------------|------------------------:|:-------------------------|------------:|
|    64 | INTEL/5060ti          |                 0.3687 | clear_CPU               | INTEL/3090             |                  0.4322 | clear_CPU                |     0.08317 |
|   128 | INTEL/5060ti          |                 2.921  | clear_GPU               | INTEL/3090             |                  2.436  | clear_GPU                |     3.827   |
|   256 | INTEL/5060ti          |                 4.416  | clear_GPU               | INTEL/3090             |                  5.494  | clear_GPU                |    15.16    |
|   512 | INTEL/3090            |                 5.253  | clear_GPU               | INTEL/3090             |                 11.16   | clear_GPU                |    60.32    |
|  1024 | INTEL/3090            |                 6.618  | clear_GPU               | INTEL/3090             |                 14.2    | clear_GPU                |    95.58    |
|  2048 | INTEL/3090            |                 8.421  | clear_GPU               | AMD/3090               |                 11.24   | clear_GPU                |    94.81    |
|  4096 | AMD/3090              |                 8.207  | clear_GPU               | AMD/3090               |                  9.614  | clear_GPU                |    84.39    |
|  8192 | AMD/3090              |                 8.894  | clear_GPU               | AMD/3090               |                 11.14   | clear_GPU                |   104.5     |
| 16384 | AMD/3090              |                 9.207  | clear_GPU               | AMD/3090               |                 10.94   | clear_GPU                |   105.8     |

## GPU generation comparison

|     N |   3090/5060Ti runtime |   3090/5060Ti energy |   3090/5060Ti GFLOP/J | Runtime result   | Energy result            |
|------:|----------------------:|---------------------:|----------------------:|:-----------------|:-------------------------|
|    64 |                0.5259 |               2.528  |                0.3956 | clear_3090       | clear_5060ti             |
|   128 |                0.5202 |               2.205  |                0.4535 | clear_3090       | clear_5060ti             |
|   256 |                0.4386 |               1.596  |                0.6266 | clear_3090       | clear_5060ti             |
|   512 |                0.326  |               0.9634 |                1.038  | clear_3090       | uncertain_3090_advantage |
|  1024 |                0.2696 |               0.7212 |                1.387  | clear_3090       | clear_3090               |
|  2048 |                0.2143 |               0.5313 |                1.882  | clear_3090       | clear_3090               |
|  4096 |                0.1973 |               0.4408 |                2.268  | clear_3090       | clear_3090               |
|  8192 |                0.1859 |               0.4218 |                2.371  | clear_3090       | clear_3090               |
| 16384 |                0.1854 |               0.3466 |                2.885  | clear_3090       | clear_3090               |

## Trade-off findings

- Clear device-level energy/runtime trade-off sizes: [128, 256]
- Uncertain device-level trade-off sizes: []
- Clear within-CPU configuration conflicts: 2

Inspect:

- `placement_by_size.csv`
- `within_platform_energy_runtime_tradeoffs.csv`
- `configuration_tradeoff_map.csv`
- `all_configuration_pareto.csv`

## Core output files

- `unified_session_medians.csv`: normalized n=5 session data.
- `unified_configuration_summary.csv`: harmonized medians, CIs and CVs.
- `native_policy_leaders.csv`: exact and tie-aware policy leaders.
- `pairwise_native_best_comparisons.csv`: all six device pairs and five metrics.
- `best_cpu_vs_best_gpu.csv`: direct placement view.
- `all_platform_metric_winners.csv`: per-size tie-aware device leaders.
- `placement_by_size.csv`: shared leader versus clear/uncertain device trade-off.
- `all_configuration_pareto.csv`: every CPU thread setting plus both GPUs.
- `configuration_tradeoff_map.csv`: dominant/compromise/dominated classes.
- `crossover_summary.csv`: changes in pairwise winner state over N.

## Questions to evaluate

1. Which size-dependent CPU/GPU crossover is robust under the tie-aware leader
   rule rather than only raw minima?
2. Does the strongest story come from device placement, CPU thread tuning, or
   the combined Pareto trade-off map?
3. Which results remain meaningful after retaining the CPU-package/GPU-board
   energy-domain limitation?
4. Is a transfer-aware `gpu_e2e` sensitivity analysis necessary for the intended
   venue, or can resident execution remain the primary compute-kernel scope?

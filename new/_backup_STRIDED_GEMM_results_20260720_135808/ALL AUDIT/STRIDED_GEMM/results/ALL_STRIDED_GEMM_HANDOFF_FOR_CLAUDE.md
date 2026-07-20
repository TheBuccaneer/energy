# Handoff for Claude: combined Intel, AMD, RTX 3090, and RTX 5060 Ti STRIDED_GEMM analysis

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
|   512 | RTX 5060 Ti, RTX 3090 | RTX 3090          | RTX 3090      | RTX 3090             | RTX 5060 Ti, RTX 3090 |
|  1024 | RTX 3090              | RTX 3090          | RTX 3090      | RTX 3090             | RTX 3090              |
|  2048 | RTX 3090              | RTX 3090          | RTX 3090      | RTX 3090             | RTX 3090              |
|  4096 | RTX 3090              | RTX 3090          | RTX 3090      | RTX 3090             | RTX 3090              |
|  8192 | RTX 3090              | RTX 3090          | RTX 3090      | RTX 3090             | RTX 3090              |
| 16384 | RTX 3090              | RTX 3090          | RTX 3090      | RTX 3090             | RTX 3090              |

## Best CPU versus best GPU

|     N | Energy best CPU/GPU   |   CPU/GPU energy ratio | Energy classification   | Runtime best CPU/GPU   |   CPU/GPU runtime ratio | Runtime classification   |   EDP ratio |
|------:|:----------------------|-----------------------:|:------------------------|:-----------------------|------------------------:|:-------------------------|------------:|
|    64 | INTEL/5060ti          |                 0.3985 | clear_CPU               | INTEL/3090             |                  0.4702 | clear_CPU                |     0.09964 |
|   128 | INTEL/5060ti          |                 2.785  | clear_GPU               | INTEL/3090             |                  2.273  | clear_GPU                |     3.503   |
|   256 | INTEL/5060ti          |                 4.419  | clear_GPU               | INTEL/3090             |                  5.392  | clear_GPU                |    13.83    |
|   512 | INTEL/5060ti          |                 5.44   | clear_GPU               | INTEL/3090             |                 11.06   | clear_GPU                |    59.94    |
|  1024 | INTEL/3090            |                 6.483  | clear_GPU               | AMD/3090               |                 13.15   | clear_GPU                |    97.36    |
|  2048 | AMD/3090              |                 7.762  | clear_GPU               | AMD/3090               |                 10.89   | clear_GPU                |    84.44    |
|  4096 | AMD/3090              |                 7.881  | clear_GPU               | AMD/3090               |                  9.769  | clear_GPU                |    82.64    |
|  8192 | AMD/3090              |                 8.497  | clear_GPU               | AMD/3090               |                 11      | clear_GPU                |    97.6     |
| 16384 | AMD/3090              |                 8.246  | clear_GPU               | AMD/3090               |                 10.43   | clear_GPU                |    90.8     |

## GPU generation comparison

|     N |   3090/5060Ti runtime |   3090/5060Ti energy |   3090/5060Ti GFLOP/J | Runtime result   | Energy result              |
|------:|----------------------:|---------------------:|----------------------:|:-----------------|:---------------------------|
|    64 |                0.536  |               2.829  |                0.3534 | clear_3090       | clear_5060ti               |
|   128 |                0.5269 |               2.377  |                0.4207 | clear_3090       | clear_5060ti               |
|   256 |                0.4448 |               1.733  |                0.5772 | clear_3090       | clear_5060ti               |
|   512 |                0.3382 |               1.026  |                0.9747 | clear_3090       | uncertain_5060ti_advantage |
|  1024 |                0.271  |               0.72   |                1.389  | clear_3090       | clear_3090                 |
|  2048 |                0.2179 |               0.5441 |                1.838  | clear_3090       | clear_3090                 |
|  4096 |                0.1975 |               0.4631 |                2.159  | clear_3090       | clear_3090                 |
|  8192 |                0.1869 |               0.4411 |                2.267  | clear_3090       | clear_3090                 |
| 16384 |                0.1856 |               0.3827 |                2.613  | clear_3090       | clear_3090                 |

## Trade-off findings

- Clear device-level energy/runtime trade-off sizes: [128, 256]
- Uncertain device-level trade-off sizes: []
- Clear within-CPU configuration conflicts: 1

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

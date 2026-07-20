# Handoff: audited Intel/AMD GEMM results

## Inputs audited

- AMD campaign `20260719_085402`: 5 sessions, 4,050 rows.
- Intel campaign `20260719_085511`: 5 sessions, 3,150 rows.
- Analysis output archives included validation reports, configuration summaries,
  session medians, robust-outlier tables, tie-aware leader tables, native-best
  comparison, common-thread comparison, EDP, and plots.

## Audit verdict

- AMD: PASS WITH WARNINGS; suitable as official data.
- Intel: PASS WITH WARNINGS; internally correct but systemically thermally
  constrained.

No hard check failed on either platform. Coverage, schema, checksum, formulas,
provenance, and multithread scaling all passed. The old catastrophic OpenBLAS
threading problem is absent.

## Warning interpretation

- Target-runtime share: AMD 81.09%, Intel 77.46%.
  Misses are concentrated at N=8192/16384, where a single GEMM already exceeds
  1.25 s and batches cannot be reduced below 1.
- AMD max temperature 96 °C, but session medians 77–78 °C and median clock drop
  only 0.29%.
- Intel max temperature 100 °C in every session; session medians 94–99 °C and
  median before/after clock drop 2.69%. Treat the Intel result as sustained,
  as-configured system performance, not thermally normalized architecture.

## Statistical semantics

- 10 repetitions within each configuration/session are not treated as 50
  independent experiments.
- Primary unit: five session medians.
- CIs: bootstrap over the five session medians.
- Primary cross-platform energy:
  `package_energy_per_op_j = device_energy_j / batches`.
- Raw `energy_per_op_j` is not cross-platform primary because Intel can include
  DRAM RAPL while AMD exposes no comparable separate DRAM domain.
- Intra-platform leaders are tolerance- and CI-aware.
- Cross-platform trade-off counts are descriptive with ±2% practical tolerance,
  not formal significance tests.

## Core findings

### Common-thread comparison

- Intel dominant: 53/63.
- AMD dominant: 4/63.
- Intel energy-efficient / AMD runtime-efficient: 6/63.
- AMD dominance only at 20 threads and N>=2048.
- Trade-off rows: 16 threads at N>=1024 plus 20 threads at N=1024.

### Native-best crossover

|     N | Energie                 | Laufzeit                | EDP                     |   Intel E-Threads |   AMD E-Threads |   Intel T-Threads |   AMD T-Threads |
|------:|:------------------------|:------------------------|:------------------------|------------------:|----------------:|------------------:|----------------:|
|    64 | Intel (72.5% niedriger) | Intel (63.2% niedriger) | Intel (89.9% niedriger) |                16 |               8 |                16 |               8 |
|   128 | Intel (24.8% niedriger) | Intel (13.2% niedriger) | Intel (31.5% niedriger) |                 4 |               4 |                10 |               4 |
|   256 | Intel (47.5% niedriger) | Intel (44.1% niedriger) | Intel (74.6% niedriger) |                 8 |               4 |                 8 |              16 |
|   512 | Intel (41.6% niedriger) | Intel (17.4% niedriger) | Intel (50.4% niedriger) |                 8 |              16 |                16 |              16 |
|  1024 | Intel (41.2% niedriger) | ≈ gleich                | Intel (46.1% niedriger) |                 8 |              16 |                 8 |              64 |
|  2048 | ≈ gleich                | AMD (39.8% niedriger)   | AMD (40.6% niedriger)   |                 8 |              64 |                 8 |              64 |
|  4096 | AMD (16.8% niedriger)   | AMD (47.4% niedriger)   | AMD (54.1% niedriger)   |                 8 |              64 |                10 |              32 |
|  8192 | AMD (8.0% niedriger)    | AMD (38.4% niedriger)   | AMD (42.6% niedriger)   |                 8 |              64 |                10 |              32 |
| 16384 | AMD (11.8% niedriger)   | AMD (44.3% niedriger)   | AMD (49.5% niedriger)   |                 8 |              64 |                10 |              32 |

Interpretation:

- Energy crossover: between N=2048 and N=4096.
- Runtime/EDP crossover: between N=1024 and N=2048.
- At N=2048, package energy is equivalent, but AMD is 39.8% faster.
- At N>=4096, AMD is both faster and lower-energy.
- Intel dominates small and medium sizes.

### Within-platform trade-offs

Intel, N=4096/8192/16384:
- 8 threads = energy optimum.
- 10 threads = runtime optimum.
- Runtime reduction: 7.25%, 12.92%, 7.63%.
- Energy penalty: 0.60%, 0.54%, 0.57%.

AMD, N=4096/8192/16384:
- 64 threads = energy optimum.
- 32 threads = runtime optimum.
- Runtime reduction: 6.66%, 4.74%, 4.80%.
- Energy penalty: 7.73%, 13.19%, 8.76%.

## Questions for interpretation

1. Is the strongest paper story the size-dependent native-best crossover, or the
   energy-vs-runtime compromise within each platform?
2. Should Intel be retained as an as-configured sustained system, or should a
   small cooling-controlled sensitivity rerun be added?
3. How should common-thread and native-best views be integrated without merging
   two different research questions?
4. Which claims should remain descriptive because cross-platform classifications
   currently use practical tolerance rather than inferential ratio CIs?

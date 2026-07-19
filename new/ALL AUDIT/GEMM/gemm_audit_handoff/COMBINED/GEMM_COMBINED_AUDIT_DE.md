# Audit of the Intel and AMD CPU GEMM campaigns

## Overall verdict

| Platform | Verdict | Interpretation |
|---|---|---|
| AMD | **PASS WITH WARNINGS** | Officially usable; warnings are limited to unavoidable long windows at large N and one high temperature peak. |
| Intel | **PASS WITH WARNINGS** | Correct and usable as an as-configured sustained-system measurement, but the persistent 94–100 °C thermal state is a material limitation. |

`PASS WITH WARNINGS` means that **no hard validity criterion failed**. The
campaigns are complete, internally consistent, formula-correct, checksum-correct,
and free of the known catastrophic OpenBLAS threading bug. A warning marks a
condition that affects interpretation but does not automatically invalidate the
measurements.

## Audit checks that passed

- 5 sessions per platform;
- 10 repetitions per configuration and session;
- 9 GEMM sizes;
- correct Intel thread grid (1–20) and AMD grid (1–64);
- exact expected row counts;
- no duplicates or missing configuration/repetition groups;
- `cpu-gpu-v2` schema;
- `workload=GEMM`, `implementation=openblas_sgemm`,
  `execution_mode=cpu_native`;
- all checksums true;
- finite positive measurements;
- all serialized formulas reproduced;
- session/file provenance consistent;
- no catastrophic multithread regression;
- useful scaling at all large sizes.

## Why the target-runtime warning is not a calibration failure

The target interval is 0.75–1.25 s. For the largest matrices, one single GEMM
already takes longer than 1.25 s. Since `batches` cannot be lower than one, these
measurements necessarily remain above target. The warning is therefore a
heterogeneous-window caveat. It does not invalidate energy-per-GEMM or
time-per-GEMM.

## Thermal assessment

| Platform | Max | Session-median range | Median clock drop | Unstable runtime configs | Unstable energy configs |
|---|---:|---:|---:|---:|---:|
| AMD | 96 °C | 77–78 °C | 0.29% | 3/81 | 0/81 |
| Intel | 100 °C | 94–99 °C | 2.69% | 9/63 | 9/63 |

The AMD warning is minor because the high value is a peak and session medians
remain moderate. The Intel warning is systematic and must be treated as a threat
to validity. The cross-platform result should be described as a comparison of
the two **measured systems under their sustained operating conditions**, not as
a thermally normalized microarchitectural comparison.

## Cross-platform scientific findings

### Common-thread view

At matched thread counts:

- Intel dominant: 53/63 configurations;
- AMD dominant: 4/63 configurations;
- Intel lower-energy / AMD lower-runtime trade-off: 6/63 configurations.

AMD dominance appears only at 20 threads for `N>=2048`. The six trade-off cases
occur at 16 threads for `N>=1024` and at 20 threads for `N=1024`.

This view shows stronger Intel per-thread/package efficiency over most of the
matched software-parallelism grid.

### Native-best view

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

The primary crossover pattern is:

- `N<=1024`: Intel generally minimizes both energy and runtime; runtime at
  `N=1024` is practically equivalent.
- `N=2048`: package energy is practically equal, while AMD is about 39.8%
  faster and has about 40.6% lower EDP.
- `N>=4096`: AMD minimizes both package energy and runtime.
- The energy crossover occurs between `N=2048` and `N=4096`.
- The runtime and EDP crossover occurs between `N=1024` and `N=2048`.

## Important interpretation limits

1. Primary cross-platform energy is package-only
   `device_energy_j / batches`. Raw total energy is not used because Intel may
   include a separate DRAM domain while AMD does not.
2. CPU package domains are vendor-specific and do not represent whole-system
   energy.
3. Five session medians measure repeatability on these systems; they do not
   establish population-level generality.
4. Cross-platform dominance counts are descriptive and use a ±2% practical
   tolerance. They should not be presented as formal hypothesis-test results.
5. Exact thread minima must be accompanied by the reported leader sets and
   `clear_leader`/`tie_or_uncertain` status.
6. The Intel thermal ceiling may bias large-N performance downward. The observed
   crossover is valid for the measured systems, but it should not be framed as
   architecture-only causality.

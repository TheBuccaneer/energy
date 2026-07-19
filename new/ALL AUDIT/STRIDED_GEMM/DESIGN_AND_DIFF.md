# Design and diff summary

## Technical reference

The repaired normal GEMM files are the sole reference for:

- option parsing and output-file creation;
- `benchmark_common.hpp` integration;
- RAPL acquisition and delta calculation;
- clock and temperature snapshots;
- calibration to the common target runtime;
- deterministic configuration shuffling;
- `BENCH_SIZE_FILTER` and `BENCH_THREAD_FILTER`;
- OpenBLAS backend detection;
- `openblas_set_num_threads()` plus requested/active verification;
- FLOP, logical-byte, checksum, CSV-row, and console-output handling.

No RAPL, CSV, PMU, or scheduling implementation was copied from the old batched Strided GEMM programs.

## Changes made only for `ld=2N`

Compared with repaired contiguous GEMM, the C++ files change only the intended Strided GEMM properties:

1. workload: `GEMM` -> `STRIDED_GEMM`;
2. implementation: `openblas_sgemm` -> `openblas_sgemm_ld2n`;
3. leading dimension: `ld = 2 * n`;
4. allocation: `N*N` -> `N*ld`, using `size_t`;
5. initialization: logical columns use the same deterministic values as GEMM, padding is deterministic zero;
6. SGEMM `lda`, `ldb`, and `ldc` use `ld`;
7. checksum result indexing uses `row*ld+col`;
8. `problem_spec` records `N=<N>;ld=<2N>`;
9. default output names and run directories use `STRIDED_GEMM`.

FLOPs remain `2*N^3` per operation. Logical bytes remain `3*N^2*sizeof(float)` and deliberately exclude padding.

## Threading fix

Both platforms now use only OpenBLAS thread control. The AMD source no longer contains:

- `<omp.h>`;
- OpenMP pragmas;
- `omp_set_dynamic()`;
- `omp_set_num_threads()`.

Both runners and quickchecks compile without `-fopenmp`. They remove inherited `OMP_PROC_BIND`, `OMP_PLACES`, and `GOMP_CPU_AFFINITY` variables before execution. There is no OpenMP wrapper around SGEMM.

## Campaign behavior

There is no session pause. `sleep 60` appears only in the background `sudo` credential keepalive.

Expected official output:

- Intel: 630 rows/session, 3150 rows/5 sessions;
- AMD: 810 rows/session, 4050 rows/5 sessions.

The restore script runs via the `EXIT` trap after success, error, signal, or `Ctrl+C`. Automatic power-off occurs only after five completely validated sessions and only when `POWER_OFF_AT_END=1`.

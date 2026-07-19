# AMD package audit

## Corrected during final audit

- Replaced the stale 42-column `cpu-gpu-v1` documentation with the actual 45-column `cpu-gpu-v2` schema.
- Corrected Reduction documentation to the implemented true one-array sum.
- Replaced all manual default CSV names ending in `_intel.csv` with `_amd.csv`.
- Added visible `preparing` and `calibrated` messages for every configuration.
- Added an OpenMP-runtime guard for oneDNN so Conv2D cannot silently ignore the requested per-configuration thread count.
- Added an exact C++ AMD package-energy probe using the same `perf_event_open` backend as the workloads.
- Added automatic rollback if AMD preparation fails partway through.
- Added a five-minute cooldown after compilation before the first measured workload.
- Added a Python CSV audit after every workload file: exact header, row count, configuration count, repetitions, positive package energy, `total=package`, `dram=-1`, runtime status, and checksum.
- Added overwrite protection, campaign console log, and system/library metadata.

## Verified statically in the build environment

- Archive structure contains all six workloads under `CPU/AMD` and all output directories under `runs2`.
- Shell syntax passes for all scripts.
- GEMM, Strided GEMM, STREAM, AXPY, and Reduction C++ syntax passes.
- Conv2D C++ syntax passes with oneDNN 3.x headers; it links against a oneDNN OpenMP runtime in the audit environment.
- The thread grid is exactly `1,2,4,8,10,16,20,32,64`.
- Default row arithmetic is 22,950 rows.

## Still requires the target machine

The final guarantee must come from `scripts/check_CPU_AMD.sh` on the Threadripper,
because only that machine can prove its installed OpenBLAS/oneDNN development
packages and the exact non-root AMD perf event. The earlier manual `perf stat`
test already showed a valid package-energy signal, but the supplied check now
verifies the exact C++ backend too.

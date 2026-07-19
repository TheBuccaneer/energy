# RTX 3090 GEMM Results Summary

We measured resident FP32 GEMM on an NVIDIA GeForce RTX 3090 across nine square
matrix sizes. Each size was evaluated ten times in five independently randomized
sessions, yielding 450 valid measurements. The implementation used
`cublasGemmEx` with `CUBLAS_COMPUTE_32F_PEDANTIC`, while TF32 was disabled.
Allocations, initialization, and PCIe transfers were excluded from the measured
interval. Energy was obtained from the direct NVML board-energy counter.

All hard validation checks passed. The campaign was complete, all checksums were
correct, all reported formulas were reproduced, and no size was classified as
unstable in runtime, energy, or throughput. Session-level repeatability was
strong: the span between the lowest and highest global session-median
throughput was approximately 0.22%. Maximum temperature was 67 °C, the median
within-measurement clock change was zero, and no serious throttle reason was
observed. The recurring `software_power_cap` flag describes operation at the
configured power limit rather than thermal throttling.

Throughput increased from approximately 0.10 TFLOP/s at N=64 to 17.75 TFLOP/s
at N=1024, 24.76 TFLOP/s at N=4096, and 26.34 TFLOP/s at N=8192. It reached
26.46 TFLOP/s at N=16384; the gain from N=8192 to N=16384 was only about 0.43%,
indicating a performance plateau near 26.4 TFLOP/s for the tested pedantic-FP32
configuration.

Board-energy efficiency improved strongly with matrix size. Median efficiency
increased from 0.62 GFLOP/J at N=64 to 53.80 GFLOP/J at N=1024, 79.95 GFLOP/J
at N=4096, and 85.01 GFLOP/J at N=16384. The result illustrates the high fixed
overhead of small GPU jobs and the substantially better utilization achieved by
large GEMMs.

The nominal 0.75–1.25 s target interval was met by 88.89% of the measurements.
All deviations occurred at N=16384, where four GEMMs produced a measurement
window of about 1.33 s. This slight overcalibration does not invalidate
per-operation metrics, which remained stable across sessions. The reported
14.44% robust-outlier share must also not be interpreted as an invalid-run
fraction: run-level runtime variation remained below 0.44% and run-level energy
variation below 3.21% for every size, while no rows were removed.

These measurements characterize resident execution and NVML board energy.
They do not include host-device transfers and are not in the same energy domain
as CPU package-only RAPL. Both qualifications must be retained in subsequent
CPU–GPU comparisons.

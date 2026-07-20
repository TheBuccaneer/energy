# RTX 5060 Ti GEMM Results Summary

We measured resident FP32 GEMM on an NVIDIA GeForce RTX 5060 Ti across nine
square matrix sizes. Each size was evaluated ten times in five independently
randomized sessions, yielding 450 valid measurements. The implementation used
`cublasGemmEx` with `CUBLAS_COMPUTE_32F_PEDANTIC`, while TF32 was disabled.
Allocations, initialization, and PCIe transfers were excluded from the measured
interval. Energy was obtained from the direct NVML board-energy counter.

All hard validation checks passed. The campaign was complete, all checksums were
correct, all reported formulas were reproduced, and no size was classified as
unstable in runtime, energy, or throughput. Maximum temperature was 75 °C,
session-median temperatures remained between 66 and 68 °C, and no throttle
reason was observed in any of the 450 measurements.

The nominal 0.75–1.25 s target interval was met by 88.89% of the measurements.
All deviations occurred at N=16384, where a single GEMM already required about
1.793 s. Because batches could not be reduced below one, the actionable
target-window share was 100%; this is an unavoidable minimum-batch condition
rather than a calibration failure.

Throughput reached 4.786 TFLOP/s at N=1024 and then formed a plateau near
4.9 TFLOP/s: 4.894 TFLOP/s at N=2048, 4.885 TFLOP/s at N=4096,
4.898 TFLOP/s at N=8192, and 4.905 TFLOP/s at N=16384. The gain from N=8192
to N=16384 was only 0.14%.

Board-energy efficiency peaked at N=1024 with 38.80 GFLOP/J. It declined to
36.59 GFLOP/J at N=2048, approximately 35.3 GFLOP/J at N=4096 and N=8192,
and 29.42 GFLOP/J at N=16384. Between N=8192 and N=16384, median board power
increased by 20.10% while throughput remained almost
constant, reducing efficiency by 16.61%. Thus, the
largest resident GEMM did not provide the highest board-energy efficiency on
this device.

The robust diagnostic marked 108 of 450 rows. These flags must not be treated as
invalid runs: maximum run-level runtime CV was 1.57%,
maximum run-level energy CV was 4.50%, and no
rows were removed. The flags were distributed across all sessions and sizes,
indicating an overly sensitive detector under narrow distributions rather than
a failed campaign.

These measurements characterize resident execution and NVML board energy. They
exclude host-device transfers and do not use the same energy domain as CPU
package-only RAPL. Both qualifications must be retained in subsequent
CPU–GPU comparisons.

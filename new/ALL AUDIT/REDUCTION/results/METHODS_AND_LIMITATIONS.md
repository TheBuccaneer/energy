# REDUCTION methods and limitations

## Workload

`sum(x[0:N]) -> FP32 scalar` with FP32 input, accumulation and output.

- FLOPs per operation: `N-1`
- Logical bytes per operation: `4*N+4`
- Logical operational intensity: `(N-1)/(4*N+4)`, approaching 0.25 FLOP/byte

CPU uses a fixed 4096-element blocked OpenMP reduction with SIMD-local accumulation and a fixed final stage. GPUs use CUB `DeviceReduce::Sum`. The mathematical workload is identical; the internal reduction hierarchy is platform-native.

## Statistics

Ten repetitions are technical repetitions. Their median forms one session observation. Five session medians form the analysis sample. Exact enumeration-based n=5 bootstrap median intervals are used for configuration summaries. Native-best pairwise and exact-winner-regret intervals are descriptive because selection and reporting use the same sessions.

## Energy

CPU primary energy is package RAPL. GPU primary energy is NVML board energy including device memory. Intel package+DRAM remains available as a within-platform sensitivity, but cross-platform primary comparisons use `device_energy_j` for both CPUs and GPUs.

## Scope

GPU mode is resident. Allocations, initialization and PCIe transfers are outside the interval. Logical useful-data rate is derived from semantic bytes and runtime, not a traffic counter. Small sizes may be cache- and synchronization-dominated; large sizes test resident memory reading plus hierarchical aggregation. CPU partial-array and CUB-workspace traffic are not counted in logical bytes.

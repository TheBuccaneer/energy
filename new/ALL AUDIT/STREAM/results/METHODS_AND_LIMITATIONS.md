# STREAM methods and limitations

## Workload

`a[i] = b[i] + 3.0f*c[i]`, FP32. One operation is one complete pass over N elements.

- FLOPs per operation: `2*N`
- Logical bytes per operation: `12*N`
- Logical operational intensity: `1/6 FLOP/byte`

## Statistics

Ten repetitions are technical repetitions. Their median forms one session observation. Five session medians form the analysis sample. Exact n=5 bootstrap median intervals are used for configuration summaries. Native-best pairwise intervals are descriptive because selection and reporting use the same sessions.

## Energy

CPU primary energy is package RAPL. GPU primary energy is NVML board energy including device memory. Intel total package+DRAM remains available as a within-platform sensitivity, but cross-platform primary comparisons use package energy for both CPUs.

## Scope

GPU mode is resident. Allocations, initialization and PCIe transfers are outside the interval. Logical bandwidth is derived from semantic bytes and runtime, not a hardware counter. Small sizes may use cache; large sizes are the principal memory-stress regime.

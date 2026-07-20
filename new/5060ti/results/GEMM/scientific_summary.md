# RTX 5060 Ti GEMM scientific analysis

- Campaign: `20260719_172746`
- Measurements: 450
- Sessions: 5
- Mode: `gpu_resident`
- Implementation: `cublas_gemm_ex_fp32_pedantic`

## Quality and repeatability

- Overall target-window share: 88.89%
- Actionable target-window share: 100.00%
- Robust outlier share: 24.00%
- Maximum temperature: 75.0 °C
- Median before/after SM-clock decline: 0.00%
- Serious throttle rows: 0
- Runtime-unstable sizes: none
- Energy-unstable sizes: none
- Throughput-unstable sizes: none

## Main findings

- Peak median throughput: 4905.17 GFLOP/s at N=16384.
- Lowest median EDP: 3.30506e-09 J·s at N=64.

## Interpretation

This analysis characterizes resident FP32 GEMM. PCIe transfers are outside the measured interval. NVML device energy is board-level and includes device-memory energy, so it is not the same measurement domain as CPU package-only RAPL. The five session medians are the primary repeatability units; the ten within-session repetitions are technical repetitions.

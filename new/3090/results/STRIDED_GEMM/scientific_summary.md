# RTX 3090 STRIDED_GEMM scientific analysis

- Campaign: `20260720_120846`
- Measurements: 450
- Sessions: 5
- Mode: `gpu_resident`
- Implementation: `cublas_gemm_ex_fp32_pedantic_ld2n`
- Layout: logical N×N matrices, physical leading dimension `ld=2N`

## Quality and repeatability

- Overall target-window share: 91.11%
- Actionable target-window share: 91.11%
- Robust outlier share: 13.11%
- Maximum temperature: 66.0 °C
- Median before/after SM-clock decline: 0.00%
- Serious throttle rows: 0
- Runtime-unstable sizes: none
- Energy-unstable sizes: none
- Throughput-unstable sizes: none

## Main findings

- Peak median throughput: 26441.90 GFLOP/s at N=16384.
- Peak board efficiency: 80.29 GFLOP/J at N=8192.

## Interpretation contract

Runtime and throughput are inverse views of the same fixed-work axis. Board energy and GFLOP/J are inverse views of the same fixed-work axis. They are not counted as independent votes. EDP is a composite of runtime and energy. The 12N² logical-byte value is a semantic anchor, while the allocated footprint is 24N²; neither is measured physical memory traffic. PCIe transfers are outside the measured interval. NVML energy is board-level.

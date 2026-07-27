# Independent audit prompt — REDUCTION analysis

Audit the REDUCTION measurement and analysis pipeline. Do not trust generated prose; recompute from CSV outputs.

## Frozen semantics

```text
operation        = sum(x[0:N]) -> FP32 scalar
FLOPs/op         = N-1
logical bytes/op = 4*N+4
CPU              = openmp_blocked_sum_fp32
GPU              = cub_device_reduce_sum_fp32
gpu mode         = gpu_resident
```

## Required checks

1. Validate exact 45-column `cpu-gpu-v2` headers and complete 5-session campaigns.
2. Recompute `(N-1)*batches`, `4*N+4`, time/op, energy/op, energy/FLOP, GFLOP/s and power.
3. Confirm CPU timing equality and GPU event/host cross-clock tolerance.
4. Confirm five session medians are the statistical units; reject n=50 pseudoreplication.
5. Recompute exact n=5 bootstrap median intervals, CVs, leader sets, pairwise ratios, Cliff's delta and Pareto flags.
6. Recompute the 36 exact-winner regret rows and preserve the post-selection qualification.
7. Confirm runtime/logical-rate and energy/logical-GB/J are inverse views, not independent evidence.
8. Keep CPU-package/GPU-board energy-domain asymmetry explicit.
9. Keep CPU partials and CUB workspace outside the logical-byte count.
10. Keep resident GPU scope explicit; PCIe transfers are excluded.

Return `PASS`, `PASS WITH REPORTING PATCHES`, `PASS WITH ANALYSIS PATCHES`, or `RERUN REQUIRED`, with exact mismatches and reproducible calculations.

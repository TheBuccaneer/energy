# REDUCTION analysis package audit

## Design basis

This package is derived from the final STREAM analysis architecture but changes every workload-dependent contract:

- workload identity: `REDUCTION`;
- campaign prefix: `reduction_`;
- operation: FP32 one-array sum;
- FLOPs: `N-1`;
- logical bytes: `4*N+4`;
- CPU implementation: `openmp_blocked_sum_fp32`;
- GPU implementation: `cub_device_reduce_sum_fp32`;
- source and runner paths: REDUCTION-specific;
- reports: aggregation/synchronization framing;
- exact-winner regret: included from the final STREAM reporting audit.

## Validation gates

- exact schema and identity;
- complete coverage: five sessions, ten repetitions, nine sizes and full thread grids;
- formula identities including energy/second and energy/FLOP;
- CPU/GPU energy semantics;
- GPU cross-clock timing threshold;
- checksum completeness;
- source/runner provenance;
- deterministic aggregate dimensions and integrity checks.

No measurements are altered or silently removed. Robust outliers are diagnostic only.

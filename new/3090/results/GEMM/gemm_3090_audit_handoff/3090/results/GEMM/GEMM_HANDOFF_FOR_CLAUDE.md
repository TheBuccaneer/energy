# Handoff for Claude: audited RTX 3090 GEMM campaign

## Dataset

- Campaign: `20260719_152731`
- Device: NVIDIA GeForce RTX 3090
- Rows: 450
- Sessions: 5
- Repetitions: 10 per size/session
- Sizes: 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384
- Mode: `gpu_resident`
- Implementation: `cublas_gemm_ex_fp32_pedantic`
- Compute: `CUBLAS_COMPUTE_32F_PEDANTIC`
- TF32: disabled
- Energy: direct NVML board-energy delta

## Audit verdict

**PASS WITH WARNINGS — official data are usable.**

No hard validation check failed:

- complete coverage;
- correct row counts;
- no duplicates;
- all checksums true;
- all formulas reproduced;
- source provenance confirms pedantic FP32, NVML energy, resident mode and no
  session pause;
- no unstable size in runtime, energy or throughput;
- no serious throttle rows.

## Warnings and their interpretation

1. GPU column order places `device_energy_j` before `total_energy_j`, unlike the
   CPU-v2 order. Values and names are correct; no rerun is needed.
2. Target-window share is 88.89%. All 50 misses are N=16384:
   batches=4 and about 0.3325 s/GEMM produce a 1.33 s interval. This is slight,
   avoidable overcalibration, but per-GEMM results are stable.
3. 65/450 rows are robust-outlier flags. Do not treat this as invalid data.
   Run-level runtime CV is at most 0.44%, run-level energy CV at most 3.21%,
   session-level runtime CV at most 0.48%, and session-level energy CV at most
   0.89%. The detector is likely sensitive under tight distributions/NVML
   quantization. No rows were removed.

## Thermal and throttle state

- Maximum temperature: 67 °C
- Session-median temperatures: 62–65 °C
- Median within-window SM-clock decline: 0.00%
- Serious throttle rows: 0
- `0x4 = software_power_cap`: 300 rows

The power-cap flag is not thermal throttling; it represents operation at the
configured software power limit and should be documented as part of the setup.

## Key results

|     N |   Runtime/GEMM (ms) |   Energy/GEMM (J) |   Throughput (TFLOP/s) |   Board power (W) |   GFLOP/J |   Temp. (°C) |
|------:|--------------------:|------------------:|-----------------------:|------------------:|----------:|-------------:|
|    64 |          0.00521703 |       0.000841273 |                  0.1   |             161.3 |      0.62 |         57   |
|   128 |          0.0051611  |       0.000922996 |                  0.813 |             178.8 |      4.54 |         58   |
|   256 |          0.00718051 |       0.0019916   |                  4.673 |             277.2 |     16.86 |         62   |
|   512 |          0.0226917  |       0.00721341  |                 11.83  |             317.9 |     37.22 |         65.5 |
|  1024 |          0.120991   |       0.0399125   |                 17.749 |             329.9 |     53.8  |         64   |
|  2048 |          0.752203   |       0.250474    |                 22.84  |             332.7 |     68.64 |         64   |
|  4096 |          5.55136    |       1.71844     |                 24.758 |             309.7 |     79.95 |         64   |
|  8192 |         41.739      |      13.1432      |                 26.343 |             315.2 |     83.58 |         64   |
| 16384 |        332.494      |     103.624       |                 26.455 |             311.2 |     85.01 |         63   |

Core interpretation:

- Throughput rises to 24.76 TFLOP/s at N=4096 and 26.34 TFLOP/s at N=8192.
- N=16384 reaches 26.46 TFLOP/s.
- Throughput gain from N=8192 to N=16384 is only ~0.43%, indicating saturation
  around 26.4 TFLOP/s.
- Board efficiency rises from 0.62 GFLOP/J at N=64 to 85.01 GFLOP/J at N=16384.
- Small jobs are dominated by fixed GPU overheads; large GEMMs utilize the
  device efficiently.

## Important correction to generated prose

“Lowest EDP at N=64” is mathematically true but scientifically trivial because
the compared jobs contain different amounts of work. Do not use absolute EDP
across matrix sizes as a placement conclusion. Use throughput, energy per FLOP,
GFLOP/J, and same-size cross-device ratios.

## Scope for later comparison

- These are resident measurements; PCIe transfers are excluded.
- NVML measures board energy, including device-memory energy.
- CPU RAPL package and GPU NVML board energy are not identical domains.
- A later CPU–GPU comparison can use these values descriptively, but must state
  the domain asymmetry.
- A separate `gpu_e2e` sensitivity analysis is needed for transfer-aware
  placement claims.

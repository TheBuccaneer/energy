# Handoff for Claude: audited RTX 5060 Ti GEMM campaign

## Dataset

- Campaign: `20260719_172746`
- Device: NVIDIA GeForce RTX 5060 Ti
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

- complete coverage and correct row counts;
- no duplicate repetitions;
- all checksums true;
- all formulas reproduced;
- source provenance confirms pedantic FP32, direct NVML energy, resident mode,
  five sessions, ten repetitions and no session pause;
- no unstable size in runtime, energy or throughput;
- no throttle rows.

## Warnings

1. GPU column order places `device_energy_j` before `total_energy_j`, unlike the
   CPU-v2 canonical order. Names and values are correct; no rerun is needed.
2. Overall target-window share is 88.89%. All 50 misses are N=16384 with
   `batches=1` and about 1.793 s/GEMM. Actionable target-window share is 100%;
   this is an unavoidable minimum-batch condition.

## Stability and thermals

- Maximum temperature: 75 °C
- Session-median temperature: 66–68 °C
- Serious throttle rows: 0
- Throttle mask: 0x0 for all 450 rows
- Board-power range: 32.4–169.6 W
- Session-median throughput span: 0.46%
- Maximum session runtime CV: 1.35%
- Maximum session energy CV: 3.33%

## Robust-outlier diagnostic

- Flagged rows: 108/450 = 24.00%
- No rows removed
- Maximum run runtime CV: 1.57%
- Maximum run energy CV: 4.50%
- Flags occur across all sessions and sizes

Interpret the 24% as an overly sensitive MAD diagnostic under narrow
distributions, not as a failed-run rate. Do not filter these rows automatically.

## Key results

|     N |   Runtime/GEMM (ms) |   Energy/GEMM (J) |   Throughput (TFLOP/s) |   Board power (W) |   GFLOP/J |   Temp. (°C) |
|------:|--------------------:|------------------:|-----------------------:|------------------:|----------:|-------------:|
|    64 |          0.0099198  |       0.000332818 |                  0.053 |              33.6 |      1.57 |         48.5 |
|   128 |          0.00992057 |       0.000418624 |                  0.423 |              41.7 |     10.13 |         50   |
|   256 |          0.0163733  |       0.00124799  |                  2.049 |              76.2 |     26.9  |         57   |
|   512 |          0.0695993  |       0.00748716  |                  3.857 |             107.6 |     35.85 |         66   |
|  1024 |          0.448735   |       0.0553448   |                  4.786 |             123.3 |     38.8  |         67   |
|  2048 |          3.51052    |       0.471474    |                  4.894 |             133.7 |     36.59 |         68   |
|  4096 |         28.1354     |       3.8981      |                  4.885 |             138.6 |     35.26 |         70   |
|  8192 |        224.475      |      31.1611      |                  4.898 |             138.8 |     35.28 |         70   |
| 16384 |       1793.22       |     298.952       |                  4.905 |             166.7 |     29.42 |         72.5 |

## Core interpretation

1. Throughput saturates early:
   - 4.786 TFLOP/s at N=1024
   - approximately 4.9 TFLOP/s from N=2048 onward
   - only 0.14% gain from N=8192 to N=16384

2. Board efficiency peaks at N=1024:
   - 38.80 GFLOP/J at N=1024
   - approximately 35.3 GFLOP/J at N=4096/8192
   - 29.42 GFLOP/J at N=16384

3. From N=8192 to N=16384:
   - board power increases by 20.10%
   - throughput remains nearly constant
   - GFLOP/J declines by 16.61%

4. Therefore, the largest GEMM does not maximize board-energy efficiency on the
   RTX 5060 Ti. The device reaches its best measured GFLOP/J at N=1024.

## Important interpretation limits

- “Lowest EDP at N=64” is trivial across differently sized jobs and should not be
  used as a placement conclusion.
- These are resident measurements; PCIe transfers are excluded.
- NVML is board-level energy, unlike CPU package-only RAPL.
- The audit validates the chosen pedantic FP32 path, not architectural peak
  performance.
- A direct RTX 3090–RTX 5060 Ti comparison should use same-size ratios and
  identical source/compute semantics.

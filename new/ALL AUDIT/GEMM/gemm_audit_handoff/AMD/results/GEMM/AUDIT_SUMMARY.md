# AMD GEMM Audit

## Verdict

**PASS WITH WARNINGS — scientifically usable.**

All hard validity checks passed:

- complete 5-session campaign;
- 4,050 rows, exactly 810 rows per session;
- complete size/thread/repetition coverage;
- no duplicate configurations;
- no checksum failures;
- all time, energy, FLOP, throughput and power formulas reproduced;
- no catastrophic multithread slowdown;
- useful parallel speedup for every large GEMM.

## Meaning of the warnings

### Target-runtime share: 81.09%

The target was 0.75–1.25 s per measurement. The misses are concentrated at the
largest matrices. With `batches=1`, one GEMM already exceeds 1.25 s, so adaptive
batching cannot shorten the window further. This is a minimum-batch limitation,
not a calibration or correctness failure.

### Maximum temperature: 96 °C

This is a peak value, not the typical campaign state. Session-median temperatures
were 77–78 °C, and the median before/after clock decline was only about 0.29%.
There were 3/81 runtime-unstable configurations and 0/81 energy-unstable
configurations.

## Scientific quality

- Robust outlier share: 4.99%.
- Between-session energy stability is strong across all 81 configurations.
- Thread scaling is plausible and the former OpenBLAS threading failure is absent.
- The campaign is suitable as the official AMD GEMM dataset.

## Main AMD result

For large GEMMs, 64 threads minimize package energy, while 32 threads usually
minimize runtime. This produces a real energy–runtime trade-off:

|     N |   energy-opt threads |   runtime-opt threads | energy penalty of runtime-opt   | runtime gain vs energy-opt   |
|------:|---------------------:|----------------------:|:--------------------------------|:-----------------------------|
|  4096 |                   64 |                    32 | 7.73%                           | 6.66%                        |
|  8192 |                   64 |                    32 | 13.19%                          | 4.74%                        |
| 16384 |                   64 |                    32 | 8.76%                           | 4.80%                        |

The effect is scientifically useful: the fastest AMD setting is not always the
most energy-efficient setting.

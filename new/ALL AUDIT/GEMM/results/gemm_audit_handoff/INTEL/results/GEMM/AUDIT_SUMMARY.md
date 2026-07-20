# Intel GEMM Audit

## Verdict

**PASS WITH WARNINGS — internally valid, but thermally constrained.**

All hard validity checks passed:

- complete 5-session campaign;
- 3,150 rows, exactly 630 rows per session;
- complete size/thread/repetition coverage;
- no duplicate configurations;
- no checksum failures;
- all time, energy, FLOP, throughput and power formulas reproduced;
- no catastrophic multithread slowdown;
- useful parallel speedup for every large GEMM.

## Meaning of the warnings

### Target-runtime share: 77.46%

The target was 0.75–1.25 s per measurement. The misses are concentrated at
`N=8192` and especially `N=16384`. With `batches=1`, one GEMM already exceeds the
upper bound; adaptive batching cannot reduce the measurement below one operation.
This warning does not imply wrong per-operation metrics.

### Maximum temperature: 100 °C

This warning is material. Every session reached 100 °C, and session-median
temperatures were 94–99 °C. The median before/after clock decline was about 2.69%.
The dataset contains 9/63 runtime-unstable and 9/63 energy-unstable
configurations.

The measurements therefore characterize the **sustained, thermally constrained
Intel system as configured**, not an unconstrained architectural maximum. They
remain usable for machine-level placement claims, but processor-level claims
must not attribute every Intel–AMD difference solely to CPU architecture.

## Scientific quality

- Robust outlier share: 7.05%.
- Formula, checksum, coverage and threading checks all pass.
- The data must not be discarded as incorrect.
- The thermal state must be disclosed prominently.
- A small cooling-controlled sensitivity rerun at representative large sizes
  would strengthen a publication, but it is not required to interpret the
  current campaign as an as-configured system measurement.

## Main Intel result

At large sizes, 8 threads minimize package energy, while 10 threads minimize
runtime with less than 1% additional package energy:

|     N |   energy-opt threads |   runtime-opt threads | energy penalty of runtime-opt   | runtime gain vs energy-opt   |
|------:|---------------------:|----------------------:|:--------------------------------|:-----------------------------|
|  4096 |                    8 |                    10 | 0.60%                           | 7.25%                        |
|  8192 |                    8 |                    10 | 0.54%                           | 12.92%                       |
| 16384 |                    8 |                    10 | 0.57%                           | 7.63%                        |

This is a strong runtime-efficient compromise: 10 threads save roughly 7–13%
runtime relative to the energy optimum while adding only about 0.5–0.6% energy.

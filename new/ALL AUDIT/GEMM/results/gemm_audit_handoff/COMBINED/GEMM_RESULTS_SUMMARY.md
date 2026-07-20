# GEMM Results Summary

We evaluated CPU-resident FP32 SGEMM on an Intel platform and an AMD Threadripper
platform across nine matrix sizes. Each configuration was measured ten times in
five independently randomized sessions. We use the median of each session as the
primary statistical unit and package energy per GEMM
(`device_energy_j / batches`) as the cross-platform energy metric.

Both campaigns passed all hard correctness, coverage, formula, provenance, and
thread-scaling checks. The AMD campaign contained 4,050 valid measurements and
the Intel campaign 3,150. The known catastrophic OpenBLAS multithreading failure
was absent. The fraction of measurements inside the nominal 0.75–1.25 s window
was 81.09% on AMD and 77.46% on Intel. The deviations were concentrated at the
largest matrices, where a single GEMM already exceeded the target interval and
`batches` could not be reduced below one.

At matched thread counts, Intel was dominant in 53 of 63 size–thread
configurations, AMD was dominant in four, and six configurations exhibited a
trade-off in which Intel consumed less package energy while AMD completed
faster. AMD dominance at matched parallelism occurred only at 20 threads for
`N>=2048`. Thus, Intel provided stronger performance and package-energy behavior
over most of the common-thread grid, whereas AMD benefited more strongly from
its larger native thread budget.

The native-best comparison showed a clear size-dependent crossover. Intel
minimized package energy through `N=1024`; at `N=2048`, the best package-energy
values were practically equal. AMD became the package-energy winner at
`N>=4096`, reducing energy by approximately 8.0–16.8% relative to Intel's best
configuration. Runtime crossed earlier: Intel was faster through `N=512`, the
platforms were practically equivalent at `N=1024`, and AMD was 39.8–47.4% faster
for `N>=2048`. Consequently, EDP favored Intel through `N=1024` and AMD from
`N=2048` onward.

Thread selection exposed different energy–runtime compromises. On Intel,
8 threads minimized package energy for large GEMMs, while 10 threads reduced
runtime by roughly 7.2–12.9% for only 0.54–0.60% additional package energy. On
AMD, 64 threads minimized energy for `N>=2048`, whereas 32 threads minimized
runtime for the largest matrices. The 32-thread setting reduced runtime by
about 4.7–6.7% relative to the 64-thread energy optimum, at an energy penalty of
approximately 7.7–13.2%. These results show that the fastest configuration is
not generally the energy-minimal configuration and support reporting explicit
runtime-efficient and energy-efficient placement choices.

The AMD campaign was stable (three runtime-unstable and no energy-unstable
configurations), with session-median temperatures of 77–78 °C. The Intel system
operated at a substantially higher thermal state: every session reached 100 °C
and session medians were 94–99 °C. Intel results therefore characterize the
sustained system as configured and should not be interpreted as a
thermally normalized estimate of processor architecture alone.

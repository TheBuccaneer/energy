# Independent review prompt — STREAM analysis pipeline

Review the installed STREAM analysis package independently. Do not trust generated prose.

## Required checks

1. Confirm project paths match the supplied tree.
2. Confirm campaign discovery ignores quickchecks and requires sessions 1–5 with one timestamp.
3. Confirm CPU grids are Intel `{1,2,4,8,10,16,20}` and AMD plus `{32,64}`.
4. Confirm GPU has one `gpu_resident` configuration and `num_threads=-1`.
5. Recompute raw formulas:
   - FLOPs = `2*N*batches`
   - logical bytes/op = `12*N`
   - time/op = interval/batches
   - energy/op = measured energy/batches
6. Check the explicit rounding tolerances for CSV-serialized fields and verify they are neither exact-equality checks nor excessively loose.
7. Confirm CPU primary energy is package RAPL and GPU primary energy is board NVML.
8. Confirm Intel package+DRAM is retained only as a sensitivity and not silently mixed into the primary CPU/GPU comparison.
9. Confirm ten repetitions are reduced to one session median and all statistics use five session medians.
10. Recompute runtime, energy, EDP, logical bandwidth and logical GB/J.
11. Confirm runtime and logical bandwidth are inverse views, and energy and logical GB/J are inverse views.
12. Recompute tie-aware leaders using CI overlap and ±2% practical equivalence.
13. Recompute all 270 pairwise native-best rows and their classifications.
14. Recompute strict and practical Pareto sets from runtime and energy only.
15. Confirm no raw row is silently removed and robust outliers are diagnostic only.
16. Confirm reports state that logical bandwidth is not measured physical traffic and GPU resident mode excludes PCIe transfers.

## Expected aggregate dimensions

- session medians: 810
- configuration summaries: 162
- native policy leaders: 180
- selected policy session rows: 900
- pairwise rows: 270
- global winner rows: 45

## Required verdict

Return one of:

- PASS
- PASS WITH CHANGES
- FAIL

Separate code/statistical defects from expected methodological warnings.

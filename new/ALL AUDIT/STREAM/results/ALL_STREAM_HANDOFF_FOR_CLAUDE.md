# STREAM handoff for independent audit

## Required independent checks

1. Recompute `2*N*batches` and `12*N` from raw rows.
2. Recompute time/op and energy/op with CSV rounding allowances.
3. Confirm exactly five sessions and ten repetitions per configuration.
4. Recompute runtime, energy, EDP, logical bandwidth and logical GB/J from session medians.
5. Confirm runtime winner equals logical-bandwidth winner and energy winner equals logical-GB/J winner.
6. Recompute tie-aware leader sets and 2% practical-equivalence rules.
7. Recompute strict/practical Pareto status from runtime and energy only.
8. Keep CPU-package/GPU-board asymmetry and resident GPU scope explicit.

## Expected aggregate dimensions

- Unified session medians: 810 rows
- Unified configurations: 162 rows
- Native policy leaders: 180 rows
- Selected policy session medians: 900 rows
- Pairwise native-best rows: 270 rows
- Global metric winner rows: 45 rows

## Current headline counts

- Clear device-level fastest-vs-greenest conflicts: 6
- Clear within-platform configuration conflicts: 0
- Stability warnings: 39

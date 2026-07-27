# REDUCTION handoff for independent audit

## Required independent checks

1. Recompute `(N-1)*batches` and `4*N+4` from raw rows.
2. Recompute time/op and energy/op with CSV serialization tolerances.
3. Confirm exactly five sessions and ten repetitions per configuration.
4. Recompute runtime, energy, EDP, logical useful-data rate and logical GB/J from session medians.
5. Confirm runtime winner equals logical-rate winner and energy winner equals logical-GB/J winner.
6. Recompute tie-aware leader sets and the 2% practical-equivalence rule.
7. Recompute strict/practical Pareto status from runtime and energy only.
8. Recompute exact runtime-winner versus exact energy-winner regret, while retaining its post-selection label.
9. Keep CPU-package/GPU-board asymmetry, internal reduction traffic and resident GPU scope explicit.

## Expected aggregate dimensions

- Unified session medians: 810 rows
- Unified configurations: 162 rows
- Native policy leaders: 180 rows
- Selected policy session medians: 900 rows
- Pairwise native-best rows: 270 rows
- Global metric winner rows: 45 rows
- Exact-winner regret rows: 36 rows

## Current headline counts

- Clear device-level fastest-vs-greenest conflicts: 6
- Clear tie-aware within-platform configuration conflicts: 2
- Clear descriptive exact-winner energy penalties: 8
- Stability warnings: 57

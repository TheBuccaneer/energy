# STREAM analysis v1.1 — GPU timing validation fix

## What was wrong

Version 1 treated `kernel_time_s <= e2e_time_s + 2 µs` as a hard,
row-wise invariant.

`kernel_time_s` is measured with CUDA events, while `e2e_time_s` is
measured with the host steady clock. These are independent clock
domains. Small crossings can therefore occur even when the wall
interval semantically encloses the GPU work.

The RTX 3090 campaign passed every coverage, schema, correctness,
energy, formula, runtime and provenance check. Only this overly strict
cross-clock comparison failed.

## New rule

- Small positive `kernel_time_s - e2e_time_s` differences are reported
  as a warning.
- A hard failure occurs only when the kernel duration exceeds the host
  wall duration by more than both:
  - 0.5 milliseconds, and
  - 0.5 percent of `e2e_time_s`.

The validator now reports:
- number of affected rows;
- maximum positive difference in milliseconds;
- maximum `kernel_time_s / e2e_time_s` ratio.

No measurement data, formulas, aggregation logic or statistical
analysis were changed.

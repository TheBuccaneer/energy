# REDUCTION Analysis v1.1 — Audit Note

## Trigger

The first real AMD campaign contained 4,050 valid rows. Coverage, schema,
semantics, checksums, time, energy and all other derived metrics passed.
Only these checks failed:

- `flops_total`: 4,050 rows
- `logical_bytes_per_op`: 3,150 rows

The campaign report showed 4,022 `in_range` and 28 `above` rows; the latter
are warnings and do not invalidate the campaign.

## Root cause

The benchmark source computes the frozen formulas correctly:

```text
flops_total           = (N-1) * batches
logical_bytes_per_op  = 4*N+4
```

The shared CSV writer serializes both fields using C++ scientific notation
with six digits after the decimal point. For example, an exact value such as

```text
103295974176
```

is stored as:

```text
1.032960e+11
```

and parsed back as:

```text
103296000000
```

v1.0 incorrectly compared this rounded CSV value with the unrounded
mathematical value using near-exact tolerances.

For `4*N+4`, the `+4` remains visible at 1M and 2M but is rounded away from
4M onward. On AMD, seven affected sizes × nine thread counts × ten
repetitions × five sessions explain exactly the observed 3,150 failures.

## Resolution

v1.1 validates the exact value that the frozen formula would produce after
the writer's documented scientific-notation serialization. It also scans
the present source file for the authoritative `N-1` and `4*N+4` declarations.

This is a validator correction. It does not change or repair measurement
data and does not require a rerun.

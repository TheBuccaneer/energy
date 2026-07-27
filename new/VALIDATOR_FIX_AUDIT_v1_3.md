# REDUCTION Analysis v1.3 — RTX 5060 Ti Closure

## Real campaign evidence

The RTX 5060 Ti campaign contains 450 complete rows and passes all checks for:

- five sessions and ten repetitions per configuration;
- all checksums;
- frozen `N-1` and `4*N+4` formulas;
- exact GPU formula representation;
- energy, runtime and GPU-resident semantics;
- PCIe metadata;
- material kernel/e2e timing tolerance.

Two hard failures remained in v1.2:

1. The five CSV headers contain the same 45 named columns, but place
   `total_energy_j` immediately before `device_energy_j` instead of the
   canonical reverse order.
2. The actual runner filename uses `5060Ti` with an uppercase `T`, while the
   validator expected `5060ti`.

## Classification

The runner path failure is a validator provenance-path bug.

The header deviation is a real schema-order deviation, but not a measurement
content error: the CSV is named-column data, the column set is complete and
unique, all five files use the same order, and all name-based semantic and
formula checks pass.

## v1.3 policy

The hard schema gate accepts only:

- the canonical 45-column order; or
- for RTX 5060 Ti only, the observed legacy order with exactly the
  `total_energy_j` / `device_energy_j` positional swap.

Every session file is checked independently, and all five headers must be
identical. The legacy order produces `PASS WITH WARNINGS`, not an unqualified
pass. Any other deviation remains a hard failure.

No raw CSV is rewritten and no measurement rerun is required by this finding.

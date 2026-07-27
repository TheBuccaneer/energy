# REDUCTION Analysis v1.2 — GPU Validator Audit

The RTX 3090 campaign has 450 complete rows. Coverage, checksums, runtime
status, energy formulas, GPU-resident semantics, PCIe metadata and the
material timing rule all passed.

The failure table showed `flops_total == expected_exact` for the GPU rows.
v1.1 incorrectly required the CPU scientific-6 representation on GPUs.
It also searched only for a source variable named `flops_per_op`, while
the GPU source computes `flops_total` directly.

v1.2 preserves the formulas and accepts only:

```text
exact_formula_value
scientific_6(exact_formula_value)
```

A campaign is rejected when distinguishable rows mix both modes. The
source-level `N-1` and `4*N+4` checks remain mandatory.

The observed RTX 3090 maximum kernel/e2e crossing of about 0.007246 ms is
far below the hard threshold `max(0.5 ms, 0.5% of e2e)` and therefore
remains a warning.

This is a validator correction. No CPU or GPU rerun is required by this
finding.

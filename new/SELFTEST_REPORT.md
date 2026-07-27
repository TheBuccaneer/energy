# REDUCTION analysis pipeline v1.3 — self-test report

## Scope

The pipeline was tested without using or modifying real measurement data.
A synthetic project tree reproduced the expected four-platform directory
structure and generated complete `cpu-gpu-v2` campaigns with:

- 5 sessions per platform;
- 10 repetitions per configuration;
- 9 sizes;
- Intel 7-thread grid;
- AMD 9-thread grid;
- one resident configuration per GPU;
- 8,100 raw rows in total.

## Static tests

```text
Python compilation: PASS
Shell syntax:       PASS
```

All Python files passed `python3 -m py_compile`. All shell files passed
`bash -n`.

## End-to-end synthetic test

The root launcher completed all four platform analyses and the combined audit.

```text
Platform validators:       4/4 PASS
Combined preflight:        PASS WITH EXPECTED SCOPE WARNINGS
Combined integrity audit:  PASS WITH EXPECTED SCOPE WARNINGS
```

Generated dimensions:

```text
unified_session_medians.csv                  810
unified_configuration_summary.csv            162
native_policy_leaders.csv                     180
native_policy_session_medians.csv             900
pairwise_native_best_comparisons.csv          270
all_platform_metric_winners.csv                45
placement_by_size.csv                           9
all_configuration_pareto.csv                  162
all_platform_stability.csv                    162
within_platform_energy_runtime_tradeoffs.csv   36
within_platform_exact_winner_regret.csv         36
```

## Mutation tests

1. A deliberately incorrect `logical_bytes_per_op` value was rejected.
2. A 0.1 ms CUDA-event-over-host crossing produced a warning, not a hard fail.
3. A material 10 ms CUDA-event-over-host crossing was rejected.

## Residual limitation

Synthetic testing verifies pipeline mechanics, dimensions, formulas and failure
gates. It does not establish correctness of future hardware measurements. Each
real campaign must still pass source/runner provenance, row-level validation,
checksum, coverage and integrity gates.

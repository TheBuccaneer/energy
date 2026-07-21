# All-platform STREAM audit report

## Scope

This report combines independently validated STREAM campaigns from Intel CPU, AMD CPU, RTX 3090 and RTX 5060 Ti.

## Coverage

- Four platforms
- Nine element counts
- Five sessions per configuration
- Ten technical repetitions per session
- Session medians as the primary statistical units

## Central outputs

- `unified_session_medians.csv`
- `unified_configuration_summary.csv`
- `native_policy_leaders.csv`
- `pairwise_native_best_comparisons.csv`
- `all_platform_metric_winners.csv`
- `placement_by_size.csv`
- `all_configuration_pareto.csv`
- `all_platform_stability.csv`

## Immediate audit indicators

- Clear device-level fastest-vs-greenest conflicts: 6
- Clear within-platform configuration conflicts: 0
- Configurations above 5% CV in runtime or energy: 39

## Methodological constraints

The primary decision axes are runtime and measured device-domain energy. Logical bandwidth and logical GB/J are normalized presentation views. EDP is joint. Cross-device energy claims apply only within the measured device domains. No physical memory-traffic claim is made from `12*N` logical bytes.

# All-platform REDUCTION analysis and audit

Combines validated campaigns from Intel CPU, AMD CPU, RTX 3090 and RTX 5060 Ti.

Run the root launcher:

```bash
./run_reduction_analysis_all.sh
```

Main outputs are written to:

```text
ALL AUDIT/REDUCTION/results/
```

The pipeline separates runtime, energy and EDP objectives, uses five session medians per configuration, reports tie-aware leaders, exact-winner descriptive regret, strict/practical Pareto fronts, pairwise comparisons and stability.

Semantics: `sum(x[0:N]) -> FP32 scalar`, `N-1` additions, `4*N+4` logical bytes. Logical useful-data rate is not measured physical traffic.

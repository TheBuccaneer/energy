# Direction of pairwise effect columns

- `a_over_b_ratio < 1` favors A for lower-is-better metrics: runtime, energy, EDP.
- `a_over_b_ratio > 1` favors A for higher-is-better metrics: throughput, GFLOP/J.
- `probability_a_better` is oriented to the metric and always means the probability that A is better.
- `cliffs_delta_a_minus_b > 0` only means A has numerically larger values:
  - favorable for throughput and GFLOP/J;
  - unfavorable for runtime, energy and EDP.

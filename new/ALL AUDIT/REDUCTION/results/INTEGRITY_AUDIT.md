# Independent integrity audit — REDUCTION

## Verdict

**PASS WITH WARNINGS**

The audit recomputed central metric identities, inverse-view leader consistency,
pairwise classifications, and Pareto status from generated CSVs rather than prose.

## Hard failures

_None._

## Warnings

| platform   | category   | check                                 | severity   | status   | observed                                           | expected                             |
|:-----------|:-----------|:--------------------------------------|:-----------|:---------|:---------------------------------------------------|:-------------------------------------|
| ALL        | statistics | exact_winner_regret_is_post_selection | WARN       | WARN     | same five sessions used for selection and interval | descriptive only                     |
| ALL        | semantics  | runtime_bandwidth_not_independent     | WARN       | WARN     | inverse views                                      | do not count as independent evidence |
| ALL        | semantics  | energy_gbj_not_independent            | WARN       | WARN     | inverse views                                      | do not count as independent evidence |
| ALL        | semantics  | logical_bandwidth_scope               | WARN       | WARN     | (4*N+4)/runtime                                    | not measured physical traffic        |
| ALL        | semantics  | energy_domain_asymmetry               | WARN       | WARN     | CPU package vs GPU board                           | must remain explicit                 |
| ALL        | semantics  | resident_gpu_scope                    | WARN       | WARN     | PCIe excluded                                      | must remain explicit                 |

## All checks

| platform   | category            | check                                 | severity   | status   | observed                                           | expected                             |
|:-----------|:--------------------|:--------------------------------------|:-----------|:---------|:---------------------------------------------------|:-------------------------------------|
| ALL        | coverage            | unified_session_rows                  | FAIL       | PASS     | 810                                                | 810                                  |
| ALL        | coverage            | configuration_rows                    | FAIL       | PASS     | 162                                                | 162                                  |
| ALL        | coverage            | leader_rows                           | FAIL       | PASS     | 180                                                | 180                                  |
| ALL        | coverage            | selected_session_rows                 | FAIL       | PASS     | 900                                                | 900                                  |
| ALL        | coverage            | pairwise_rows                         | FAIL       | PASS     | 270                                                | 270                                  |
| ALL        | coverage            | winner_rows                           | FAIL       | PASS     | 45                                                 | 45                                   |
| ALL        | coverage            | exact_winner_regret_rows              | FAIL       | PASS     | 36                                                 | 36                                   |
| ALL        | identity            | logical_bandwidth                     | FAIL       | PASS     | max_abs_error=2.274e-13                            | <1e-12                               |
| ALL        | identity            | logical_gb_per_j                      | FAIL       | PASS     | max_abs_error=1.776e-15                            | <1e-12                               |
| ALL        | identity            | edp                                   | FAIL       | PASS     | max_abs_error=9.926e-17                            | <1e-12                               |
| ALL        | identity            | throughput                            | FAIL       | PASS     | max_abs_error=5.684e-14                            | <1e-12                               |
| ALL        | identity            | efficiency                            | FAIL       | PASS     | max_abs_error=4.441e-16                            | <1e-12                               |
| ALL        | leaders             | runtime_equals_bandwidth_exact        | FAIL       | PASS     | 0                                                  | 0                                    |
| ALL        | leaders             | energy_equals_bytes_per_j_exact       | FAIL       | PASS     | 0                                                  | 0                                    |
| ALL        | pairwise            | ratio_identity                        | FAIL       | PASS     | max_abs_error=1.421e-14                            | <1e-12                               |
| ALL        | pairwise            | classification_rule                   | FAIL       | PASS     | 0                                                  | 0                                    |
| ALL        | pareto              | strict_recomputed                     | FAIL       | PASS     | 0                                                  | 0                                    |
| ALL        | pareto              | practical_recomputed                  | FAIL       | PASS     | 0                                                  | 0                                    |
| ALL        | exact_winner_regret | energy_penalty_identity               | FAIL       | PASS     | max_abs_error=3.553e-15                            | <1e-12                               |
| ALL        | exact_winner_regret | runtime_gain_identity                 | FAIL       | PASS     | max_abs_error=1.155e-14                            | <1e-12                               |
| ALL        | statistics          | exact_winner_regret_is_post_selection | WARN       | WARN     | same five sessions used for selection and interval | descriptive only                     |
| ALL        | semantics           | runtime_bandwidth_not_independent     | WARN       | WARN     | inverse views                                      | do not count as independent evidence |
| ALL        | semantics           | energy_gbj_not_independent            | WARN       | WARN     | inverse views                                      | do not count as independent evidence |
| ALL        | semantics           | logical_bandwidth_scope               | WARN       | WARN     | (4*N+4)/runtime                                    | not measured physical traffic        |
| ALL        | semantics           | energy_domain_asymmetry               | WARN       | WARN     | CPU package vs GPU board                           | must remain explicit                 |
| ALL        | semantics           | resident_gpu_scope                    | WARN       | WARN     | PCIe excluded                                      | must remain explicit                 |

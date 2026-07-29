# CONV2D platform analysis — Intel Core i9-7900X

## Campaign

- Selected campaign: `20260729_004307`
- Raw measurements: 2100
- Session medians: 210
- Configurations summarized: 42
- Energy domain: CPU package RAPL plus DRAM RAPL when available

## Tie-aware leaders (2% practical tolerance)

|   problem_size | objective         | configuration   |   num_threads |          value | exact_winner   | within_2pct   |
|---------------:|:------------------|:----------------|--------------:|---------------:|:---------------|:--------------|
|              1 | runtime           | 10T             |            10 |    0.00455617  | False          | True          |
|              1 | runtime           | 20T             |            20 |    0.00453487  | True           | True          |
|              1 | energy            | 20T             |            20 |    0.718269    | True           | True          |
|              1 | edp               | 20T             |            20 |    0.0032551   | True           | True          |
|              1 | throughput        | 10T             |            10 | 1623.9         | False          | True          |
|              1 | throughput        | 20T             |            20 | 1631.53        | True           | True          |
|              1 | energy_efficiency | 20T             |            20 |   10.3008      | True           | True          |
|              2 | runtime           | 10T             |            10 |    0.00236744  | False          | True          |
|              2 | runtime           | 20T             |            20 |    0.00236109  | True           | True          |
|              2 | energy            | 20T             |            20 |    0.379173    | True           | True          |
|              2 | edp               | 20T             |            20 |    0.000896923 | True           | True          |
|              2 | throughput        | 10T             |            10 | 1562.6         | False          | True          |
|              2 | throughput        | 20T             |            20 | 1566.81        | True           | True          |
|              2 | energy_efficiency | 20T             |            20 |    9.75643     | True           | True          |
|              3 | runtime           | 10T             |            10 |    0.00224244  | False          | True          |
|              3 | runtime           | 20T             |            20 |    0.00224112  | True           | True          |
|              3 | energy            | 10T             |            10 |    0.358497    | True           | True          |
|              3 | edp               | 10T             |            10 |    0.000816851 | True           | True          |
|              3 | throughput        | 10T             |            10 | 1649.71        | False          | True          |
|              3 | throughput        | 20T             |            20 | 1650.69        | True           | True          |
|              3 | energy_efficiency | 10T             |            10 |   10.3191      | True           | True          |
|              4 | runtime           | 20T             |            20 |    0.00249149  | True           | True          |
|              4 | energy            | 10T             |            10 |    0.410718    | True           | True          |
|              4 | energy            | 16T             |            16 |    0.414687    | False          | True          |
|              4 | edp               | 10T             |            10 |    0.00105624  | True           | True          |
|              4 | throughput        | 20T             |            20 | 1484.8         | True           | True          |
|              4 | energy_efficiency | 10T             |            10 |    9.00709     | True           | True          |
|              4 | energy_efficiency | 16T             |            16 |    8.92088     | False          | True          |
|              5 | runtime           | 10T             |            10 |    0.00631741  | True           | True          |
|              5 | energy            | 8T              |             8 |    1.01336     | True           | True          |
|              5 | energy            | 10T             |            10 |    1.01739     | False          | True          |
|              5 | edp               | 10T             |            10 |    0.0064295   | True           | True          |
|              5 | throughput        | 10T             |            10 | 1195.57        | True           | True          |
|              5 | energy_efficiency | 8T              |             8 |    7.45333     | True           | True          |
|              5 | energy_efficiency | 10T             |            10 |    7.42376     | False          | True          |
|              6 | runtime           | 20T             |            20 |    0.00969994  | True           | True          |
|              6 | energy            | 10T             |            10 |    1.59263     | True           | True          |
|              6 | energy            | 20T             |            20 |    1.6082      | False          | True          |
|              6 | edp               | 10T             |            10 |    0.0157685   | False          | True          |
|              6 | edp               | 20T             |            20 |    0.0156355   | True           | True          |
|              6 | throughput        | 10T             |            10 | 1328.96        | False          | True          |
|              6 | throughput        | 20T             |            20 | 1356.02        | True           | True          |
|              6 | energy_efficiency | 10T             |            10 |    8.25888     | True           | True          |
|              6 | energy_efficiency | 20T             |            20 |    8.17891     | False          | True          |

## Strict and practical Pareto status

|   problem_size | configuration   |   num_threads |   runtime_per_op_s |   total_energy_per_op_j | strict_pareto   | practical_pareto_2pct   | platform   |   session_count |   runtime_per_op_s_robust_cv_pct |   total_energy_per_op_j_robust_cv_pct |   edp_total_j_s_robust_cv_pct |
|---------------:|:----------------|--------------:|-------------------:|------------------------:|:----------------|:------------------------|:-----------|----------------:|---------------------------------:|--------------------------------------:|------------------------------:|
|              1 | 1T              |             1 |         0.0365684  |                2.17903  | False           | False                   | INTEL      |               5 |                         0.589358 |                              0.256978 |                      0.932602 |
|              1 | 2T              |             2 |         0.0189302  |                1.49059  | False           | False                   | INTEL      |               5 |                         1.35753  |                              1.37894  |                      1.03538  |
|              1 | 4T              |             4 |         0.00986395 |                1.03561  | False           | False                   | INTEL      |               5 |                         2.30132  |                              2.27283  |                      1.01054  |
|              1 | 8T              |             8 |         0.00549681 |                0.813031 | False           | False                   | INTEL      |               5 |                         2.28323  |                              0.785381 |                      3.28984  |
|              1 | 10T             |            10 |         0.00455617 |                0.734954 | False           | True                    | INTEL      |               5 |                         6.68547  |                              1.08591  |                      2.64181  |
|              1 | 16T             |            16 |         0.00510194 |                0.799798 | False           | False                   | INTEL      |               5 |                         4.42628  |                              0.833225 |                      4.53053  |
|              1 | 20T             |            20 |         0.00453487 |                0.718269 | True            | True                    | INTEL      |               5 |                         0.781457 |                              1.65588  |                      2.23405  |
|              2 | 1T              |             1 |         0.0195395  |                1.17764  | False           | False                   | INTEL      |               5 |                         3.30687  |                              1.77714  |                      1.21219  |
|              2 | 2T              |             2 |         0.0101917  |                0.814011 | False           | False                   | INTEL      |               5 |                         2.73683  |                              1.26101  |                      4.91758  |
|              2 | 4T              |             4 |         0.0053187  |                0.562642 | False           | False                   | INTEL      |               5 |                         0.74402  |                              1.50664  |                      0.767843 |
|              2 | 8T              |             8 |         0.0028661  |                0.425898 | False           | False                   | INTEL      |               5 |                         3.35962  |                              3.85669  |                      8.31133  |
|              2 | 10T             |            10 |         0.00236744 |                0.393457 | False           | True                    | INTEL      |               5 |                         3.98449  |                              1.07865  |                      3.6587   |
|              2 | 16T             |            16 |         0.00273006 |                0.420182 | False           | False                   | INTEL      |               5 |                         4.05526  |                              0.184403 |                      3.79333  |
|              2 | 20T             |            20 |         0.00236109 |                0.379173 | True            | True                    | INTEL      |               5 |                         2.7003   |                              0.265057 |                      2.66848  |
|              3 | 1T              |             1 |         0.0188872  |                1.13113  | False           | False                   | INTEL      |               5 |                         2.41883  |                              2.22754  |                      4.74003  |
|              3 | 2T              |             2 |         0.0091348  |                0.747542 | False           | False                   | INTEL      |               5 |                         4.8456   |                              1.6401   |                      1.04218  |
|              3 | 4T              |             4 |         0.00484339 |                0.517818 | False           | False                   | INTEL      |               5 |                         0.752893 |                              1.5528   |                      2.24722  |
|              3 | 8T              |             8 |         0.00262357 |                0.391517 | False           | False                   | INTEL      |               5 |                         6.34499  |                              3.67209  |                     11.2705   |
|              3 | 10T             |            10 |         0.00224244 |                0.358497 | True            | True                    | INTEL      |               5 |                         3.70977  |                              1.89853  |                      1.77885  |
|              3 | 16T             |            16 |         0.00256255 |                0.410172 | False           | False                   | INTEL      |               5 |                         6.34298  |                              0.488154 |                      8.61605  |
|              3 | 20T             |            20 |         0.00224112 |                0.39074  | True            | True                    | INTEL      |               5 |                         7.71099  |                              1.75978  |                      5.12675  |
|              4 | 1T              |             1 |         0.0228942  |                1.48171  | False           | False                   | INTEL      |               5 |                         2.90852  |                              5.06654  |                      5.3648   |
|              4 | 2T              |             2 |         0.0114243  |                0.89414  | False           | False                   | INTEL      |               5 |                         3.94002  |                              2.30045  |                      5.67255  |
|              4 | 4T              |             4 |         0.0056336  |                0.5916   | False           | False                   | INTEL      |               5 |                         1.36147  |                              0.345067 |                      1.28616  |
|              4 | 8T              |             8 |         0.00307133 |                0.45161  | False           | False                   | INTEL      |               5 |                         3.01389  |                              2.36288  |                      5.4179   |
|              4 | 10T             |            10 |         0.00257606 |                0.410718 | True            | True                    | INTEL      |               5 |                         1.79587  |                              0.26248  |                      3.23464  |
|              4 | 16T             |            16 |         0.00260604 |                0.414687 | False           | True                    | INTEL      |               5 |                         2.31692  |                              0.874139 |                      6.90137  |
|              4 | 20T             |            20 |         0.00249149 |                0.430116 | True            | True                    | INTEL      |               5 |                         6.34908  |                              0.435577 |                      4.32539  |
|              5 | 1T              |             1 |         0.045671   |                2.75299  | False           | False                   | INTEL      |               5 |                         2.79517  |                             11.88     |                     12.7931   |
|              5 | 2T              |             2 |         0.022737   |                1.90375  | False           | False                   | INTEL      |               5 |                         1.51547  |                              1.64552  |                      1.72568  |
|              5 | 4T              |             4 |         0.0121015  |                1.29708  | False           | False                   | INTEL      |               5 |                         2.72034  |                              2.26444  |                      1.08093  |
|              5 | 8T              |             8 |         0.00672689 |                1.01336  | True            | True                    | INTEL      |               5 |                         2.07837  |                              3.21086  |                      5.40896  |
|              5 | 10T             |            10 |         0.00631741 |                1.01739  | True            | True                    | INTEL      |               5 |                         0.371187 |                              0.355097 |                      0.459528 |
|              5 | 16T             |            16 |         0.00688921 |                1.07734  | False           | False                   | INTEL      |               5 |                         0.992006 |                              0.815508 |                      2.41474  |
|              5 | 20T             |            20 |         0.00646358 |                1.04985  | False           | False                   | INTEL      |               5 |                         0.709926 |                              1.03989  |                      0.389205 |
|              6 | 1T              |             1 |         0.0712389  |                4.30706  | False           | False                   | INTEL      |               5 |                         1.78869  |                              1.27003  |                      1.14043  |
|              6 | 2T              |             2 |         0.0359557  |                2.95153  | False           | False                   | INTEL      |               5 |                         4.57109  |                              2.25646  |                      4.66721  |
|              6 | 4T              |             4 |         0.0191117  |                2.06095  | False           | False                   | INTEL      |               5 |                         3.23891  |                              3.06627  |                      4.29137  |
|              6 | 8T              |             8 |         0.0110583  |                1.64867  | False           | False                   | INTEL      |               5 |                         1.06271  |                              2.43169  |                      3.83411  |
|              6 | 10T             |            10 |         0.00989747 |                1.59263  | True            | True                    | INTEL      |               5 |                         0.992699 |                              0.665019 |                      1.64119  |
|              6 | 16T             |            16 |         0.0113506  |                1.76689  | False           | False                   | INTEL      |               5 |                         0.203047 |                              2.36887  |                      1.94779  |
|              6 | 20T             |            20 |         0.00969994 |                1.6082   | True            | True                    | INTEL      |               5 |                         0.318656 |                              1.29209  |                      0.992011 |

## Interpretation guardrails

- Ten repetitions are technical repetitions; the five session medians are the scientific units.
- Runtime, total measured energy and EDP are reported separately.
- cuDNN and oneDNN may select different legal algorithms or primitives per shape and platform; `problem_spec` and `implementation` retain that provenance.
- Cross-platform energy comparison must retain the meter-domain caveat: CPU package/DRAM RAPL versus GPU board NVML.

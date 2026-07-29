# CONV2D platform analysis — AMD Threadripper 3970X

## Campaign

- Selected campaign: `20260729_005653`
- Raw measurements: 2700
- Session medians: 270
- Configurations summarized: 54
- Energy domain: CPU package RAPL plus DRAM RAPL when available

## Tie-aware leaders (2% practical tolerance)

|   problem_size | objective         | configuration   |   num_threads |          value | exact_winner   | within_2pct   |
|---------------:|:------------------|:----------------|--------------:|---------------:|:---------------|:--------------|
|              1 | runtime           | 32T             |            32 |    0.00219826  | True           | True          |
|              1 | runtime           | 64T             |            64 |    0.00221127  | False          | True          |
|              1 | energy            | 32T             |            32 |    0.578634    | True           | True          |
|              1 | edp               | 32T             |            32 |    0.00127209  | True           | True          |
|              1 | throughput        | 32T             |            32 | 3365.73        | True           | True          |
|              1 | throughput        | 64T             |            64 | 3345.93        | False          | True          |
|              1 | energy_efficiency | 32T             |            32 |   12.7866      | True           | True          |
|              1 | energy_efficiency | 64T             |            64 |   12.5346      | False          | True          |
|              2 | runtime           | 32T             |            32 |    0.00115798  | True           | True          |
|              2 | energy            | 64T             |            64 |    0.302222    | True           | True          |
|              2 | edp               | 32T             |            32 |    0.00036561  | False          | True          |
|              2 | edp               | 64T             |            64 |    0.000359442 | True           | True          |
|              2 | throughput        | 32T             |            32 | 3194.69        | True           | True          |
|              2 | energy_efficiency | 64T             |            64 |   12.2406      | True           | True          |
|              3 | runtime           | 32T             |            32 |    0.00115548  | True           | True          |
|              3 | energy            | 64T             |            64 |    0.299612    | True           | True          |
|              3 | edp               | 32T             |            32 |    0.000357163 | False          | True          |
|              3 | edp               | 64T             |            64 |    0.000354521 | True           | True          |
|              3 | throughput        | 32T             |            32 | 3201.58        | True           | True          |
|              3 | energy_efficiency | 64T             |            64 |   12.3472      | True           | True          |
|              4 | runtime           | 32T             |            32 |    0.00116071  | True           | True          |
|              4 | energy            | 64T             |            64 |    0.290327    | True           | True          |
|              4 | edp               | 64T             |            64 |    0.000347639 | True           | True          |
|              4 | throughput        | 32T             |            32 | 3187.18        | True           | True          |
|              4 | energy_efficiency | 64T             |            64 |   12.7421      | True           | True          |
|              5 | runtime           | 64T             |            64 |    0.00287995  | True           | True          |
|              5 | energy            | 64T             |            64 |    0.74278     | True           | True          |
|              5 | edp               | 64T             |            64 |    0.00213873  | True           | True          |
|              5 | throughput        | 64T             |            64 | 2622.58        | True           | True          |
|              5 | energy_efficiency | 64T             |            64 |   10.1684      | True           | True          |
|              6 | runtime           | 32T             |            32 |    0.00575229  | True           | True          |
|              6 | runtime           | 64T             |            64 |    0.00581804  | False          | True          |
|              6 | energy            | 64T             |            64 |    1.40561     | True           | True          |
|              6 | edp               | 64T             |            64 |    0.00817201  | True           | True          |
|              6 | throughput        | 32T             |            32 | 2286.63        | True           | True          |
|              6 | throughput        | 64T             |            64 | 2260.79        | False          | True          |
|              6 | energy_efficiency | 64T             |            64 |    9.35775     | True           | True          |

## Strict and practical Pareto status

|   problem_size | configuration   |   num_threads |   runtime_per_op_s |   total_energy_per_op_j | strict_pareto   | practical_pareto_2pct   | platform   |   session_count |   runtime_per_op_s_robust_cv_pct |   total_energy_per_op_j_robust_cv_pct |   edp_total_j_s_robust_cv_pct |
|---------------:|:----------------|--------------:|-------------------:|------------------------:|:----------------|:------------------------|:-----------|----------------:|---------------------------------:|--------------------------------------:|------------------------------:|
|              1 | 1T              |             1 |         0.0580189  |                4.51392  | False           | False                   | AMD        |               5 |                       0.00950564 |                             0.792945  |                     0.393654  |
|              1 | 2T              |             2 |         0.029196   |                2.81181  | False           | False                   | AMD        |               5 |                       0.271423   |                             0.351114  |                     0.423356  |
|              1 | 4T              |             4 |         0.0143661  |                1.90246  | False           | False                   | AMD        |               5 |                       0.748732   |                             0.423879  |                     1.20817   |
|              1 | 8T              |             8 |         0.00719886 |                1.43698  | False           | False                   | AMD        |               5 |                       0.218492   |                             0.0749457 |                     0.143783  |
|              1 | 10T             |            10 |         0.00588833 |                1.3504   | False           | False                   | AMD        |               5 |                       0.142603   |                             0.023692  |                     0.150849  |
|              1 | 16T             |            16 |         0.00379272 |                1.08726  | False           | False                   | AMD        |               5 |                       0.458755   |                             0.419548  |                     0.857621  |
|              1 | 20T             |            20 |         0.00314141 |                0.90093  | False           | False                   | AMD        |               5 |                       1.05444    |                             0.980552  |                     2.05452   |
|              1 | 32T             |            32 |         0.00219826 |                0.578634 | True            | True                    | AMD        |               5 |                       0.396686   |                             0.491692  |                     0.898911  |
|              1 | 64T             |            64 |         0.00221127 |                0.590266 | False           | True                    | AMD        |               5 |                       0.22248    |                             0.315188  |                     0.492679  |
|              2 | 1T              |             1 |         0.0337963  |                2.52585  | False           | False                   | AMD        |               5 |                       2.90633    |                             2.23123   |                     5.4413    |
|              2 | 2T              |             2 |         0.0164453  |                1.51066  | False           | False                   | AMD        |               5 |                       2.95139    |                             1.66932   |                     4.63819   |
|              2 | 4T              |             4 |         0.00750685 |                0.960303 | False           | False                   | AMD        |               5 |                       0.533665   |                             0.291624  |                     0.383864  |
|              2 | 8T              |             8 |         0.00382914 |                0.727696 | False           | False                   | AMD        |               5 |                       0.52967    |                             0.316898  |                     0.853968  |
|              2 | 10T             |            10 |         0.00310999 |                0.681705 | False           | False                   | AMD        |               5 |                       0.441823   |                             1.02628   |                     1.33226   |
|              2 | 16T             |            16 |         0.00199946 |                0.572612 | False           | False                   | AMD        |               5 |                       0.0383775  |                             0.128505  |                     0.202606  |
|              2 | 20T             |            20 |         0.00164445 |                0.471693 | False           | False                   | AMD        |               5 |                       0.460861   |                             0.399337  |                     0.835485  |
|              2 | 32T             |            32 |         0.00115798 |                0.315732 | True            | True                    | AMD        |               5 |                       0.909076   |                             1.43035   |                     2.25114   |
|              2 | 64T             |            64 |         0.00119053 |                0.302222 | True            | True                    | AMD        |               5 |                       0.32263    |                             0.364535  |                     0.525253  |
|              3 | 1T              |             1 |         0.0310758  |                2.3729   | False           | False                   | AMD        |               5 |                       0.891219   |                             0.905894  |                     1.82641   |
|              3 | 2T              |             2 |         0.0149275  |                1.42711  | False           | False                   | AMD        |               5 |                       0.120359   |                             0.210491  |                     0.349753  |
|              3 | 4T              |             4 |         0.00741997 |                0.971781 | False           | False                   | AMD        |               5 |                       0.206334   |                             0.692803  |                     1.10562   |
|              3 | 8T              |             8 |         0.00377273 |                0.739041 | False           | False                   | AMD        |               5 |                       0.55102    |                             0.250659  |                     0.712073  |
|              3 | 10T             |            10 |         0.00310421 |                0.696709 | False           | False                   | AMD        |               5 |                       0.156132   |                             0.063022  |                     0.213878  |
|              3 | 16T             |            16 |         0.00197965 |                0.567968 | False           | False                   | AMD        |               5 |                       0.0944765  |                             0.231549  |                     0.32252   |
|              3 | 20T             |            20 |         0.00166423 |                0.477627 | False           | False                   | AMD        |               5 |                       0.171889   |                             0.18054   |                     0.352192  |
|              3 | 32T             |            32 |         0.00115548 |                0.309102 | True            | True                    | AMD        |               5 |                       0.136175   |                             0.360253  |                     0.427775  |
|              3 | 64T             |            64 |         0.00118757 |                0.299612 | True            | True                    | AMD        |               5 |                       0.689166   |                             0.491072  |                     1.63514   |
|              4 | 1T              |             1 |         0.0304818  |                2.32104  | False           | False                   | AMD        |               5 |                       0.54194    |                             0.289584  |                     0.89081   |
|              4 | 2T              |             2 |         0.0151544  |                1.42706  | False           | False                   | AMD        |               5 |                       0.285312   |                             0.383366  |                     0.0951601 |
|              4 | 4T              |             4 |         0.00759734 |                0.967936 | False           | False                   | AMD        |               5 |                       0.318512   |                             0.208043  |                     0.358686  |
|              4 | 8T              |             8 |         0.00386474 |                0.729572 | False           | False                   | AMD        |               5 |                       0.784997   |                             0.4133    |                     1.41056   |
|              4 | 10T             |            10 |         0.00313084 |                0.680624 | False           | False                   | AMD        |               5 |                       0.347343   |                             0.30235   |                     0.367685  |
|              4 | 16T             |            16 |         0.0020041  |                0.574995 | False           | False                   | AMD        |               5 |                       0.558941   |                             0.64277   |                     1.19523   |
|              4 | 20T             |            20 |         0.00168707 |                0.483476 | False           | False                   | AMD        |               5 |                       0.579006   |                             0.420162  |                     1.07309   |
|              4 | 32T             |            32 |         0.00116071 |                0.31738  | True            | True                    | AMD        |               5 |                       1.12254    |                             1.31496   |                     2.4702    |
|              4 | 64T             |            64 |         0.00119659 |                0.290327 | True            | True                    | AMD        |               5 |                       0.618002   |                             0.393801  |                     0.945381  |
|              5 | 1T              |             1 |         0.0935558  |                6.78406  | False           | False                   | AMD        |               5 |                       1.86727    |                             2.19203   |                     3.68606   |
|              5 | 2T              |             2 |         0.0494659  |                4.13988  | False           | False                   | AMD        |               5 |                       7.17649    |                             6.24739   |                    13.4072    |
|              5 | 4T              |             4 |         0.0257073  |                2.71387  | False           | False                   | AMD        |               5 |                       9.5813     |                             2.53808   |                    10.1352    |
|              5 | 8T              |             8 |         0.0115913  |                1.79051  | False           | False                   | AMD        |               5 |                       1.70127    |                             0.531548  |                     2.3407    |
|              5 | 10T             |            10 |         0.0098475  |                1.71953  | False           | False                   | AMD        |               5 |                       0.699414   |                             1.4627    |                     2.08788   |
|              5 | 16T             |            16 |         0.00596149 |                1.40999  | False           | False                   | AMD        |               5 |                       6.51746    |                             2.72889   |                     9.05956   |
|              5 | 20T             |            20 |         0.00517173 |                1.36484  | False           | False                   | AMD        |               5 |                       0.652799   |                             0.178857  |                     0.238468  |
|              5 | 32T             |            32 |         0.0030901  |                0.886894 | False           | False                   | AMD        |               5 |                       0.189655   |                             0.239129  |                     0.432983  |
|              5 | 64T             |            64 |         0.00287995 |                0.74278  | True            | True                    | AMD        |               5 |                       0.219697   |                             0.666864  |                     0.875802  |
|              6 | 1T              |             1 |         0.121109   |                9.19017  | False           | False                   | AMD        |               5 |                       0.927438   |                             3.27917   |                     3.64978   |
|              6 | 2T              |             2 |         0.060888   |                5.50774  | False           | False                   | AMD        |               5 |                       0.243971   |                             1.29945   |                     1.5957    |
|              6 | 4T              |             4 |         0.0307384  |                3.6725   | False           | False                   | AMD        |               5 |                       0.440041   |                             1.06806   |                     0.527123  |
|              6 | 8T              |             8 |         0.0158111  |                2.75228  | False           | False                   | AMD        |               5 |                       0.532683   |                             1.20806   |                     1.673     |
|              6 | 10T             |            10 |         0.0127148  |                2.54486  | False           | False                   | AMD        |               5 |                       0.0773815  |                             0.997148  |                     1.15468   |
|              6 | 16T             |            16 |         0.00860879 |                2.24913  | False           | False                   | AMD        |               5 |                       0.265822   |                             0.121027  |                     0.144051  |
|              6 | 20T             |            20 |         0.00718397 |                2.06092  | False           | False                   | AMD        |               5 |                       0.261879   |                             0.112057  |                     0.360671  |
|              6 | 32T             |            32 |         0.00575229 |                1.5777   | True            | True                    | AMD        |               5 |                       0.1327     |                             0.565856  |                     0.462543  |
|              6 | 64T             |            64 |         0.00581804 |                1.40561  | True            | True                    | AMD        |               5 |                       0.379116   |                             1.0647    |                     2.21153   |

## Interpretation guardrails

- Ten repetitions are technical repetitions; the five session medians are the scientific units.
- Runtime, total measured energy and EDP are reported separately.
- cuDNN and oneDNN may select different legal algorithms or primitives per shape and platform; `problem_spec` and `implementation` retain that provenance.
- Cross-platform energy comparison must retain the meter-domain caveat: CPU package/DRAM RAPL versus GPU board NVML.

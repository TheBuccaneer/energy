# CONV2D all-platform comparison

## Scope

Validated five-session campaigns from AMD Threadripper 3970X, Intel i9-7900X, RTX 3090 and RTX 5060 Ti are combined only after each platform pipeline passes.

## Statistical contract

- Ten repetitions within a session are technical repetitions.
- The median within each session is the primary scientific unit.
- Five session medians form each platform/configuration summary.
- CPU thread counts are optimized independently for each shape and objective before platform-level comparison.
- Practical ties use a ±2% tolerance.

## Cross-platform leaders

|   problem_size | objective         | platform   | platform_label   | configuration   |           value | exact_winner   | within_2pct   | energy_domain                         |
|---------------:|:------------------|:-----------|:-----------------|:----------------|----------------:|:---------------|:--------------|:--------------------------------------|
|              1 | edp               | 3090       | RTX 3090         | gpu_resident    |     3.31803e-05 | True           | True          | GPU board NVML TotalEnergyConsumption |
|              1 | energy            | 3090       | RTX 3090         | gpu_resident    |     0.106577    | True           | True          | GPU board NVML TotalEnergyConsumption |
|              1 | energy_efficiency | 3090       | RTX 3090         | gpu_resident    |    69.4217      | True           | True          | GPU board NVML TotalEnergyConsumption |
|              1 | runtime           | 3090       | RTX 3090         | gpu_resident    |     0.000311445 | True           | True          | GPU board NVML TotalEnergyConsumption |
|              1 | throughput        | 3090       | RTX 3090         | gpu_resident    | 23756.2         | True           | True          | GPU board NVML TotalEnergyConsumption |
|              2 | edp               | 3090       | RTX 3090         | gpu_resident    |     2.83465e-05 | True           | True          | GPU board NVML TotalEnergyConsumption |
|              2 | energy            | 5060ti     | RTX 5060 Ti      | gpu_resident    |     0.0924464   | True           | True          | GPU board NVML TotalEnergyConsumption |
|              2 | energy_efficiency | 5060ti     | RTX 5060 Ti      | gpu_resident    |    40.0165      | True           | True          | GPU board NVML TotalEnergyConsumption |
|              2 | runtime           | 3090       | RTX 3090         | gpu_resident    |     0.000289307 | True           | True          | GPU board NVML TotalEnergyConsumption |
|              2 | throughput        | 3090       | RTX 3090         | gpu_resident    | 12787           | True           | True          | GPU board NVML TotalEnergyConsumption |
|              3 | edp               | 3090       | RTX 3090         | gpu_resident    |     1.86526e-05 | True           | True          | GPU board NVML TotalEnergyConsumption |
|              3 | energy            | 3090       | RTX 3090         | gpu_resident    |     0.0784748   | True           | True          | GPU board NVML TotalEnergyConsumption |
|              3 | energy_efficiency | 3090       | RTX 3090         | gpu_resident    |    47.1409      | True           | True          | GPU board NVML TotalEnergyConsumption |
|              3 | runtime           | 3090       | RTX 3090         | gpu_resident    |     0.000238035 | True           | True          | GPU board NVML TotalEnergyConsumption |
|              3 | throughput        | 3090       | RTX 3090         | gpu_resident    | 15541.3         | True           | True          | GPU board NVML TotalEnergyConsumption |
|              4 | edp               | 3090       | RTX 3090         | gpu_resident    |     3.06734e-05 | True           | True          | GPU board NVML TotalEnergyConsumption |
|              4 | energy            | 5060ti     | RTX 5060 Ti      | gpu_resident    |     0.0861507   | True           | True          | GPU board NVML TotalEnergyConsumption |
|              4 | energy_efficiency | 5060ti     | RTX 5060 Ti      | gpu_resident    |    42.9408      | True           | True          | GPU board NVML TotalEnergyConsumption |
|              4 | runtime           | 3090       | RTX 3090         | gpu_resident    |     0.000306245 | True           | True          | GPU board NVML TotalEnergyConsumption |
|              4 | throughput        | 3090       | RTX 3090         | gpu_resident    | 12079.8         | True           | True          | GPU board NVML TotalEnergyConsumption |
|              5 | edp               | 3090       | RTX 3090         | gpu_resident    |     7.51685e-05 | True           | True          | GPU board NVML TotalEnergyConsumption |
|              5 | energy            | 3090       | RTX 3090         | gpu_resident    |     0.158769    | True           | True          | GPU board NVML TotalEnergyConsumption |
|              5 | energy_efficiency | 3090       | RTX 3090         | gpu_resident    |    47.5715      | True           | True          | GPU board NVML TotalEnergyConsumption |
|              5 | runtime           | 3090       | RTX 3090         | gpu_resident    |     0.000473308 | True           | True          | GPU board NVML TotalEnergyConsumption |
|              5 | throughput        | 3090       | RTX 3090         | gpu_resident    | 15957.7         | True           | True          | GPU board NVML TotalEnergyConsumption |
|              6 | edp               | 3090       | RTX 3090         | gpu_resident    |     0.000188859 | True           | True          | GPU board NVML TotalEnergyConsumption |
|              6 | energy            | 5060ti     | RTX 5060 Ti      | gpu_resident    |     0.217972    | True           | True          | GPU board NVML TotalEnergyConsumption |
|              6 | energy_efficiency | 5060ti     | RTX 5060 Ti      | gpu_resident    |    60.3442      | True           | True          | GPU board NVML TotalEnergyConsumption |
|              6 | runtime           | 3090       | RTX 3090         | gpu_resident    |     0.000747416 | True           | True          | GPU board NVML TotalEnergyConsumption |
|              6 | throughput        | 3090       | RTX 3090         | gpu_resident    | 17598.4         | True           | True          | GPU board NVML TotalEnergyConsumption |

## Runtime-versus-energy conflicts

|   problem_size |   runtime_tie_set | energy_tie_set   | same_tie_set   |   overlap |
|---------------:|------------------:|:-----------------|:---------------|----------:|
|              1 |              3090 | 3090             | True           |      3090 |
|              2 |              3090 | 5060ti           | False          |           |
|              3 |              3090 | 3090             | True           |      3090 |
|              4 |              3090 | 5060ti           | False          |           |
|              5 |              3090 | 3090             | True           |      3090 |
|              6 |              3090 | 5060ti           | False          |           |

## Global runtime-energy Pareto status

| platform   | platform_label         | device_kind   |   problem_size | configuration   |   num_threads |   runtime_per_op_s |   total_energy_per_op_j | strict_pareto   | practical_pareto_2pct   |
|:-----------|:-----------------------|:--------------|---------------:|:----------------|--------------:|-------------------:|------------------------:|:----------------|:------------------------|
| AMD        | AMD Threadripper 3970X | CPU           |              1 | 1T              |             1 |        0.0580189   |               4.51392   | False           | False                   |
| AMD        | AMD Threadripper 3970X | CPU           |              1 | 2T              |             2 |        0.029196    |               2.81181   | False           | False                   |
| AMD        | AMD Threadripper 3970X | CPU           |              1 | 4T              |             4 |        0.0143661   |               1.90246   | False           | False                   |
| AMD        | AMD Threadripper 3970X | CPU           |              1 | 8T              |             8 |        0.00719886  |               1.43698   | False           | False                   |
| AMD        | AMD Threadripper 3970X | CPU           |              1 | 10T             |            10 |        0.00588833  |               1.3504    | False           | False                   |
| AMD        | AMD Threadripper 3970X | CPU           |              1 | 16T             |            16 |        0.00379272  |               1.08726   | False           | False                   |
| AMD        | AMD Threadripper 3970X | CPU           |              1 | 20T             |            20 |        0.00314141  |               0.90093   | False           | False                   |
| AMD        | AMD Threadripper 3970X | CPU           |              1 | 32T             |            32 |        0.00219826  |               0.578634  | False           | False                   |
| AMD        | AMD Threadripper 3970X | CPU           |              1 | 64T             |            64 |        0.00221127  |               0.590266  | False           | False                   |
| INTEL      | Intel Core i9-7900X    | CPU           |              1 | 1T              |             1 |        0.0365684   |               2.17903   | False           | False                   |
| INTEL      | Intel Core i9-7900X    | CPU           |              1 | 2T              |             2 |        0.0189302   |               1.49059   | False           | False                   |
| INTEL      | Intel Core i9-7900X    | CPU           |              1 | 4T              |             4 |        0.00986395  |               1.03561   | False           | False                   |
| INTEL      | Intel Core i9-7900X    | CPU           |              1 | 8T              |             8 |        0.00549681  |               0.813031  | False           | False                   |
| INTEL      | Intel Core i9-7900X    | CPU           |              1 | 10T             |            10 |        0.00455617  |               0.734954  | False           | False                   |
| INTEL      | Intel Core i9-7900X    | CPU           |              1 | 16T             |            16 |        0.00510194  |               0.799798  | False           | False                   |
| INTEL      | Intel Core i9-7900X    | CPU           |              1 | 20T             |            20 |        0.00453487  |               0.718269  | False           | False                   |
| 3090       | RTX 3090               | GPU           |              1 | gpu_resident    |            -1 |        0.000311445 |               0.106577  | True            | True                    |
| 5060ti     | RTX 5060 Ti            | GPU           |              1 | gpu_resident    |            -1 |        0.000716653 |               0.11989   | False           | False                   |
| AMD        | AMD Threadripper 3970X | CPU           |              2 | 1T              |             1 |        0.0337963   |               2.52585   | False           | False                   |
| AMD        | AMD Threadripper 3970X | CPU           |              2 | 2T              |             2 |        0.0164453   |               1.51066   | False           | False                   |
| AMD        | AMD Threadripper 3970X | CPU           |              2 | 4T              |             4 |        0.00750685  |               0.960303  | False           | False                   |
| AMD        | AMD Threadripper 3970X | CPU           |              2 | 8T              |             8 |        0.00382914  |               0.727696  | False           | False                   |
| AMD        | AMD Threadripper 3970X | CPU           |              2 | 10T             |            10 |        0.00310999  |               0.681705  | False           | False                   |
| AMD        | AMD Threadripper 3970X | CPU           |              2 | 16T             |            16 |        0.00199946  |               0.572612  | False           | False                   |
| AMD        | AMD Threadripper 3970X | CPU           |              2 | 20T             |            20 |        0.00164445  |               0.471693  | False           | False                   |
| AMD        | AMD Threadripper 3970X | CPU           |              2 | 32T             |            32 |        0.00115798  |               0.315732  | False           | False                   |
| AMD        | AMD Threadripper 3970X | CPU           |              2 | 64T             |            64 |        0.00119053  |               0.302222  | False           | False                   |
| INTEL      | Intel Core i9-7900X    | CPU           |              2 | 1T              |             1 |        0.0195395   |               1.17764   | False           | False                   |
| INTEL      | Intel Core i9-7900X    | CPU           |              2 | 2T              |             2 |        0.0101917   |               0.814011  | False           | False                   |
| INTEL      | Intel Core i9-7900X    | CPU           |              2 | 4T              |             4 |        0.0053187   |               0.562642  | False           | False                   |
| INTEL      | Intel Core i9-7900X    | CPU           |              2 | 8T              |             8 |        0.0028661   |               0.425898  | False           | False                   |
| INTEL      | Intel Core i9-7900X    | CPU           |              2 | 10T             |            10 |        0.00236744  |               0.393457  | False           | False                   |
| INTEL      | Intel Core i9-7900X    | CPU           |              2 | 16T             |            16 |        0.00273006  |               0.420182  | False           | False                   |
| INTEL      | Intel Core i9-7900X    | CPU           |              2 | 20T             |            20 |        0.00236109  |               0.379173  | False           | False                   |
| 3090       | RTX 3090               | GPU           |              2 | gpu_resident    |            -1 |        0.000289307 |               0.0979852 | True            | True                    |
| 5060ti     | RTX 5060 Ti            | GPU           |              2 | gpu_resident    |            -1 |        0.000582856 |               0.0924464 | True            | True                    |
| AMD        | AMD Threadripper 3970X | CPU           |              3 | 1T              |             1 |        0.0310758   |               2.3729    | False           | False                   |
| AMD        | AMD Threadripper 3970X | CPU           |              3 | 2T              |             2 |        0.0149275   |               1.42711   | False           | False                   |
| AMD        | AMD Threadripper 3970X | CPU           |              3 | 4T              |             4 |        0.00741997  |               0.971781  | False           | False                   |
| AMD        | AMD Threadripper 3970X | CPU           |              3 | 8T              |             8 |        0.00377273  |               0.739041  | False           | False                   |
| AMD        | AMD Threadripper 3970X | CPU           |              3 | 10T             |            10 |        0.00310421  |               0.696709  | False           | False                   |
| AMD        | AMD Threadripper 3970X | CPU           |              3 | 16T             |            16 |        0.00197965  |               0.567968  | False           | False                   |
| AMD        | AMD Threadripper 3970X | CPU           |              3 | 20T             |            20 |        0.00166423  |               0.477627  | False           | False                   |
| AMD        | AMD Threadripper 3970X | CPU           |              3 | 32T             |            32 |        0.00115548  |               0.309102  | False           | False                   |
| AMD        | AMD Threadripper 3970X | CPU           |              3 | 64T             |            64 |        0.00118757  |               0.299612  | False           | False                   |
| INTEL      | Intel Core i9-7900X    | CPU           |              3 | 1T              |             1 |        0.0188872   |               1.13113   | False           | False                   |
| INTEL      | Intel Core i9-7900X    | CPU           |              3 | 2T              |             2 |        0.0091348   |               0.747542  | False           | False                   |
| INTEL      | Intel Core i9-7900X    | CPU           |              3 | 4T              |             4 |        0.00484339  |               0.517818  | False           | False                   |
| INTEL      | Intel Core i9-7900X    | CPU           |              3 | 8T              |             8 |        0.00262357  |               0.391517  | False           | False                   |
| INTEL      | Intel Core i9-7900X    | CPU           |              3 | 10T             |            10 |        0.00224244  |               0.358497  | False           | False                   |
| INTEL      | Intel Core i9-7900X    | CPU           |              3 | 16T             |            16 |        0.00256255  |               0.410172  | False           | False                   |
| INTEL      | Intel Core i9-7900X    | CPU           |              3 | 20T             |            20 |        0.00224112  |               0.39074   | False           | False                   |
| 3090       | RTX 3090               | GPU           |              3 | gpu_resident    |            -1 |        0.000238035 |               0.0784748 | True            | True                    |
| 5060ti     | RTX 5060 Ti            | GPU           |              3 | gpu_resident    |            -1 |        0.000473118 |               0.08427   | False           | False                   |
| AMD        | AMD Threadripper 3970X | CPU           |              4 | 1T              |             1 |        0.0304818   |               2.32104   | False           | False                   |
| AMD        | AMD Threadripper 3970X | CPU           |              4 | 2T              |             2 |        0.0151544   |               1.42706   | False           | False                   |
| AMD        | AMD Threadripper 3970X | CPU           |              4 | 4T              |             4 |        0.00759734  |               0.967936  | False           | False                   |
| AMD        | AMD Threadripper 3970X | CPU           |              4 | 8T              |             8 |        0.00386474  |               0.729572  | False           | False                   |
| AMD        | AMD Threadripper 3970X | CPU           |              4 | 10T             |            10 |        0.00313084  |               0.680624  | False           | False                   |
| AMD        | AMD Threadripper 3970X | CPU           |              4 | 16T             |            16 |        0.0020041   |               0.574995  | False           | False                   |
| AMD        | AMD Threadripper 3970X | CPU           |              4 | 20T             |            20 |        0.00168707  |               0.483476  | False           | False                   |
| AMD        | AMD Threadripper 3970X | CPU           |              4 | 32T             |            32 |        0.00116071  |               0.31738   | False           | False                   |
| AMD        | AMD Threadripper 3970X | CPU           |              4 | 64T             |            64 |        0.00119659  |               0.290327  | False           | False                   |
| INTEL      | Intel Core i9-7900X    | CPU           |              4 | 1T              |             1 |        0.0228942   |               1.48171   | False           | False                   |
| INTEL      | Intel Core i9-7900X    | CPU           |              4 | 2T              |             2 |        0.0114243   |               0.89414   | False           | False                   |
| INTEL      | Intel Core i9-7900X    | CPU           |              4 | 4T              |             4 |        0.0056336   |               0.5916    | False           | False                   |
| INTEL      | Intel Core i9-7900X    | CPU           |              4 | 8T              |             8 |        0.00307133  |               0.45161   | False           | False                   |
| INTEL      | Intel Core i9-7900X    | CPU           |              4 | 10T             |            10 |        0.00257606  |               0.410718  | False           | False                   |
| INTEL      | Intel Core i9-7900X    | CPU           |              4 | 16T             |            16 |        0.00260604  |               0.414687  | False           | False                   |
| INTEL      | Intel Core i9-7900X    | CPU           |              4 | 20T             |            20 |        0.00249149  |               0.430116  | False           | False                   |
| 3090       | RTX 3090               | GPU           |              4 | gpu_resident    |            -1 |        0.000306245 |               0.100129  | True            | True                    |
| 5060ti     | RTX 5060 Ti            | GPU           |              4 | gpu_resident    |            -1 |        0.000484615 |               0.0861507 | True            | True                    |
| AMD        | AMD Threadripper 3970X | CPU           |              5 | 1T              |             1 |        0.0935558   |               6.78406   | False           | False                   |
| AMD        | AMD Threadripper 3970X | CPU           |              5 | 2T              |             2 |        0.0494659   |               4.13988   | False           | False                   |
| AMD        | AMD Threadripper 3970X | CPU           |              5 | 4T              |             4 |        0.0257073   |               2.71387   | False           | False                   |
| AMD        | AMD Threadripper 3970X | CPU           |              5 | 8T              |             8 |        0.0115913   |               1.79051   | False           | False                   |
| AMD        | AMD Threadripper 3970X | CPU           |              5 | 10T             |            10 |        0.0098475   |               1.71953   | False           | False                   |
| AMD        | AMD Threadripper 3970X | CPU           |              5 | 16T             |            16 |        0.00596149  |               1.40999   | False           | False                   |
| AMD        | AMD Threadripper 3970X | CPU           |              5 | 20T             |            20 |        0.00517173  |               1.36484   | False           | False                   |
| AMD        | AMD Threadripper 3970X | CPU           |              5 | 32T             |            32 |        0.0030901   |               0.886894  | False           | False                   |
| AMD        | AMD Threadripper 3970X | CPU           |              5 | 64T             |            64 |        0.00287995  |               0.74278   | False           | False                   |
| INTEL      | Intel Core i9-7900X    | CPU           |              5 | 1T              |             1 |        0.045671    |               2.75299   | False           | False                   |
| INTEL      | Intel Core i9-7900X    | CPU           |              5 | 2T              |             2 |        0.022737    |               1.90375   | False           | False                   |
| INTEL      | Intel Core i9-7900X    | CPU           |              5 | 4T              |             4 |        0.0121015   |               1.29708   | False           | False                   |
| INTEL      | Intel Core i9-7900X    | CPU           |              5 | 8T              |             8 |        0.00672689  |               1.01336   | False           | False                   |
| INTEL      | Intel Core i9-7900X    | CPU           |              5 | 10T             |            10 |        0.00631741  |               1.01739   | False           | False                   |
| INTEL      | Intel Core i9-7900X    | CPU           |              5 | 16T             |            16 |        0.00688921  |               1.07734   | False           | False                   |
| INTEL      | Intel Core i9-7900X    | CPU           |              5 | 20T             |            20 |        0.00646358  |               1.04985   | False           | False                   |
| 3090       | RTX 3090               | GPU           |              5 | gpu_resident    |            -1 |        0.000473308 |               0.158769  | True            | True                    |
| 5060ti     | RTX 5060 Ti            | GPU           |              5 | gpu_resident    |            -1 |        0.00104877  |               0.186239  | False           | False                   |
| AMD        | AMD Threadripper 3970X | CPU           |              6 | 1T              |             1 |        0.121109    |               9.19017   | False           | False                   |
| AMD        | AMD Threadripper 3970X | CPU           |              6 | 2T              |             2 |        0.060888    |               5.50774   | False           | False                   |
| AMD        | AMD Threadripper 3970X | CPU           |              6 | 4T              |             4 |        0.0307384   |               3.6725    | False           | False                   |
| AMD        | AMD Threadripper 3970X | CPU           |              6 | 8T              |             8 |        0.0158111   |               2.75228   | False           | False                   |
| AMD        | AMD Threadripper 3970X | CPU           |              6 | 10T             |            10 |        0.0127148   |               2.54486   | False           | False                   |
| AMD        | AMD Threadripper 3970X | CPU           |              6 | 16T             |            16 |        0.00860879  |               2.24913   | False           | False                   |
| AMD        | AMD Threadripper 3970X | CPU           |              6 | 20T             |            20 |        0.00718397  |               2.06092   | False           | False                   |
| AMD        | AMD Threadripper 3970X | CPU           |              6 | 32T             |            32 |        0.00575229  |               1.5777    | False           | False                   |
| AMD        | AMD Threadripper 3970X | CPU           |              6 | 64T             |            64 |        0.00581804  |               1.40561   | False           | False                   |
| INTEL      | Intel Core i9-7900X    | CPU           |              6 | 1T              |             1 |        0.0712389   |               4.30706   | False           | False                   |

## Measurement-domain caveat

CPU energy is package RAPL plus DRAM RAPL when available; GPU energy is NVML board energy. The comparison is therefore a comparison of the study's measured device domains, not a whole-system AC-wall comparison. Runtime and energy are kept as separate primitive objectives; EDP is reported only as a composite.

## CONV2D-specific interpretation

The six shapes differ in geometry and operational intensity. cuDNN and oneDNN are allowed to select different legal algorithms or primitives per shape and platform. The comparison therefore evaluates the best observed platform implementation under the frozen mathematical CONV2D semantics, not one identical low-level algorithm.

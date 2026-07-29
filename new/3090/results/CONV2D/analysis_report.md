# CONV2D platform analysis — RTX 3090

## Campaign

- Selected campaign: `20260729_093014`
- Raw measurements: 300
- Session medians: 30
- Configurations summarized: 6
- Energy domain: GPU board NVML TotalEnergyConsumption

## Tie-aware leaders (2% practical tolerance)

|   problem_size | objective         | configuration   |   num_threads |           value | exact_winner   | within_2pct   |
|---------------:|:------------------|:----------------|--------------:|----------------:|:---------------|:--------------|
|              1 | runtime           | gpu_resident    |            -1 |     0.000311445 | True           | True          |
|              1 | energy            | gpu_resident    |            -1 |     0.106577    | True           | True          |
|              1 | edp               | gpu_resident    |            -1 |     3.31803e-05 | True           | True          |
|              1 | throughput        | gpu_resident    |            -1 | 23756.2         | True           | True          |
|              1 | energy_efficiency | gpu_resident    |            -1 |    69.4217      | True           | True          |
|              2 | runtime           | gpu_resident    |            -1 |     0.000289307 | True           | True          |
|              2 | energy            | gpu_resident    |            -1 |     0.0979852   | True           | True          |
|              2 | edp               | gpu_resident    |            -1 |     2.83465e-05 | True           | True          |
|              2 | throughput        | gpu_resident    |            -1 | 12787           | True           | True          |
|              2 | energy_efficiency | gpu_resident    |            -1 |    37.7545      | True           | True          |
|              3 | runtime           | gpu_resident    |            -1 |     0.000238035 | True           | True          |
|              3 | energy            | gpu_resident    |            -1 |     0.0784748   | True           | True          |
|              3 | edp               | gpu_resident    |            -1 |     1.86526e-05 | True           | True          |
|              3 | throughput        | gpu_resident    |            -1 | 15541.3         | True           | True          |
|              3 | energy_efficiency | gpu_resident    |            -1 |    47.1409      | True           | True          |
|              4 | runtime           | gpu_resident    |            -1 |     0.000306245 | True           | True          |
|              4 | energy            | gpu_resident    |            -1 |     0.100129    | True           | True          |
|              4 | edp               | gpu_resident    |            -1 |     3.06734e-05 | True           | True          |
|              4 | throughput        | gpu_resident    |            -1 | 12079.8         | True           | True          |
|              4 | energy_efficiency | gpu_resident    |            -1 |    36.946       | True           | True          |
|              5 | runtime           | gpu_resident    |            -1 |     0.000473308 | True           | True          |
|              5 | energy            | gpu_resident    |            -1 |     0.158769    | True           | True          |
|              5 | edp               | gpu_resident    |            -1 |     7.51685e-05 | True           | True          |
|              5 | throughput        | gpu_resident    |            -1 | 15957.7         | True           | True          |
|              5 | energy_efficiency | gpu_resident    |            -1 |    47.5715      | True           | True          |
|              6 | runtime           | gpu_resident    |            -1 |     0.000747416 | True           | True          |
|              6 | energy            | gpu_resident    |            -1 |     0.252665    | True           | True          |
|              6 | edp               | gpu_resident    |            -1 |     0.000188859 | True           | True          |
|              6 | throughput        | gpu_resident    |            -1 | 17598.4         | True           | True          |
|              6 | energy_efficiency | gpu_resident    |            -1 |    52.0585      | True           | True          |

## Strict and practical Pareto status

|   problem_size | configuration   |   num_threads |   runtime_per_op_s |   total_energy_per_op_j | strict_pareto   | practical_pareto_2pct   |   platform |   session_count |   runtime_per_op_s_robust_cv_pct |   total_energy_per_op_j_robust_cv_pct |   edp_total_j_s_robust_cv_pct |
|---------------:|:----------------|--------------:|-------------------:|------------------------:|:----------------|:------------------------|-----------:|----------------:|---------------------------------:|--------------------------------------:|------------------------------:|
|              1 | gpu_resident    |            -1 |        0.000311445 |               0.106577  | True            | True                    |       3090 |               5 |                         0.475457 |                             0.747018  |                      1.28177  |
|              2 | gpu_resident    |            -1 |        0.000289307 |               0.0979852 | True            | True                    |       3090 |               5 |                         0.347028 |                             0.513935  |                      0.826625 |
|              3 | gpu_resident    |            -1 |        0.000238035 |               0.0784748 | True            | True                    |       3090 |               5 |                         0.461048 |                             0.457753  |                      0.636723 |
|              4 | gpu_resident    |            -1 |        0.000306245 |               0.100129  | True            | True                    |       3090 |               5 |                         0.122606 |                             0.0782668 |                      0.20859  |
|              5 | gpu_resident    |            -1 |        0.000473308 |               0.158769  | True            | True                    |       3090 |               5 |                         0.121597 |                             0.217203  |                      0.319396 |
|              6 | gpu_resident    |            -1 |        0.000747416 |               0.252665  | True            | True                    |       3090 |               5 |                         0.497885 |                             1.61406   |                      3.89075  |

## Interpretation guardrails

- Ten repetitions are technical repetitions; the five session medians are the scientific units.
- Runtime, total measured energy and EDP are reported separately.
- cuDNN and oneDNN may select different legal algorithms or primitives per shape and platform; `problem_spec` and `implementation` retain that provenance.
- Cross-platform energy comparison must retain the meter-domain caveat: CPU package/DRAM RAPL versus GPU board NVML.

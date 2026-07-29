# CONV2D platform analysis — RTX 5060 Ti

## Campaign

- Selected campaign: `20260729_100925`
- Raw measurements: 300
- Session medians: 30
- Configurations summarized: 6
- Energy domain: GPU board NVML TotalEnergyConsumption

## Tie-aware leaders (2% practical tolerance)

|   problem_size | objective         | configuration   |   num_threads |           value | exact_winner   | within_2pct   |
|---------------:|:------------------|:----------------|--------------:|----------------:|:---------------|:--------------|
|              1 | runtime           | gpu_resident    |            -1 |     0.000716653 | True           | True          |
|              1 | energy            | gpu_resident    |            -1 |     0.11989     | True           | True          |
|              1 | edp               | gpu_resident    |            -1 |     8.59219e-05 | True           | True          |
|              1 | throughput        | gpu_resident    |            -1 | 10324           | True           | True          |
|              1 | energy_efficiency | gpu_resident    |            -1 |    61.7128      | True           | True          |
|              2 | runtime           | gpu_resident    |            -1 |     0.000582856 | True           | True          |
|              2 | energy            | gpu_resident    |            -1 |     0.0924464   | True           | True          |
|              2 | edp               | gpu_resident    |            -1 |     5.38802e-05 | True           | True          |
|              2 | throughput        | gpu_resident    |            -1 |  6346.98        | True           | True          |
|              2 | energy_efficiency | gpu_resident    |            -1 |    40.0165      | True           | True          |
|              3 | runtime           | gpu_resident    |            -1 |     0.000473118 | True           | True          |
|              3 | energy            | gpu_resident    |            -1 |     0.08427     | True           | True          |
|              3 | edp               | gpu_resident    |            -1 |     3.9832e-05  | True           | True          |
|              3 | throughput        | gpu_resident    |            -1 |  7819.15        | True           | True          |
|              3 | energy_efficiency | gpu_resident    |            -1 |    43.8991      | True           | True          |
|              4 | runtime           | gpu_resident    |            -1 |     0.000484615 | True           | True          |
|              4 | energy            | gpu_resident    |            -1 |     0.0861507   | True           | True          |
|              4 | edp               | gpu_resident    |            -1 |     4.16208e-05 | True           | True          |
|              4 | throughput        | gpu_resident    |            -1 |  7633.63        | True           | True          |
|              4 | energy_efficiency | gpu_resident    |            -1 |    42.9408      | True           | True          |
|              5 | runtime           | gpu_resident    |            -1 |     0.00104877  | True           | True          |
|              5 | energy            | gpu_resident    |            -1 |     0.186239    | True           | True          |
|              5 | edp               | gpu_resident    |            -1 |     0.000194951 | True           | True          |
|              5 | throughput        | gpu_resident    |            -1 |  7201.7         | True           | True          |
|              5 | energy_efficiency | gpu_resident    |            -1 |    40.6278      | True           | True          |
|              6 | runtime           | gpu_resident    |            -1 |     0.00121779  | True           | True          |
|              6 | energy            | gpu_resident    |            -1 |     0.217972    | True           | True          |
|              6 | edp               | gpu_resident    |            -1 |     0.00026581  | True           | True          |
|              6 | throughput        | gpu_resident    |            -1 | 10801           | True           | True          |
|              6 | energy_efficiency | gpu_resident    |            -1 |    60.3442      | True           | True          |

## Strict and practical Pareto status

|   problem_size | configuration   |   num_threads |   runtime_per_op_s |   total_energy_per_op_j | strict_pareto   | practical_pareto_2pct   | platform   |   session_count |   runtime_per_op_s_robust_cv_pct |   total_energy_per_op_j_robust_cv_pct |   edp_total_j_s_robust_cv_pct |
|---------------:|:----------------|--------------:|-------------------:|------------------------:|:----------------|:------------------------|:-----------|----------------:|---------------------------------:|--------------------------------------:|------------------------------:|
|              1 | gpu_resident    |            -1 |        0.000716653 |               0.11989   | True            | True                    | 5060ti     |               5 |                        0.198479  |                              2.19718  |                     2.80112   |
|              2 | gpu_resident    |            -1 |        0.000582856 |               0.0924464 | True            | True                    | 5060ti     |               5 |                        0.0102153 |                              0.148562 |                     0.138739  |
|              3 | gpu_resident    |            -1 |        0.000473118 |               0.08427   | True            | True                    | 5060ti     |               5 |                        0.0848131 |                              0.266947 |                     1.09312   |
|              4 | gpu_resident    |            -1 |        0.000484615 |               0.0861507 | True            | True                    | 5060ti     |               5 |                        0.0695507 |                              0.632323 |                     0.773084  |
|              5 | gpu_resident    |            -1 |        0.00104877  |               0.186239  | True            | True                    | 5060ti     |               5 |                        0.158721  |                              0.651797 |                     1.64068   |
|              6 | gpu_resident    |            -1 |        0.00121779  |               0.217972  | True            | True                    | 5060ti     |               5 |                        0.109299  |                              0.172895 |                     0.0437415 |

## Interpretation guardrails

- Ten repetitions are technical repetitions; the five session medians are the scientific units.
- Runtime, total measured energy and EDP are reported separately.
- cuDNN and oneDNN may select different legal algorithms or primitives per shape and platform; `problem_spec` and `implementation` retain that provenance.
- Cross-platform energy comparison must retain the meter-domain caveat: CPU package/DRAM RAPL versus GPU board NVML.

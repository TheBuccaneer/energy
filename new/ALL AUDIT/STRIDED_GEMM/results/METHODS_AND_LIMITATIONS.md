# Combined STRIDED_GEMM methods and limitations

## Statistical unit

Ten repetitions inside a session quantify technical noise. They are first
collapsed to one median per session and configuration. All reported uncertainty
therefore uses five session medians.

## Native-best policy views

The analysis emits five policy views for convenience: energy-optimal,
runtime-optimal, EDP-optimal, throughput-optimal, and GFLOP/J-optimal. At fixed
`N`, however, throughput is the inverse normalization of runtime and GFLOP/J is
the inverse normalization of energy. These pairs are not independent criteria.
The implementation does not use one configuration's energy together with
another configuration's runtime in a single operating point.

## Practical equivalence

A 2% tolerance prevents tiny numerical differences from becoming categorical
winner claims. For the final all-platform table, a clear winner must also be
clearly favored against every alternative by the bootstrapped pairwise ratio
interval outside the practical-equivalence band. Otherwise the result is marked
`tie_or_uncertain`. Marginal-CI separation remains available as a diagnostic but
is not the final classification rule.

## Post-selection limitation

Native-best CPU configurations are selected and summarized on the same five
sessions. Ratio intervals describe observed robustness but are not confirmatory
post-selection inference. A future preregistered confirmation could fix thread
counts using the current campaign and rerun independent sessions.

## Energy domains

CPU package RAPL and GPU NVML board energy are not identical domains. GPU values
include device-memory energy; CPU package values exclude system DRAM when a
separate comparable domain is unavailable. The study therefore compares
measured device domains, not whole-system energy.

## GPU execution scope

GPU data are `gpu_resident`: allocations, initialization and PCIe transfers lie
outside the interval. Placement claims involving short jobs or repeated data
movement need a separate `gpu_e2e` sensitivity analysis.

## EDP across sizes

Absolute EDP is not interpreted across different N because each size represents
a different amount of work. EDP is used only for same-size device/configuration
comparisons.

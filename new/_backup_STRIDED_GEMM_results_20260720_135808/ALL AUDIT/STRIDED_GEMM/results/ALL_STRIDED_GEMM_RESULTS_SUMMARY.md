# All-platform STRIDED_GEMM results summary

We compared FP32 STRIDED_GEMM across two CPUs and two GPUs using 8,100 validated
measurements. Each platform was measured in five sessions with ten technical
repetitions per configuration. We summarize each configuration by the median of
its five session medians and report 95% bootstrap intervals. CPU energy denotes
package-only RAPL energy; GPU energy denotes NVML board energy. GPU measurements
use resident execution and exclude PCIe transfers.

The analysis distinguishes configuration selection from device placement. For
CPUs, all measured thread counts remain available in the configuration-level
Pareto analysis. For platform-level comparisons, energy-, runtime-, and EDP-best
configurations are selected separately, with a 2% practical-equivalence tolerance
and ratio-CI-aware leader sets. Throughput and GFLOP/J are reported as normalized
inverse views of runtime and energy, not as independent evidence. The resulting
tie-aware device leaders are:

|     N | Energy leaders        | Runtime leaders   | EDP leaders   | Throughput leaders   | Efficiency leaders    |
|------:|:----------------------|:------------------|:--------------|:---------------------|:----------------------|
|    64 | Intel CPU             | Intel CPU         | Intel CPU     | Intel CPU            | Intel CPU             |
|   128 | RTX 5060 Ti           | RTX 3090          | RTX 5060 Ti   | RTX 3090             | RTX 5060 Ti           |
|   256 | RTX 5060 Ti           | RTX 3090          | RTX 3090      | RTX 3090             | RTX 5060 Ti           |
|   512 | RTX 5060 Ti, RTX 3090 | RTX 3090          | RTX 3090      | RTX 3090             | RTX 5060 Ti, RTX 3090 |
|  1024 | RTX 3090              | RTX 3090          | RTX 3090      | RTX 3090             | RTX 3090              |
|  2048 | RTX 3090              | RTX 3090          | RTX 3090      | RTX 3090             | RTX 3090              |
|  4096 | RTX 3090              | RTX 3090          | RTX 3090      | RTX 3090             | RTX 3090              |
|  8192 | RTX 3090              | RTX 3090          | RTX 3090      | RTX 3090             | RTX 3090              |
| 16384 | RTX 3090              | RTX 3090          | RTX 3090      | RTX 3090             | RTX 3090              |

The direct best-CPU-versus-best-GPU comparison is:

|     N | Energy best CPU/GPU   |   CPU/GPU energy ratio | Energy classification   | Runtime best CPU/GPU   |   CPU/GPU runtime ratio | Runtime classification   |   EDP ratio |
|------:|:----------------------|-----------------------:|:------------------------|:-----------------------|------------------------:|:-------------------------|------------:|
|    64 | INTEL/5060ti          |                 0.3985 | clear_CPU               | INTEL/3090             |                  0.4702 | clear_CPU                |     0.09964 |
|   128 | INTEL/5060ti          |                 2.785  | clear_GPU               | INTEL/3090             |                  2.273  | clear_GPU                |     3.503   |
|   256 | INTEL/5060ti          |                 4.419  | clear_GPU               | INTEL/3090             |                  5.392  | clear_GPU                |    13.83    |
|   512 | INTEL/5060ti          |                 5.44   | clear_GPU               | INTEL/3090             |                 11.06   | clear_GPU                |    59.94    |
|  1024 | INTEL/3090            |                 6.483  | clear_GPU               | AMD/3090               |                 13.15   | clear_GPU                |    97.36    |
|  2048 | AMD/3090              |                 7.762  | clear_GPU               | AMD/3090               |                 10.89   | clear_GPU                |    84.44    |
|  4096 | AMD/3090              |                 7.881  | clear_GPU               | AMD/3090               |                  9.769  | clear_GPU                |    82.64    |
|  8192 | AMD/3090              |                 8.497  | clear_GPU               | AMD/3090               |                 11      | clear_GPU                |    97.6     |
| 16384 | AMD/3090              |                 8.246  | clear_GPU               | AMD/3090               |                 10.43   | clear_GPU                |    90.8     |

The complete results show whether each N has a shared energy/runtime leader or a
true placement trade-off. Clear device-level trade-offs occur at
128, 256.
Cases where the exact energy and runtime winners differ but leader uncertainty
remains are reported separately rather than counted as decisive conflicts.

Within-platform CPU tuning remains narrower than raw `idxmin` counts suggest:
1 CPU size/platform cases have disjoint, clear energy-
and runtime-optimal leader sets. All other apparent differences are ties or
uncertain under the combined 2% and CI rule.

The pairwise tables provide effect magnitude rather than only winner labels:
median ratios, 95% ratio intervals, probability of superiority, and Cliff's
delta. Because native-best configurations are selected from the same five
sessions used for comparison, these intervals are descriptive and are not
presented as confirmatory hypothesis tests.

Across all CPU thread settings and both GPUs, the practical Pareto analysis
identifies dominant points, energy-efficient compromises, runtime-efficient
compromises, balanced trade-offs, and dominated configurations. This provides
the appropriate empirical basis for later job-level placement rules; absolute
EDP values are never compared across different matrix sizes as though the jobs
contained equal work.

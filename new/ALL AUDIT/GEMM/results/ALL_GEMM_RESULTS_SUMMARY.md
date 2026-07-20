# All-platform GEMM results summary

We compared FP32 GEMM across two CPUs and two GPUs using 8,100 validated
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
|   512 | RTX 3090, RTX 5060 Ti | RTX 3090          | RTX 3090      | RTX 3090             | RTX 3090, RTX 5060 Ti |
|  1024 | RTX 3090              | RTX 3090          | RTX 3090      | RTX 3090             | RTX 3090              |
|  2048 | RTX 3090              | RTX 3090          | RTX 3090      | RTX 3090             | RTX 3090              |
|  4096 | RTX 3090              | RTX 3090          | RTX 3090      | RTX 3090             | RTX 3090              |
|  8192 | RTX 3090              | RTX 3090          | RTX 3090      | RTX 3090             | RTX 3090              |
| 16384 | RTX 3090              | RTX 3090          | RTX 3090      | RTX 3090             | RTX 3090              |

The direct best-CPU-versus-best-GPU comparison is:

|     N | Energy best CPU/GPU   |   CPU/GPU energy ratio | Energy classification   | Runtime best CPU/GPU   |   CPU/GPU runtime ratio | Runtime classification   |   EDP ratio |
|------:|:----------------------|-----------------------:|:------------------------|:-----------------------|------------------------:|:-------------------------|------------:|
|    64 | INTEL/5060ti          |                 0.3687 | clear_CPU               | INTEL/3090             |                  0.4322 | clear_CPU                |     0.08317 |
|   128 | INTEL/5060ti          |                 2.921  | clear_GPU               | INTEL/3090             |                  2.436  | clear_GPU                |     3.827   |
|   256 | INTEL/5060ti          |                 4.416  | clear_GPU               | INTEL/3090             |                  5.494  | clear_GPU                |    15.16    |
|   512 | INTEL/3090            |                 5.253  | clear_GPU               | INTEL/3090             |                 11.16   | clear_GPU                |    60.32    |
|  1024 | INTEL/3090            |                 6.618  | clear_GPU               | INTEL/3090             |                 14.2    | clear_GPU                |    95.58    |
|  2048 | INTEL/3090            |                 8.421  | clear_GPU               | AMD/3090               |                 11.24   | clear_GPU                |    94.81    |
|  4096 | AMD/3090              |                 8.207  | clear_GPU               | AMD/3090               |                  9.614  | clear_GPU                |    84.39    |
|  8192 | AMD/3090              |                 8.894  | clear_GPU               | AMD/3090               |                 11.14   | clear_GPU                |   104.5     |
| 16384 | AMD/3090              |                 9.207  | clear_GPU               | AMD/3090               |                 10.94   | clear_GPU                |   105.8     |

The complete results show whether each N has a shared energy/runtime leader or a
true placement trade-off. Clear device-level trade-offs occur at
128, 256.
Cases where the exact energy and runtime winners differ but leader uncertainty
remains are reported separately rather than counted as decisive conflicts.

Within-platform CPU tuning remains narrower than raw `idxmin` counts suggest:
2 CPU size/platform cases have disjoint, clear energy-
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

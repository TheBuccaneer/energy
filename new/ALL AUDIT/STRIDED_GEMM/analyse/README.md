# All-platform STRIDED_GEMM audit and comparison — v3

This pipeline combines validated STRIDED_GEMM campaigns from:

- Intel CPU
- AMD CPU
- RTX 3090
- RTX 5060 Ti

It performs individual-platform validation first, then unified statistics, all-platform placement analysis, dense-GEMM versus STRIDED_GEMM comparison, report generation, and an independent integrity audit.

## Run

```bash
cd ~/projects/energy/new
source ~/projects/energy/.venv/bin/activate
./run_strided_analysis_all.sh
```

## Stages

1. `01_preflight_all_strided.py`
   - requires all four individual analyses;
   - checks latest complete raw campaigns;
   - verifies five session medians per configuration;
   - checks GPU source identity and `ld=2N` provenance;
   - records CPU-package versus GPU-board energy-domain asymmetry.

2. `02_build_unified_stats.py`
   - normalizes all platforms to one session-level schema;
   - derives exact throughput and GFLOP/J from primitive runtime/energy axes;
   - creates session-bootstrap intervals and native policy leaders;
   - classifies within-platform runtime-energy conflicts.

3. `03_compare_all_platforms.py`
   - all six platform pairs;
   - ratio bootstrap intervals;
   - probability of superiority and Cliff's delta;
   - best CPU versus best GPU;
   - global leader sets;
   - strict and practical Pareto fronts;
   - placement and crossover tables.

4. `04_compare_dense_vs_strided.py`
   - configuration-matched comparison;
   - descriptive native-best comparison;
   - independent-session ratio intervals;
   - thread-selection changes;
   - point-estimate winner changes caused by layout;
   - order-invariant leader-set changes;
   - decisive placement changes only when both workloads have a clear leader.

5. `05_generate_reports.py`
   - combined audit report;
   - results summary;
   - Claude handoff;
   - methods and limitations.

6. `06_integrity_audit.py`
   - recomputes metric identities;
   - rechecks leader consistency;
   - rechecks pairwise classification;
   - independently recomputes Pareto classes;
   - audits dense-vs-strided ratio identities and classification rules;
   - recomputes point-estimate, leader-set, and decisive-placement flags.

## Statistical contract

- Five session medians are the primary units.
- Ten repetitions per session are technical repetitions.
- Runtime and energy are the primitive decision axes.
- Throughput and GFLOP/J are inverse presentation views, not extra votes.
- EDP is a composite.
- Practical-equivalence tolerance is ±2%.
- Native-best inference is descriptive post-selection.

## Energy contract

- CPU primary: package RAPL (`device_energy_j`).
- Intel package+DRAM: optional within-platform sensitivity only.
- AMD DRAM RAPL may be unavailable and is not required.
- GPU primary: NVML board energy.
- CPU/GPU energy comparisons therefore use documented but asymmetric device domains.

## Layout contract

- Logical operation: N×N GEMM.
- Leading dimension: `ld=2N`.
- FLOPs: `2N³`.
- Logical bytes: `12N²`.
- Allocated A+B+C footprint: `24N²`.
- No claim of measured physical memory traffic is made from either logical bytes or footprint.

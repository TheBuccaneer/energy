# All-platform GEMM audit and comparison — v2

This pipeline combines the already validated GEMM analyses for:

- Intel CPU
- AMD CPU
- NVIDIA GeForce RTX 3090
- NVIDIA GeForce RTX 5060 Ti

It must be installed under:

```text
~/projects/energy/new/ALL AUDIT/GEMM/analyse/
```

It reads each platform's `results/GEMM/` directory and verifies that the results
correspond to the latest complete five-session campaign in `runs/GEMM/`.

## Install

```bash
cd ~/projects/energy/new
unzip -o ~/Downloads/all_gemm_analysis_v2.zip

source ~/projects/energy/.venv/bin/activate
pip install -r "ALL AUDIT/GEMM/analyse/requirements.txt"
```

## Run

```bash
cd ~/projects/energy/new/ALL\ AUDIT/GEMM/analyse
./run_all.sh
```

Outputs are written to:

```text
~/projects/energy/new/ALL AUDIT/GEMM/results/
```

## Pipeline stages

1. `01_preflight_all_gemm.py`
   - checks all required individual outputs;
   - rejects any hard validation failure;
   - confirms that result manifests match the latest complete raw campaigns;
   - checks common size/session coverage and GPU source provenance;
   - records energy-domain and post-selection limitations.

2. `02_build_unified_stats.py`
   - normalizes the four platforms into one schema;
   - uses five session medians as the statistical units;
   - derives throughput as `2*N^3 / e2e_runtime`;
   - derives GFLOP/J as `2*N^3 / measured_energy`;
   - computes medians, exact bootstrap intervals and CVs;
   - selects tie-aware energy/runtime/EDP policies;
   - labels GPUs as `single_configuration` instead of pretending that a GPU
     configuration optimization was performed.

3. `03_compare_all_platforms.py`
   - compares every platform pair;
   - reports median ratios, bootstrap ratio intervals, probability of superiority
     and Cliff's delta;
   - uses the pairwise ratio-CI rule for the final all-platform leader table;
   - produces best-CPU-vs-best-GPU placement tables;
   - computes strict and practical 2% Pareto fronts over all CPU threads and GPUs;
   - classifies dominant, compromise and dominated configurations;
   - detects size-dependent crossover-state changes;
   - produces all figures.

4. `04_generate_reports.py`
   - creates the combined audit report;
   - creates a paper-ready Results summary;
   - creates a detailed Claude handoff;
   - documents methods and limitations.

5. `05_integrity_audit.py`
   - independently recomputes metric identities;
   - verifies leader-table/pairwise consistency;
   - recomputes Pareto flags and trade-off classes;
   - checks row counts, session keys and pairwise effect ranges;
   - creates stability and selected-leader diagnostics;
   - fails the pipeline if a generated result is internally inconsistent.

## Statistical rules

- Ten within-session repetitions are technical repetitions, not independent n=50.
- Primary statistical unit: five session medians.
- Practical-equivalence tolerance: 2%.
- Final all-platform clear leaders require ratio-CI support against every alternative.
- Native-best ratio intervals are descriptive post-selection analyses; no
  confirmatory p-values are claimed.
- Positive Cliff's delta means numerically larger A values. This favors A only
  for higher-is-better metrics.

## Metric identities

For a fixed `N`, every configuration performs the same work. Therefore:

- runtime and throughput are inverse views of one dimension;
- energy per GEMM and GFLOP/J are inverse views of one dimension;
- these four columns must not be counted as four independent findings;
- EDP is the joint energy-runtime metric.

## Energy semantics

- CPU: package-only RAPL in the current GEMM analysis.
- GPU: NVML board energy including device memory.
- GPU mode: resident; PCIe transfers excluded.

Before the full multi-workload paper is frozen, reconcile package-only GEMM with
any broader project rule that uses CPU package+DRAM as the primary CPU/GPU energy
domain. Cross-device energy results must always retain the domain limitation.

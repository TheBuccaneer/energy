# CPU GEMM scientific analysis pipeline

The same files are installed under:

- `AMD/analyse/GEMM/`
- `INTEL/analyse/GEMM/`

Each Python script discovers its platform from its path. `run_all.sh` validates and
analyzes **both** processors, then writes the cross-platform comparison to both
result folders. Thus the command behaves identically from the AMD or Intel copy.

## Dependencies

```bash
python3 -m pip install --user pandas numpy matplotlib tabulate
```

`tabulate` is needed only for Markdown tables in the reports.

## Run

From either platform directory:

```bash
cd ~/projects/energy/new/INTEL/analyse/GEMM
./run_all.sh
```

or:

```bash
cd ~/projects/energy/new/AMD/analyse/GEMM
./run_all.sh
```

The latest complete five-session campaign is selected automatically for each CPU.
For a targeted diagnostic of one older campaign, invoke the individual local
script with `--campaign`, for example:

```bash
python3 01_validate_gemm.py --campaign 20260719_085511
python3 02_analyze_gemm.py --campaign 20260719_085511
```

The cross-CPU comparison intentionally uses the newest complete campaign for each
platform.

## Main outputs

Local outputs:

- `validation_report.md`
- `validation_checks.csv`
- `campaign_manifest.csv`
- `scientific_report.md`
- `configuration_summary.csv`
- `session_configuration_medians.csv`
- `session_overview.csv`
- `robust_outliers.csv`
- `threading_scaling_sanity.csv`
- `near_optimal_candidates.csv`
- `best_energy_by_size.csv`
- `best_runtime_by_size.csv`
- `best_edp_by_size.csv`
- figures under `figures/`

Cross-platform outputs:

- `cross_platform_report.md`
- `cross_common_thread_comparison.csv`
- `cross_native_best_comparison.csv`
- `cross_tradeoff_counts.csv`
- cross-platform figures under `figures/`

## Scientific choices

- The five session medians are the primary repeatability units.
- Adjacent repetitions quantify run noise but are not treated as 50 independent sessions.
- Cross-platform energy uses package-only `device_energy_j / batches`.
- Common-thread and native-best comparisons are kept separate.
- Robust MAD outlier detection is reported but does not silently delete rows.
- Large-GEMM scaling is checked automatically; catastrophic multithread regressions fail validation.
- Exact minima are labeled clear only when the median gap exceeds 2% and the 95% session-bootstrap interval is separated from all competitors.
- Strict and practical-2% Pareto frontiers are both retained.
- The scripts fail on structural/formula/checksum errors and warn on plausibility issues.

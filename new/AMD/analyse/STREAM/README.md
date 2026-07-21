# STREAM platform analysis

This directory validates and analyzes one complete five-session STREAM campaign.

## Run

```bash
cd <PLATFORM>/analyse/STREAM
./run_all.sh
```

Optional explicit campaign:

```bash
./run_all.sh --campaign YYYYMMDD_HHMMSS
```

## Outputs

Written to `<PLATFORM>/results/STREAM/`:

- validation checks and report;
- campaign manifest with SHA-256 provenance;
- session medians and configuration summaries;
- runtime, energy, EDP and logical-bandwidth leaders;
- tie-aware fastest-vs-greenest conflicts;
- Pareto and stability tables;
- figures and a scientific report.

No raw row is silently removed. Robust outliers are diagnostic only.

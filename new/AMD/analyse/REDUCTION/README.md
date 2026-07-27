# REDUCTION platform analysis

This directory validates and analyzes one complete five-session REDUCTION campaign for the enclosing platform.

## Run

```bash
cd <PLATFORM>/analyse/REDUCTION
./run_all.sh
```

Optional campaign selection:

```bash
./run_all.sh --campaign YYYYMMDD_HHMMSS
```

The pipeline requires session files named:

```text
reduction_<platform>_YYYYMMDD_HHMMSS_session1.csv
...
reduction_<platform>_YYYYMMDD_HHMMSS_session5.csv
```

Outputs are written to `<PLATFORM>/results/REDUCTION/` and include strict validation, session medians, configuration summaries, tie-aware leaders, Pareto status, stability diagnostics and figures.

The scientific unit is the session median, not the ten technical repetitions.

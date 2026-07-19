# RTX 3090 GEMM audit pipeline

Install under `3090/analyse/GEMM/`. The scripts read the newest complete official
5-session campaign from `3090/runs/GEMM/GPU/RTX_3090/`, `3090/runs/GEMM/`, or
`3090/runs/`. Quickcheck CSVs are ignored.

```bash
python3 -m pip install pandas numpy matplotlib tabulate
cd ~/projects/energy/new/3090/analyse/GEMM
./run_all.sh
```

Results are written to `3090/results/GEMM/`.

The audit checks campaign coverage, 45-column `cpu-gpu-v2` data, formulas,
checksums, RTX-3090 identity, resident execution, NVML board energy, pedantic FP32
source provenance, TF32 disablement, PCIe telemetry, clocks, temperatures,
throttle reasons, robust outliers and five-session reproducibility.

The current GPU source emits `device_energy_j,total_energy_j`, while the CPU v2
order is `total_energy_j,device_energy_j`. This is reported as a warning, not a
measurement failure, because both values are identical on GPU and the analysis
addresses columns by name.

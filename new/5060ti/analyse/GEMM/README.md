# RTX 5060 Ti GEMM audit pipeline

Install under `5060ti/analyse/GEMM/`. The scripts read the newest complete official
5-session campaign from `5060ti/runs/GEMM/GPU/RTX_5060_Ti/`, `5060ti/runs/GEMM/`, or
`5060ti/runs/`. Quickcheck CSVs are ignored.

```bash
python3 -m pip install pandas numpy matplotlib tabulate
cd ~/projects/energy/new/5060ti/analyse/GEMM
./run_all.sh
```

Results are written to `5060ti/results/GEMM/`.

The audit checks campaign coverage, 45-column `cpu-gpu-v2` data, formulas,
checksums, RTX-5060-Ti identity, resident execution, NVML board energy, pedantic FP32
source provenance, TF32 disablement, PCIe telemetry, clocks, temperatures,
throttle reasons, robust outliers and five-session reproducibility.

The current GPU source emits `device_energy_j,total_energy_j`, while the CPU v2
order is `total_energy_j,device_energy_j`. This is reported as a warning, not a
measurement failure, because both values are identical on GPU and the analysis
addresses columns by name.

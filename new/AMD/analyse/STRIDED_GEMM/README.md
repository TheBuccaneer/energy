# CPU STRIDED_GEMM individual audit

Run:

```bash
./run_all.sh
```

The validator reads the newest complete official five-session campaign and ignores quickchecks.

Primary cross-platform energy is CPU package RAPL (`device_energy_j / batches`). DRAM RAPL is optional; package+DRAM remains a sensitivity only where it exists. Session IDs are accepted when they contain the campaign timestamp and end in `_sessionN`.

Results are written to `<platform>/results/STRIDED_GEMM/`.

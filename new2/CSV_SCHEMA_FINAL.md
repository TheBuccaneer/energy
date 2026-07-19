# Final CPU/GPU Raw CSV Schema (`cpu-gpu-v2`)

All CPU workloads emit the same 45-column header. AMD uses the available Linux package-energy backend: powercap when exposed, otherwise the perf `power/energy-pkg/` event. Because this platform exposes no separate DRAM-energy event, `dram_energy_j` is written as `-1.000000` and `total_energy_j` equals `device_energy_j`.

1. `schema_version` — fixed `cpu-gpu-v2`
2. `timestamp` — ISO-8601 local time
3. `session_id` — complete-pass identifier supplied by the runner
4. `sequence_index` — measurement order within one CSV file
5. `run_id_global` — currently identical to `sequence_index` within the file
6. `repetition` — 1 through the configured repetition count
7. `workload`
8. `implementation` — concrete library/kernel implementation
9. `execution_mode` — CPU: `cpu_native`
10. `device_name`
11. `num_threads`
12. `problem_size` — GEMM dimension, element count, or Conv2D shape ID
13. `problem_spec` — explicit dimensions/shape description
14. `batches` — logical operations in the measured interval
15. `e2e_time_s`
16. `kernel_time_s` — equals CPU measured compute interval
17. `wall_time_s` — equals CPU measured compute interval
18. `device_energy_j` — AMD CPU package energy
19. `total_energy_j` — AMD: equal to package energy
20. `dram_energy_j` — AMD: `-1.000000` (unavailable)
21. `energy_per_op_j`
22. `energy_per_second_j`
23. `energy_per_flop_j`
24. `time_per_op_ms_kernel`
25. `time_per_op_ms_e2e`
26. `flops_total`
27. `gflops_per_s`
28. `logical_bytes_per_op`
29. `avg_power_w`
30. `runtime_status` — `below` (<0.75 s), `in_range` (0.75–1.25 s), or `above` (>1.25 s)
31. `pcie_gen` — empty on CPU
32. `pcie_width` — empty on CPU
33. `sm_clock_mhz` — empty on CPU
34. `clock_before_mhz` — mean snapshot across online logical CPUs before the interval
35. `clock_after_mhz` — mean snapshot across online logical CPUs after the interval
36. `mem_clock_mhz` — empty on CPU
37. `temp_c` — maximum of pre/post CPU sensor snapshots
38. `temp_before_c`
39. `temp_after_c`
40. `throttle_reasons` — empty on CPU
41. `cpu_cycles` — reserved, currently `-1`
42. `cpu_instructions` — reserved, currently `-1`
43. `cpu_ipc` — reserved, currently `-1.000000`
44. `cpu_cache_misses` — reserved, currently `-1`
45. `checksum_ok` — `t` for accepted rows

## Workload definitions

- GEMM: contiguous row-major SGEMM, `ld=N`; FLOPs/op = `2N³`; logical bytes/op = `3N²×4`.
- Strided GEMM: row-major SGEMM with `ld=2N`; padding excluded from semantic logical bytes.
- STREAM Triad: `a[i] = b[i] + 3*c[i]`; FLOPs/op = `2×elements`; logical bytes/op = `3×elements×4`.
- AXPY: `y[i] = 3*x[i] + y[i]`; FLOPs/op = `2×elements`; logical bytes/op = `3×elements×4`.
- Reduction: true one-array sum; FLOPs/op = `elements-1`; logical bytes/op = `elements×4 + 4`.
- Conv2D: oneDNN `convolution_auto`; FLOPs/op = `2×N×K×C×R×S×H'×W'`; logical bytes/op = `(input + weights + output)×4`.

## Campaign layout

Thread grid: `1,2,4,8,10,16,20,32,64`.

- GEMM, Strided GEMM, STREAM, AXPY, Reduction: 9 problem sizes × 9 thread counts = 81 configurations per complete pass.
- Conv2D: 6 shapes × 9 thread counts = 54 configurations per complete pass.
- Default: 10 repetitions per configuration × 5 complete passes = 50 rows per concrete configuration.
- Total default data rows: 22,950.
- Configuration order is deterministically shuffled within each workload/pass; every configuration is included exactly once per pass.

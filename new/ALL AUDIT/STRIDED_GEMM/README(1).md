# CPU Strided GEMM repair package

This package contains the repaired Intel and AMD `STRIDED_GEMM` sources, official five-session runners, and platform-specific quickchecks.

## Integration

Copy the `INTEL/` and `AMD/` directories into:

```text
~/projects/energy/new/
```

The package deliberately does **not** replace these existing platform files:

```text
INTEL/scripts/common/benchmark_common.hpp
AMD/scripts/common/benchmark_common.hpp
INTEL/scripts/01_enable_CPU_Intel.sh
INTEL/scripts/03_disable_CPU_Intel.sh
AMD/scripts/01_enable_CPU_AMD.sh
AMD/scripts/03_disable_CPU_AMD.sh
```

The repaired Strided GEMM programs inherit CSV, RAPL, clock, temperature, calibration, and row-writing semantics from the existing platform-specific `benchmark_common.hpp`, exactly like the repaired normal GEMM programs.

## Dependencies

On Ubuntu/Debian:

```bash
sudo apt update
sudo apt install build-essential libopenblas-dev python3 coreutils
```

A threaded OpenBLAS build is mandatory. The programs abort if `openblas_get_parallel()` reports a sequential library or if a requested thread count is not honored.

## Quickchecks

Intel:

```bash
cd ~/projects/energy/new/INTEL/scripts
./quickcheck_CPU_Intel_STRIDED_GEMM.sh
```

AMD:

```bash
cd ~/projects/energy/new/AMD/scripts
./quickcheck_CPU_AMD_STRIDED_GEMM.sh
```

The quickchecks use `N=4096`, three repetitions, and these thread grids:

```text
Intel: 1,4,10,20
AMD:   1,4,10,32,64
```

They validate compilation, OpenBLAS backend/thread control, CSV row count, checksums, positive measurements, and multithread scaling. At least one multithread configuration must reach `1.25x` speedup over one thread; a configuration at least `5x` slower than one thread is a hard failure.

## Official campaigns

Intel:

```bash
cd ~/projects/energy/new/INTEL/scripts
./02_run_CPU_Intel_STRIDED_GEMM_only.sh
```

AMD:

```bash
cd ~/projects/energy/new/AMD/scripts
./02_run_CPU_AMD_STRIDED_GEMM_only.sh
```

The official runner defaults are:

```text
5 sessions
10 repetitions per configuration
no pause between sessions
```

Expected data rows:

```text
Intel: 9 sizes x 7 thread counts x 10 reps = 630 rows/session
       3150 rows over five sessions
AMD:   9 sizes x 9 thread counts x 10 reps = 810 rows/session
       4050 rows over five sessions
```

The default is `POWER_OFF_AT_END=1`, so the machine powers off only after all five sessions validate successfully. To keep it online:

```bash
POWER_OFF_AT_END=0 ./02_run_CPU_Intel_STRIDED_GEMM_only.sh
POWER_OFF_AT_END=0 ./02_run_CPU_AMD_STRIDED_GEMM_only.sh
```

On errors, failed validation, or `Ctrl+C`, the restore script is run and automatic power-off is suppressed.

## Memory requirement

For `N=16384`, `ld=32768`, each FP32 matrix occupies 2 GiB. The three matrices therefore require approximately 6 GiB, in addition to OpenBLAS and operating-system memory.

## Output

Official CSV and log files are written to:

```text
INTEL/runs/STRIDED_GEMM/
AMD/runs/STRIDED_GEMM/
```

The in-field `problem_spec` representation is `N=<N>;ld=<2N>`. A semicolon is used deliberately so the value remains one CSV field without requiring a schema change.

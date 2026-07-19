# Reusing the analysis for the RTX 5060 Ti

## Can the current scripts be copied unchanged?

**Not completely unchanged.**

The scientific logic is reusable, but the current validator is intentionally
specific to the RTX 3090. It checks the 3090 directory, filename pattern,
device name, source path, and runner path.

For the RTX 5060 Ti, change or parameterize:

- project root: `3090` → `5060ti`;
- run pattern: `gemm_3090_...` → `gemm_5060ti_...`;
- expected device name: `NVIDIA GeForce RTX 5060 Ti`;
- result and analysis paths;
- audited source and runner paths;
- optional board-power plausibility bounds.

The following should remain identical:

- matrix sizes;
- five sessions × ten repetitions;
- `gpu_resident`;
- pedantic FP32;
- TF32 disabled;
- direct NVML energy;
- CSV formulas;
- session-median statistics;
- stability and throttle analysis.

The CUDA GEMM source should ideally be shared between both GPUs. Only the
device-specific runner, expected device identity, output paths, and enable/restore
logic should differ.

Do not silently fix the N=16384 calibration only for the 5060 Ti. Either run the
same current code on both devices, or improve the calibrator and rerun both GPUs.
Identical measurement code is more important than a one-sided protocol change.

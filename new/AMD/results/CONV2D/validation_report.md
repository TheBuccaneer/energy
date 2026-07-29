# CONV2D validation report — AMD Threadripper 3970X

## Verdict

**PASS**

- Campaign: `20260729_005653`
- Files: 5
- Raw rows: 2700
- Expected sessions: 5
- Expected repetitions per configuration: 10
- Energy domain: CPU package RAPL plus DRAM RAPL when available
- Execution mode: `cpu_native`

## Hard failures

_None._

## Warnings

_None._

## Measurement contract

- Six frozen CONV2D shapes, FP32 NCHW/OIHW, cross-correlation, no bias or activation.
- Logical FLOPs: `2*N*K*C*R*S*Hout*Wout` per convolution.
- Logical bytes: input + weights + output, each counted once.
- CPU primary measurement: RAPL package plus DRAM when available.
- GPU primary measurement: NVML board energy in `gpu_resident` mode.
- The scientific unit used downstream is the median of ten repetitions within each session.

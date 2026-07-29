# CONV2D validation report — RTX 5060 Ti

## Verdict

**PASS**

- Campaign: `20260729_100925`
- Files: 5
- Raw rows: 300
- Expected sessions: 5
- Expected repetitions per configuration: 10
- Energy domain: GPU board NVML TotalEnergyConsumption
- Execution mode: `gpu_resident`

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

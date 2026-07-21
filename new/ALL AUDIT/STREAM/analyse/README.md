# All-platform STREAM analysis and audit

This pipeline combines validated STREAM campaigns from Intel CPU, AMD CPU, RTX 3090 and RTX 5060 Ti.

## Run all individual analyses and then the combined audit

From the project root:

```bash
source ~/projects/energy/.venv/bin/activate
./run_stream_analysis_all.sh
```

## Statistical contract

- Ten repetitions per configuration/session are technical repetitions.
- The median within each session is the primary unit.
- Five session medians are used for summaries and descriptive ratio intervals.
- Runtime and energy are the primitive decision axes.
- Logical bandwidth is the inverse runtime view.
- Logical GB/J is the inverse energy view.
- EDP is a composite.
- Practical-equivalence tolerance is ±2%.

## Measurement contract

- STREAM Triad: `a = b + 3*c`, FP32.
- One operation is one complete pass over N elements.
- `2*N` logical FLOPs and `12*N` logical bytes per operation.
- Logical bytes are not measured physical traffic.
- CPU primary energy: package RAPL.
- GPU primary energy: NVML board energy.
- GPU mode: resident, excluding allocations and PCIe transfers.

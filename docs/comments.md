# Benchmark Context Notes

This file preserves the implementation context that was previously scattered through inline comments.

## Runtime goals

- Keep Colab responsive during long TPU and host-memory sweeps.
- Favor predictable progression over aggressive one-shot allocation.
- Save every meaningful signal to disk so interrupted runs still leave usable artifacts.
- Emit TensorBoard events directly from Python without adding a heavyweight training framework dependency.

## Mega stress design

- `host_ram_saturation` ramps memory in smaller `64MB` chunks instead of `256MB` chunks.
- Allocation is deliberately throttled with a short sleep so notebook output and interrupts remain responsive.
- The allocator stops before host memory reaches the container cliff, leaving explicit safety headroom.
- After host pressure is staged, the suite runs a real sparse attention kernel to measure latency under pressure instead of treating allocation alone as success.

## Ceiling sweep design

- The ceiling test walks through progressively harder shapes instead of jumping directly to worst-case shapes.
- Sparse and dense are evaluated separately so failure in one path does not hide usable information in the other.
- Each step records estimated memory footprint, latency, and throughput when successful.

## Accumulation stress design

- The accumulation path models large effective batch sizes while keeping each micro-step device-safe.
- Jitter is tracked from per-step latency so the run shows stability, not only throughput.

## Mixed pressure design

- Host residency and device execution are tested together because real TPU jobs do not run in isolation from input staging.
- The mixed test keeps a host-resident dataset while launching repeated sparse kernels and tracking latency drift.

## Telemetry design

- Console output is short and stage-oriented.
- Colab receives a live inline status view.
- Metrics are persisted in `results/live/*.jsonl` and `results/live/*_summary.json`.
- TensorBoard events are written under `results/tensorboard/<run_name>/`.

## Why the old run felt broken

- The previous implementation pushed large allocations fast enough that Colab became difficult to interact with.
- `tqdm` made the run look active, but it did not give clear headroom or safety state.
- When users saw the notebook stuck on a single fast allocation bar, interruption was the only obvious control path.

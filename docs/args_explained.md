# Benchmark Args Explained

This file explains the main `run_benchmarks.py` flags in plain terms and what they add beyond a basic benchmark loop.

## `--mega-stress`

Runs the high-pressure suite. It combines:

- host RAM saturation
- sparse vs dense OOM ceiling detection
- gradient accumulation scaling
- mixed host-plus-device pressure

Use it when you want operational confidence, not just a best-case latency number.

## `--target-gb`

Sets the host-memory pressure target for `--mega-stress`.

- On Colab TPU v5e-1, the VM typically exposes about `47GB` host RAM.
- The suite intentionally stops before the absolute limit so the notebook stays alive.
- Asking for `40GB` means "approach this level safely", not "blindly allocate exactly 40GB no matter what".

## `--roofline`

Runs a roofline analysis. This is not just another latency report.

It combines:

- theoretical FLOPs
- theoretical memory traffic
- measured latency
- achieved throughput

From those values it derives:

- arithmetic intensity: FLOPs per byte
- achieved TFLOPs
- MFU: model FLOPs utilization
- HFU: hardware FLOPs utilization
- bandwidth utilization

What it tells you:

- whether the kernel is memory-bound or compute-bound
- whether optimization effort should focus on memory traffic, fusion, or compute efficiency
- whether sparse attention is moving the workload into a better operating region than dense attention

## `--stress`

Runs the standard stress suite:

- OOM boundary sweep
- sustained throughput tracking
- numerical stability checks
- compilation overhead measurement

Use it when you want runtime stability metrics such as jitter, tail latency, and failure boundaries.

## `--quick`

Shrinks the tested shape set and iteration counts so you can validate the pipeline quickly on Colab without paying for a full sweep.

## `--no-pallas`

Disables the custom Pallas path and falls back to the pure JAX implementation.

Use it for:

- correctness comparison
- portability debugging
- measuring how much the custom kernel is actually buying you

## `--block-size`

Controls sparse tiling granularity.

Why it matters:

- too small increases scheduling and memory traffic overhead
- too large can hurt on-chip residency and lower utilization
- the best value depends on shape, kernel layout, and device

## `--sparsity`

Selects the block-mask pattern:

- `causal`
- `strided`
- `fixed`
- `random`
- `combined`

This changes both algorithmic work and memory movement, so it affects latency, throughput, and roofline position.

## `--viz-only`

Builds plots from already-saved results without rerunning the benchmarks.

## `--viz-all`

Exports the separate visualization bundles for throughput, memory, stress, scaling, and training logs when the source artifacts are present.

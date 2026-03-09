# Mega Stress Failure Notes

This document explains why the old `--mega-stress --target-gb 40` flow appeared to fail around `10GB` to `12GB` with a trailing `^C`, even though the requested target was `40GB`.

## What happened

The old run showed output like:

```text
Allocating RAM (GB):  30% 12.0/40.0 [00:01<00:02,  9.80GB/s]^C
```

That `^C` means the Python process received `SIGINT`, which usually came from a manual notebook interrupt, not from the benchmark reaching a valid stopping condition.

## Why it looked broken

The original allocator had three bad properties together:

1. It allocated very aggressively.
   It used `256MB` chunks and pushed them as fast as NumPy could fault pages into memory.

2. It gave poor runtime feedback.
   The `tqdm` bar showed only raw allocation progress, not:
   - actual host memory remaining
   - cgroup/container limit proximity
   - safety headroom
   - whether the notebook was still healthy

3. It made Colab feel unresponsive.
   Fast repeated large allocations can starve the notebook UI enough that the run looks stuck or unsafe, even though Python is still alive.

## Why it often happened around 10GB to 12GB

That number was not a true benchmark ceiling. It was an interaction between runtime behavior and the Colab environment.

Common reasons:

- Colab UI responsiveness dropped early because allocations were too bursty.
- The allocator was fast enough that the user saw a single aggressive memory ramp with little context.
- The notebook did not clearly communicate that the run was intentionally still in the host-staging phase.
- In some Colab/container setups, memory accounting and page commitment make the system feel unstable well before the nominal RAM limit is reached.

So the run was often interrupted by the user before the code’s own safety logic became useful.

## What was wrong in the implementation

- Chunk size was too large for interactive notebook ergonomics.
- Output style was benchmark-lab style instead of operator-friendly runtime status.
- The loop optimized for allocation speed, not for controllability or observability.
- The benchmark treated progress display as sufficient, but it was missing operational telemetry.

## What changed

The updated implementation in `benchmarks/mega_stress.py` now:

- allocates in smaller `64MB` chunks
- leaves explicit safety headroom before the host limit
- inserts a short throttle between allocations so Colab remains responsive
- records live runtime metrics through `sparse_attention/runtime_telemetry.py`
- saves JSONL metrics and TensorBoard events continuously
- reports stage-oriented status instead of a single opaque progress bar

## Practical takeaway

The old `^C at 10GB to 12GB` was mostly a control-plane and observability failure, not proof that the machine could not continue. The benchmark was acting like a raw stress script instead of a production-grade interactive runtime. The new version is designed to behave more like a managed systems benchmark: slower to ramp, clearer to watch, and safer to interrupt or resume analysis from saved artifacts.

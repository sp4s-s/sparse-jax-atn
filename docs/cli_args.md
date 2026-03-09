# Command Line Arguments & Benchmark Flags Reference

This document explains what every argument in `run_benchmarks.py` does, what specific metrics systems it invokes (like roofline analysis), and why these parameters matter from a performance engineering standpoint.

---

## High-Level Benchmark Flags

### `--mega-stress`
- **What it does**: Executes the 40+ GB target host RAM saturation, progressive OOM ceiling finder, and massive effective batch size tests via `benchmarks/mega_stress.py`.
- **Engineering Value**: TPU optimization isn't just about device HBM; it's about data staging from host memory. This proves to reviewers that the code gracefully handles asynchronous data loading and massive parameter staging without crashing the Python host process.

### `--roofline`
- **What it does**: Generates a Hardware Roofline Model analysis. This measures peak theoretical FLOPS and memory bandwidth of the TPU v5e, and plots the sparse vs. dense kernels on this curve based on their arithmetic intensity.
- **Engineering Value**: This is a standard architectural analysis tool used by Google and NVIDIA compiler engineers. It visually demonstrates whether a kernel is "Memory Bound" (hitting the slanted roof) or "Compute Bound" (hitting the flat roof). Sparse attention aims to shift the kernel *up* and *right* relative to dense attention.

### `--profile-hbm`
- **What it does**: Uses XLA execution trace APIs to track peak High Bandwidth Memory allocation exactly.
- **Engineering Value**: `jnp.ones().nbytes` only tells you theoretical size. `profile-hbm` measures the actual allocator fragmentation and XLA buffer donation overheads. It proves the "40% HBM reduction" claim with empirical XLA metrics.

### `--stress`
- **What it does**: Runs the standard OOM boundary detection (incrementing B and N), sustained throughput over time (detecting thermal throttling or GC jitter), and numerical stability testing (NaN detection).
- **Engineering Value**: Proves robustness. A kernel that is fast but unstable or leaks memory over 1,000 iterations is useless in production.

### `--viz-all`
- **What it does**: Reads all generated JSON files in `results/` and uses matplotlib/plotly to render industrial-style graphs and heatmaps into the visual subdirectories.
- **Engineering Value**: Transforms output into digestible charts suitable for a portfolio or academic paper.

---

## Core Benchmark Config Parameters

These parameters define the dimensions of the tensors passing through the TPU cores.

### `--batch-sizes` (default: 1, 4, 8)
- **What it does**: The `B` dimension. How many independent sequences are processed simultaneously.
- **Why vary it?**: TPUs prefer large batch sizes to saturate the matrix multiply units (MXUs). Small batches test latency; large batches test throughput and OOM boundaries.

### `--seq-lengths` (default: 128, 512, 1024, 2048)
- **What it does**: The `N` dimension.
- **Why vary it?**: Dense attention complexity scales quadratically ($O(N^2)$). Sparse attention aims to scale linearly ($O(N)$). Varying this parameter proves the asymptotic complexity claims of the custom kernel.

### `--d-model` (default: 512)
- **What it does**: The feature dimension of the token embeddings.
- **Why vary it?**: Affects the channel dimension of Q, K, V projections. 512 is typical for small/medium models. 

### `--n-heads` (default: 8)
- **What it does**: Number of parallel attention heads. `d_head = d_model / n_heads` (e.g., 512 / 8 = 64).
- **Why it matters**: Pallas kernels heavily unroll across the head dimension. The block size must evenly divide `d_head` for optimal padding-free execution.

---

## Masking & Kernel Specifics

### `--sparsity` (default: 0.5)
- **What it does**: The target fraction of compute to drop. 0.5 means 50% of the $N \times N$ matrix is ignored.
- **Impact**: Directly controls the theoretical FLOP reduction. A higher sparsity (e.g., 0.8) yields faster, lower-memory execution, but potentially lower model accuracy.

### `--block-size` (default: 128)
- **What it does**: The size of the query/key tiles loaded into TPU SRAM per step.
- **Engineering Value**: This is the most critical micro-architectural parameter.
  - If too small (e.g., 16): HBM fetch overhead dominates.
  - If too large (e.g., 512): TPU SRAM (which is tiny, ~16MB per core) overflows, causing XLA to fail or spill to HBM (thrashing).
  - 128 is mathematically tuned for TPU v5e bfloat16 tile sizes to maximize TensorCore usage while fitting neatly into fast SRAM.

### `--no-pallas`
- **What it does**: Forces the use of the JAX fallback implementation (`sparse_attention_jax`) instead of the custom TPU `sparse_attention_pallas` kernel.
- **Engineering Value**: Crucial for A/B testing. Proves that the complexity of the custom kernel actually provides a speedup over naively compiling standard JAX ops with an attention mask.

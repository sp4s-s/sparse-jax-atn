# Metrics Explanation & Measurement Methodology

Detailed explanation of every metric collected, how it's measured, expected ranges, and how to interpret results.

## Metrics Overview

| Metric | Category | How Measured | Key For |
|--------|----------|-------------|---------|
| HBM Bandwidth | Memory | Theoretical calculation | Primary optimization target |
| FLOPs | Compute | Operation counting | Computational efficiency |
| Latency | Time | Wall-clock with block_until_ready | User-facing performance |
| Throughput | Time | Tokens / second | Practical inference speed |
| Peak Memory | Memory | Device memory stats | OOM prevention |
| Sparsity Ratio | Structure | Block mask analysis | Configuration tuning |
| Speedup | Comparison | Dense / Sparse ratio | Optimization effectiveness |

---

## 1. HBM Bandwidth Reduction (Primary Metric)

### What It Is

High Bandwidth Memory (HBM) is the main memory on TPUs. Bandwidth = bytes transferred per second between HBM and the compute units. This is often the bottleneck for attention (attention is memory-bound, not compute-bound).

### How It's Measured

**Theoretical calculation** based on bytes read/written:

#### Dense Attention HBM:
```
Q read:              B × N × H × D × dtype_bytes
K read:              B × N × H × D × dtype_bytes
V read:              B × N × H × D × dtype_bytes
Attention matrix:    B × H × N × N × dtype_bytes  ← THIS IS THE BIG ONE
Output write:        B × N × H × D × dtype_bytes

Total Dense = 4 × B×N×H×D×2 + B×H×N²×2
```

#### Sparse Attention HBM:
```
Q read:              B × N × H × D × dtype_bytes  (same — read all queries)
K read:              B × active_kv_blocks × bs × H × D × dtype_bytes  (REDUCED)
V read:              B × active_kv_blocks × bs × H × D × dtype_bytes  (REDUCED)
Attention matrix:    0  ← ELIMINATED (online softmax, never materialized)
Output write:        B × N × H × D × dtype_bytes  (same)

Total Sparse = 2×B×N×H×D×2 + 2×B×active×bs×H×D×2
```

#### Reduction:
```
reduction_% = (Total_Dense - Total_Sparse) / Total_Dense × 100
```

### Expected Values

| Config | Sparsity Ratio | Expected HBM Reduction |
|--------|----------------|------------------------|
| combined, N=1024 | ~45% | ~35-40% |
| combined, N=2048 | ~50% | ~40-45% |
| combined, N=4096 | ~55% | ~45-50% |
| causal only, N=2048 | ~47% | ~25-35% |
| strided, N=2048 | ~40% | ~35-40% |

### Key Insight

The biggest saving comes from **not materializing the N×N attention matrix**. For N=2048 with 8 heads, that's `8 × 2048² × 2 = 67 MB` per batch element — completely eliminated by online softmax.

---

## 2. FLOPs (Floating Point Operations)

### How It's Measured

Counted analytically per operation:

| Operation | FLOPs (dense) | FLOPs (sparse) |
|-----------|---------------|----------------|
| Q×K^T | 2·B·H·N²·D | 2·B·H·active·bs²·D |
| Softmax | 5·B·H·N² | 5·B·H·active·bs² |
| Attn×V | 2·B·H·N²·D | 2·B·H·active·bs²·D |

Where `active` = number of active blocks, `bs` = block_size.

### Achieved TFLOPs

```
achieved_tflops = theoretical_flops / (latency_seconds) / 1e12
```

This tells you how efficiently you're utilizing the TPU's compute units.

### Expected Values

- TPU v5e peak: ~197 TFLOPs (bf16)
- Attention typically achieves 30-60% of peak
- Sparse attention FLOPs reduction: ~40-55% depending on sparsity

---

## 3. Latency

### How It's Measured

```python
# Warmup: compile and cache (not measured)
for _ in range(n_warmup):
    result = fn(inputs)
    result.block_until_ready()  # CRITICAL: ensures TPU computation finishes

# Timed iterations
for _ in range(n_iterations):
    start = time.perf_counter()
    result = fn(inputs)
    result.block_until_ready()  # Must wait for async TPU execution!
    end = time.perf_counter()
```

### Important Notes

- `block_until_ready()` is **essential** on TPU/GPU — without it, you only measure the dispatch time, not the actual computation
- First call includes XLA compilation (that's why we have warmup)
- Measured in milliseconds

### Expected Values

| Config | Dense (ms) | Sparse (ms) | Speedup |
|--------|-----------|-------------|---------|
| B=1, N=1024 | ~2-5 | ~1.5-3 | 1.3-1.7x |
| B=4, N=2048 | ~15-30 | ~8-18 | 1.5-2.0x |
| B=8, N=4096 | ~100-200 | ~50-120 | 1.5-2.5x |

---

## 4. Throughput

### How It's Measured

```
tokens_per_second = (batch_size × seq_len) / (latency_ms / 1000)
```

### Expected Values

Varies greatly by config. Higher is better.

---

## 5. Peak Memory

### How It's Measured

```python
stats = jax.local_devices()[0].memory_stats()
peak_bytes = stats.get('peak_bytes_in_use', 0)
```

Note: This measures total device memory, not just attention. Availability depends on JAX version.

---

## 6. Sparsity Ratio

### How It's Measured

```
sparsity = 1 - (active_blocks / total_blocks)
```

From the `BlockMask` object directly. Total blocks = `(N/bs)²`.

### Relation to HBM Reduction

Sparsity ratio doesn't directly equal HBM reduction because:
1. Q and output reads/writes are the same for both
2. The N×N attention matrix elimination gives a large fixed saving
3. K/V savings scale with sparsity

---

## 7. Bandwidth (GB/s)

### How It's Measured

```
bandwidth_gb_s = total_bytes / (latency_seconds) / 1e9
```

### Expected Values

- TPU v5e HBM bandwidth: ~820 GB/s
- Good utilization: >50% of peak = >410 GB/s

---

## Running All Metrics

```bash
# Everything at once
python run_benchmarks.py --quick

# Just HBM (fastest)
python run_benchmarks.py --hbm --quick

# Just FLOPs
python run_benchmarks.py --flops --quick

# Scaling analysis
python run_benchmarks.py --scaling --quick

# Save results to JSON
python run_benchmarks.py --quick --output-dir my_results
cat my_results/benchmark_results.json
```

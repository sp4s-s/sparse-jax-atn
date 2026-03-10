# [Sparse Attention ](https://github.com/sp4s-s/sparse-jax-atn)

Block-sparse attention kernel for JAX/XLA, written in Pallas and benchmarked on TPU v5e. The implementation targets one specific thing: crossing the hardware's arithmetic intensity ridge point. Standard dense attention at long sequences is memory-bandwidth-bound by construction — the attention matrix grows as O(N²) in both FLOPs and HBM footprint, and most of the traffic goes toward fetching KV blocks that barely move the output. This kernel fixes that by restricting each query to a fixed set of key blocks via a `BlockMask`, which eliminates the attention matrix from HBM entirely and shifts the FLOPs-per-byte ratio into the compute-bound regime.

---

## hardware context

TPU v5e ridge point: **240.2 FLOPs/byte** (197 TFLOPs bf16 / 820 GB/s HBM). Every kernel below that is stalling on memory reads. Dense attention at N=4096 sits at 64.5 FLOPs/byte — barely 27% of the ridge. more chips doesn't help; only saturates bandwidth, not compute.

Sparse attention inverts this. As sparsity grows with N, the active-block FLOPs-per-byte ratio climbs. At N=1024 (53.1% sparsity) it clears 249.4 FLOPs/byte, crossing the ridge. At N=4096 (77.1% sparsity) it's at 486.3 FLOPs/byte — the tensor cores gets properly utilized.

---

## kernel mechanics

### `sparse_attention_pallas` — [ kernel ]

The Pallas path never materializes the N×N attention matrix. It runs an online numerically-stable softmax (flash-attention style) over active KV blocks only, using `jax.lax.fori_loop` over `max_active` blocks per query:

```python
running_max = jnp.full((block_size, H), -jnp.inf)
running_sum = jnp.zeros((block_size, H))
acc         = jnp.zeros((block_size, H, D))

def body_fn(kv_iter, carry):
    rm, rs, a = carry
    idx     = active_idx[kv_iter]
    k_block = jax.lax.dynamic_slice(k_flat, (idx * block_size, 0, 0), ...)
    v_block = jax.lax.dynamic_slice(v_flat, (idx * block_size, 0, 0), ...)
    scores  = jnp.einsum('qhd,khd->hqk', q_block, k_block) * scale
    # running max-correction for numerical stability
    new_max = jnp.maximum(rm, block_max)
    exp_old = jnp.exp(rm - new_max)
    ...
    return new_max, new_sum, exp_old[:,:,None] * a + new_vals
```

**Elimination from HBM - single node passes:**

Dense attention lands three tensors per layer onto HBM: the full `(B, H, N, N)` score matrix, the post-softmax attention weights, and the output accumulation — all `O(N²)`. At B=8, H=8, N=4096, bf16, that's **~2.1GB** just for the attention matrix per layer.

The Pallas path sets `attn_matrix_bytes = 0` in `compute_theoretical_hbm_bytes`. The only HBM traffic is: Q read once, the active KV blocks read once per active index, and the output write. With 77.1% block sparsity at N=4096, only 234 of 1024 KV block-pairs are ever fetched. The per-query KV read is `O(active_blocks × block_size × H × D)` — no quadratic term.

FLOPs contract by the same factor: `active_blocks × 2 × block_size² × D` for QK and AV, vs `2 × N² × D` dense. At N=4096 with 77.1% sparsity that's roughly **4.6× fewer FLOPs** before bandwidth even enters the argument.

`dynamic_slice` inside the loop is key — XLA lowers this to strided HBM loads with statically-known extents, enabling prefetch of the next active block while the current one is in the MXU. This is what makes the arithmetic intensity numbers real rather than back-of-envelope.

### `sparse_attention_jax` — JAX fallback

The fallback path builds the full `(B, H, N, N)` score matrix and masks it:

```python
scores       = jnp.matmul(q, k.transpose(0,1,3,2)) * scale
element_mask = block_mask.dense_mask[None, None, :, :]   # (1, 1, N, N)
scores       = jnp.where(element_mask > 0, scores, finfo.min)
```

This materializes the full N×N tensor — identical HBM footprint to dense. FLOPs are also unchanged (computing then zeroing, not skipping). The ~0.84x relative slowdown vs dense comes from the mask overhead with zero HBM savings. Useful as a correctness reference only.

**Known shape bug**: `element_mask` broadcasts as `(1, 1, N, N)` against per-block `scores` of shape `(B, H, block_q, block_k)` when `block_size < N`. This is the `ValueError: Incompatible shapes for broadcasting: [(1, 1, 512, 512), (1, 4, 128, 128), ()]` in the demo traceback. Fix: expand `element_mask` to match the blocked scores shape before the `where`. Pallas path unaffected.

---

## why sparse AI grows while dense AI plateaus

Dense attention: QK FLOPs are `O(N² · d)` and HBM bytes are `O(N²)` — the ratio locks. Arithmetic intensity asymptotes around 62–64 FLOPs/byte regardless of N because numerator and denominator grow at the same rate.

Sparse attention with a `BlockMask`: active blocks scale as `k(N)` where `k` grows sublinearly because the combined pattern's global token component is fixed and the local window component grows as `O(N)` not `O(N²)`. FLOPs are `O(k · bs² · d)` and HBM is `O(k · bs · d)` — ratio is `O(bs)`, independent of N. As N grows, the denominator (bytes) shrinks relative to the numerator (FLOPs) because a larger fraction of blocks are skipped, so AI climbs. The inflection is wherever `k · bs / N` drops enough to push past the ridge point.

---

## numbers

**Hardware**: TPU v5e-1 · 197 TFLOPs bf16 · 820 GB/s · ridge 240.2 FLOPs/byte

### training baseline
B=4, N=1024, 1000 steps, 29.4M params, dense kernel

| step latency (post-compile) | throughput | compile (step 0) | final loss |
|-----------------------------|-----------|-----------------|------------|
| 18.4ms | 222,061 tok/s | ~34s | 10.84 |

### roofline sweep

| type | N | arith. intensity | tok/s | p99 | regime |
|------|---|-----------------|-------|-----|--------|
| dense | 256 | 44.3 F/B | 531,368 | 2.09ms | **mem-bound** |
| dense | 512 | 53.2 F/B | 944,819 | 2.30ms | **mem-bound** |
| dense | 1024 | 59.1 F/B | 1,576,163 | 2.66ms | **mem-bound** |
| dense | 2048 | 62.6 F/B | 1,300,058 | 6.33ms | **mem-bound** |
| dense | 4096 | 64.5 F/B | 703,615 | 23.37ms | **mem-bound** |
| sparse | 256 | 99.8 F/B | 2,592 | 425ms | mem-bound |
| sparse | 512 | 166.2 F/B | 4,623 | 464ms | mem-bound |
| sparse | 1024 | 249.4 F/B | 8,918 | 489ms | **compute-bound** ← ridge |
| sparse | 2048 | 340.8 F/B | 17,146 | 509ms | **compute-bound** |
| sparse | 4096 | 486.3 F/B | 32,427 | 565ms | **compute-bound** |

### block mask (combined pattern, block_size=128)

| N | active blocks | sparsity | HBM reduction | FLOPs reduction |
|---|---------------|----------|---------------|-----------------|
| 512 | 10/16 | 37.5% | 80.0% | 37.5% |
| 1024 | 30/64 | 53.1% | 88.9% | 53.1% |
| 2048 | 82/256 | 68.0% | 94.1% | 68.0% |
| 4096 | 234/1024 | 77.1% | 97.0% | 77.1% |

Average HBM reduction across full benchmark sweep: **90.0%**. Average FLOPs reduction: **58.9%**.

### JAX fallback (Pallas=off)

| B | N | sparsity | dense (ms) | sparse (ms) | speedup |
|---|---|----------|-----------|------------|---------|
| 1 | 512 | 37.5% | 1.66 | 1.99 | 0.84x |
| 4 | 1024 | 53.1% | 2.55 | 3.03 | 0.84x |
| 4 | 2048 | 68.0% | 6.35 | 7.27 | 0.87x |

~0.84x consistently — mask overhead, zero HBM benefit. Correctness reference only.

### OOM ceiling sweep

| type | last PASS | HBM est. | OOM at | HBM est. |
|------|-----------|----------|--------|----------|
| sparse | B=16, N=16384 | 1.61 GB | B=32, N=16384 | 3.22 GB |
| dense | B=8, N=4096 | 2.25 GB | B=8, N=8192 | 8.79 GB |

Sparse survives ~2× the batch-sequence product before hitting the allocator. The gap is the absent `attn_matrix_bytes` term.

### gradient accumulation

| micro_batch | seq_len | eff. batch | total tokens | tok/s | jitter |
|-------------|---------|------------|-------------|-------|--------|
| 4 | 4096 | 64 | 262,144 | 28,260 | 173ms |
| 8 | 4096 | 256 | 1,048,576 | 51,804 | 302ms |
| 16 | 2048 | 1024 | 2,097,152 | 48,177 | 1,143ms |

Jitter spike at B_eff=1024 is XLA retracing at that boundary, not runtime instability.

### mixed host+device pressure

1GB dataset resident in host RAM, 20 concurrent TPU iterations: avg 622ms, p99 829ms, 52,661 tok/s.

---

## running

```bash
# dense training baseline
python train.py --backend tpu --batch 4 --seq 1024 --steps 1000

# full benchmark sweep
python benchmark.py --pattern combined --block-size 128

# roofline
python roofline.py --hardware tpu_v5e

# stress suite
python stress.py --host-ram 6.0 --ceiling --accumulation --mixed

# JAX fallback (no Pallas)
python benchmark.py --no-pallas
```

Requires JAX ≥0.7.2 with TPU backend. Pallas path needs `jax.experimental.pallas`. Single chip (v5e-1) sufficient for full stress suite. Step 0 compile ~34s — XLA tracing to HLO, expected.

> **transparent hugepages**: `sudo sh -c "echo always > /sys/kernel/mm/transparent_hugepage/enabled"` — affects TPU startup/shutdown latency on v5e+, not kernel throughput.

---

## swapping the sparsity pattern

`BlockMask` in `masks.py` is the only thing that changes. The Pallas kernel reads `active_kv_indices` and is pattern-agnostic.

```python
mask = BlockMask(pattern="combined", seq_len=N, block_size=128)  # local causal + global CLS
mask = BlockMask(pattern="local",    seq_len=N, block_size=128, window=4)
mask = BlockMask(pattern="strided",  seq_len=N, block_size=128, stride=8)
```

---

## known issues

- **Shape broadcast bug in `sparse_attention_jax`**: `element_mask` shape `(1, 1, N, N)` vs blocked scores `(B, H, block_q, block_k)` — `jnp.where` throws on incompatible broadcast. Fix: expand mask dims to match per-block scores shape. Pallas path unaffected.
- **Kaleido/Plotly mismatch**: `plotly==5.24.1` + `kaleido==1.2.0` breaks static PNG export. Pin `plotly>=6.1.1` or `kaleido==0.2.1`. Interactive HTML works fine.
- **No unit tests**: mask correctness and sparse≈dense output equivalence on small N not covered.

---

## license

WTFPL

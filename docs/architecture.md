# Technical Architecture

Deep-dive into the kernel design, memory access patterns, and optimization strategies.

## System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        Input Pipeline                           │
│  Text → GPT-2 Tokenizer → Token IDs → Embedding → Q, K, V     │
└──────────────┬───────────────────────────────────────────────────┘
               │
┌──────────────▼───────────────────────────────────────────────────┐
│                    Block Mask Generator                          │
│  Sparsity Pattern (causal/strided/combined) → BlockMask         │
│  (n_q_blocks × n_kv_blocks) boolean array                       │
└──────────────┬───────────────────────────────────────────────────┘
               │
┌──────────────▼───────────────────────────────────────────────────┐
│                  Sparse Attention Kernel                         │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ For each query block q_i:                                   │ │
│  │   1. Load Q block from HBM → registers                     │ │
│  │   2. For each KV block j in active_blocks[i]:               │ │
│  │      a. Load K_j, V_j from HBM → registers                 │ │
│  │      b. Compute S = Q_i × K_j^T × scale                    │ │
│  │      c. Online softmax: update (max, sum, accumulator)      │ │
│  │   3. Normalize: output = acc / sum                          │ │
│  │   4. Write output block to HBM                              │ │
│  └─────────────────────────────────────────────────────────────┘ │
└──────────────┬───────────────────────────────────────────────────┘
               │
┌──────────────▼───────────────────────────────────────────────────┐
│                   Transformer Stack                              │
│  [Attention → Residual → LayerNorm → FFN → Residual] × N       │
└──────────────────────────────────────────────────────────────────┘
```

## Kernel Design

### Online Softmax Algorithm

Standard softmax requires two passes over the data:
1. **Pass 1**: Find max → compute exp(x - max) → sum
2. **Pass 2**: Normalize by sum

This means the entire N×N attention matrix must be materialized in HBM.

**Online softmax** does it in a single pass by maintaining running statistics:

```
Initialize: max = -∞, sum = 0, acc = 0

For each KV block k:
  scores = Q_block × K_block^T × scale        # (bs × bs)
  block_max = max(scores)
  
  # Update running max
  new_max = max(running_max, block_max)
  
  # Rescale old accumulator to new max
  correction = exp(old_max - new_max)
  
  # Compute new exponentials
  exp_scores = exp(scores - new_max)
  
  # Update sum and accumulator
  new_sum = correction × old_sum + sum(exp_scores)
  new_acc = correction × old_acc + exp_scores × V_block

Final: output = acc / sum
```

**Result**: The N×N attention matrix is **never stored in memory**. Each block's scores are computed, used, and discarded immediately.

### Block-Sparse Masking Strategy

The attention matrix is divided into `(N/bs) × (N/bs)` blocks of size `bs × bs`.

```
Full attention matrix (N=8, bs=2):

  K₀ K₁ K₂ K₃
Q₀ [██ ██ ░░ ░░]     ██ = Active block (computed)
Q₁ [██ ██ ██ ░░]     ░░ = Masked block (SKIPPED)
Q₂ [██ ░░ ██ ██]
Q₃ [░░ ██ ░░ ██]

Active blocks: 10/16 = 62.5% density (37.5% sparsity)
```

For each masked block ░░:
- **K/V NOT loaded from HBM** → bandwidth saved
- **QK^T NOT computed** → FLOPs saved
- **No softmax contribution** → correct by definition

### Memory Access Pattern

#### Dense Attention

```
HBM reads:   Q(B×N×H×D) + K(B×N×H×D) + V(B×N×H×D)
HBM writes:  Attn(B×H×N×N) + Output(B×N×H×D)
             ↑↑↑↑↑↑↑↑↑↑↑↑↑↑
             This is O(N²) and dominates!
```

#### Sparse Attention (this kernel)

```
HBM reads:   Q(B×N×H×D) + K_active(B×ρ×N×H×D) + V_active(B×ρ×N×H×D)
HBM writes:  Output(B×N×H×D)
             No attention matrix write!

Where ρ = fraction of active KV blocks < 1
```

### Fused Operations

Instead of separate operations:
```
S = Q × K^T           # Write to HBM
S = S * scale          # Read/Write HBM
S = mask(S)            # Read/Write HBM
S = softmax(S)         # Read/Write HBM
O = S × V             # Read/Write HBM
```

The kernel fuses everything:
```
For each active KV block:      # Single fused loop
  S = Q × K^T * scale         # In registers
  update_online_softmax(S, V)  # In registers
  # Nothing written to HBM until final output!
```

**Result**: Instead of 5 HBM round-trips per block, we do 1 read + 1 write total.

## Sparsity Patterns

### Combined (Recommended)

Union of causal + strided. Provides both autoregressive validity and long-range connectivity:

```
Block mask for N=1024, bs=128 (8×8 blocks):

  ████████
  ████░░░░      ██ = Active
  ██░░██░░      ░░ = Skipped
  ████░░░░
  ██░░██░░
  ░░██░░██
  ██░░░░██
  ░░████░░
```

Typical sparsity: 45-55% → ~40% HBM reduction

### Why 40% HBM Reduction?

Given combined pattern with ~50% block sparsity:

| Component | Dense | Sparse | Saving |
|-----------|-------|--------|--------|
| Q read | 100% | 100% | 0% |
| K read | 100% | ~50% | 50% |
| V read | 100% | ~50% | 50% |
| Attn matrix | 100% | 0% | 100% |
| Output write | 100% | 100% | 0% |

The attn matrix elimination alone gives huge savings for large N. Combined with K/V savings, total HBM reduction reaches ~40%.

## TPU Hardware Considerations

### TPU v5e-1 Specs

- **TensorCores**: Optimized for matrix multiply (bf16)
- **HBM**: ~16 GB, ~820 GB/s bandwidth
- **VMEM**: ~32 MB on-chip (fast scratchpad)
- **Peak Compute**: ~197 TFLOPs (bf16)

### Memory Hierarchy

```
┌─────────────────┐
│    Registers     │  ← Fastest, smallest
├─────────────────┤
│      VMEM        │  ← TPU scratchpad (32 MB)
├─────────────────┤
│       HBM        │  ← Main memory (16 GB, 820 GB/s)
└─────────────────┘
```

Our kernel keeps intermediate values (softmax state, block scores) in registers/VMEM and only reads Q/K/V from HBM once per block.

### Why Block-Sparse is TPU-Friendly

1. **Regular access patterns**: Block-aligned reads map to HBM burst transfers
2. **No dynamic indexing overhead**: Block mask is known at compile time
3. **TensorCore utilization**: `bs × bs` matmuls map directly to TPU MXUs
4. **Prefetchable**: Block indices can be loaded into scalar registers ahead of time

## Algorithm Complexity

| Operation | Dense Attention | Sparse Attention |
|-----------|----------------|------------------|
| Time | O(N²·D) | O(ρ·N²·D) where ρ < 1 |
| HBM reads | O(N²) | O(ρ·N²) |
| HBM writes | O(N²) | O(N·D) |
| Peak memory | O(N²) | O(N·D + bs²) |

Where ρ = density of block mask (typically 0.4-0.6).

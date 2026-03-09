import argparse
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

from config import ProjectConfig
from sparse_attention.masks import create_block_mask
from sparse_attention.model import create_model, init_model, count_parameters
from sparse_attention.data import (
    create_demo_batch,
    create_random_token_batch,
    tokenize_text,
    decode_tokens,
    DEMO_CORPUS,
)
from sparse_attention.dense_attention import dense_attention
from sparse_attention.kernel import sparse_attention, compute_theoretical_hbm_bytes
from sparse_attention.metrics import print_device_info, time_function


def main():
    parser = argparse.ArgumentParser(
        description="Sparse Attention Transformer Demo"
    )
    parser.add_argument("--quick", action="store_true",
                        help="Use minimal config for quick test")
    parser.add_argument("--seq-len", type=int, default=None,
                        help="Override sequence length")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size")
    parser.add_argument("--no-pallas", action="store_true",
                        help="Disable Pallas kernel (use JAX fallback)")
    args = parser.parse_args()

    if args.quick:
        cfg = ProjectConfig.for_quick_test()
    else:
        cfg = ProjectConfig()

    if args.seq_len:
        cfg.model.max_seq_len = args.seq_len
        cfg.training.seq_len = args.seq_len

    use_pallas = not args.no_pallas
    seq_len = cfg.training.seq_len
    batch_size = args.batch_size

    if seq_len % cfg.sparse.block_size != 0:
        seq_len = (seq_len // cfg.sparse.block_size) * cfg.sparse.block_size
        print(f"[INFO] Adjusted seq_len to {seq_len} (multiple of block_size)")

    print("=" * 70)
    print("SPARSE ATTENTION TRANSFORMER – DEMO")
    print("=" * 70)
    print_device_info()

    print("\n" + "=" * 70)
    print("STEP 1: Tokenization")
    print("=" * 70)

    try:
        tokens = tokenize_text(DEMO_CORPUS, max_length=seq_len)
        print(f"  Corpus length:  {len(DEMO_CORPUS)} chars")
        print(f"  Token count:    {np.sum(tokens > 0)} tokens")
        print(f"  Padded length:  {len(tokens)}")
        print(f"  First 20 tokens: {tokens[:20]}")

        decoded = decode_tokens(tokens[:50])
        print(f"  Decoded (first 50 tokens): {decoded[:100]}...")
    except ImportError:
        print("  [WARN] tiktoken not available, using random tokens")

    print("\n" + "=" * 70)
    print("STEP 2: Block-Sparse Mask Generation")
    print("=" * 70)

    block_mask = create_block_mask(
        seq_len=seq_len,
        block_size=cfg.sparse.block_size,
        pattern=cfg.sparse.sparsity_type,
    )
    print(f"  {block_mask.summary()}")
    print(f"  Active blocks:  {block_mask.num_active_blocks} / {block_mask.num_total_blocks}")
    print(f"  Sparsity ratio: {block_mask.sparsity_ratio:.1%}")

    print(f"\n  Block mask visualization ({block_mask.num_q_blocks}×{block_mask.num_kv_blocks}):")
    mask_np = np.array(block_mask.mask)
    for i in range(min(block_mask.num_q_blocks, 16)):
        row = ""
        for j in range(min(block_mask.num_kv_blocks, 16)):
            row += "█" if mask_np[i, j] else "░"
        print(f"    {row}")

    print("\n" + "=" * 70)
    print("STEP 3: Theoretical HBM Analysis")
    print("=" * 70)

    dense_hbm = compute_theoretical_hbm_bytes(
        batch_size, seq_len, cfg.model.n_heads, cfg.model.d_head
    )
    sparse_hbm = compute_theoretical_hbm_bytes(
        batch_size, seq_len, cfg.model.n_heads, cfg.model.d_head, block_mask
    )
    hbm_reduction = (
        (dense_hbm["total_bytes"] - sparse_hbm["total_bytes"])
        / dense_hbm["total_bytes"] * 100
    )

    print(f"  Dense HBM:      {dense_hbm['total_mb']:.2f} MB")
    print(f"  Sparse HBM:     {sparse_hbm['total_mb']:.2f} MB")
    print(f"  HBM Reduction:  {hbm_reduction:.1f}%")
    print(f"  Key savings:    Attention matrix NOT materialized (fused online softmax)")
    print(f"                  K/V blocks skipped for {block_mask.sparsity_ratio:.0%} of blocks")

    print("\n" + "=" * 70)
    print("STEP 4: Model Forward Pass")
    print("=" * 70)

    rng = jax.random.PRNGKey(cfg.training.seed)

    sparse_model = create_model(
        attention_type="sparse",
        block_mask=block_mask,
        use_pallas=use_pallas,
        vocab_size=cfg.model.vocab_size,
        d_model=cfg.model.d_model,
        n_heads=cfg.model.n_heads,
        n_layers=cfg.model.n_layers,
        d_ff=cfg.model.d_ff,
        max_seq_len=cfg.model.max_seq_len,
    )

    dense_model = create_model(
        attention_type="dense",
        use_pallas=False,
        vocab_size=cfg.model.vocab_size,
        d_model=cfg.model.d_model,
        n_heads=cfg.model.n_heads,
        n_layers=cfg.model.n_layers,
        d_ff=cfg.model.d_ff,
        max_seq_len=cfg.model.max_seq_len,
    )

    print(f"  Initializing model...")
    variables = init_model(sparse_model, rng, batch_size=1, seq_len=min(seq_len, 128))
    n_params = count_parameters(variables["params"])
    print(f"  Parameters:     {n_params:,}")
    print(f"  Config:         d_model={cfg.model.d_model}, n_heads={cfg.model.n_heads}, "
          f"n_layers={cfg.model.n_layers}")

    input_ids = create_random_token_batch(batch_size, seq_len, cfg.model.vocab_size)

    print(f"\n  Running sparse forward pass (B={batch_size}, N={seq_len})...")
    sparse_logits = sparse_model.apply(variables, input_ids, deterministic=True)
    sparse_logits.block_until_ready()
    print(f"  Sparse output shape: {sparse_logits.shape}")

    print(f"  Running dense forward pass (B={batch_size}, N={seq_len})...")
    dense_logits = dense_model.apply(variables, input_ids, deterministic=True)
    dense_logits.block_until_ready()
    print(f"  Dense output shape: {dense_logits.shape}")

    print("\n" + "=" * 70)
    print("STEP 5: Quick Timing Comparison")
    print("=" * 70)

    n_warmup = 3
    n_iters = 10

    print(f"  Timing sparse model ({n_warmup} warmup, {n_iters} timed)...", end=" ", flush=True)
    sparse_ms, sparse_std, *_ = time_function(
        lambda x: sparse_model.apply(variables, x, deterministic=True),
        input_ids, n_warmup=n_warmup, n_iterations=n_iters,
    )
    print(f"{sparse_ms:.2f} ± {sparse_std:.2f} ms")

    print(f"  Timing dense model ({n_warmup} warmup, {n_iters} timed)...", end=" ", flush=True)
    dense_ms, dense_std, *_ = time_function(
        lambda x: dense_model.apply(variables, x, deterministic=True),
        input_ids, n_warmup=n_warmup, n_iterations=n_iters,
    )
    print(f"{dense_ms:.2f} ± {dense_std:.2f} ms")

    speedup = dense_ms / sparse_ms if sparse_ms > 0 else 0
    print(f"\n  Speedup: {speedup:.2f}x")
    print(f"  Tokens/sec (sparse): {batch_size * seq_len / (sparse_ms / 1000):.0f}")
    print(f"  Tokens/sec (dense):  {batch_size * seq_len / (dense_ms / 1000):.0f}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  ✓ Tokenization:      GPT-2 BPE ({np.sum(tokens > 0)} tokens)")
    print(f"  ✓ Sparsity pattern:  {cfg.sparse.sparsity_type} ({block_mask.sparsity_ratio:.1%} sparse)")
    print(f"  ✓ HBM reduction:     {hbm_reduction:.1f}% (theoretical)")
    print(f"  ✓ Latency speedup:   {speedup:.2f}x")
    print(f"  ✓ Model parameters:  {n_params:,}")
    print(f"\n  Run 'python run_benchmarks.py' for full benchmark suite")
    print(f"  Run 'python run_tests.py' for all unit tests")
    print("=" * 70)


if __name__ == "__main__":
    main()

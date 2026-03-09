from __future__ import annotations

from typing import Dict, List

import jax
import jax.numpy as jnp
import numpy as np

from sparse_attention.kernel import compute_theoretical_flops
from sparse_attention.masks import create_block_mask
from sparse_attention.data import create_dummy_inputs
from sparse_attention.metrics import time_function
from sparse_attention.kernel import sparse_attention
from sparse_attention.dense_attention import dense_attention


def profile_flops_detailed(
    batch_size: int = 4,
    seq_len: int = 2048,
    n_heads: int = 8,
    d_head: int = 32,
    block_size: int = 128,
    sparsity_type: str = "combined",
    n_warmup: int = 3,
    n_iterations: int = 10,
) -> Dict:
    block_mask = create_block_mask(seq_len, block_size, sparsity_type)

    dense_flops = compute_theoretical_flops(batch_size, seq_len, n_heads, d_head)
    sparse_flops = compute_theoretical_flops(
        batch_size, seq_len, n_heads, d_head, block_mask
    )

    reduction = dense_flops["total_flops"] - sparse_flops["total_flops"]
    reduction_pct = (
        reduction / dense_flops["total_flops"] * 100
        if dense_flops["total_flops"] > 0 else 0
    )

    q, k, v = create_dummy_inputs(batch_size, seq_len, n_heads, d_head)

    dense_mean_ms, *_ = time_function(
        lambda q, k, v: dense_attention(q, k, v, causal=True),
        q, k, v, n_warmup=n_warmup, n_iterations=n_iterations,
    )

    sparse_mean_ms, *_ = time_function(
        lambda q, k, v: sparse_attention(q, k, v, block_mask=block_mask),
        q, k, v, n_warmup=n_warmup, n_iterations=n_iterations,
    )

    dense_achieved = (
        dense_flops["total_flops"] / (dense_mean_ms / 1000) / 1e12
        if dense_mean_ms > 0 else 0
    )
    sparse_achieved = (
        sparse_flops["total_flops"] / (sparse_mean_ms / 1000) / 1e12
        if sparse_mean_ms > 0 else 0
    )

    return {
        "config": {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "n_heads": n_heads,
            "d_head": d_head,
            "block_size": block_size,
            "sparsity_type": sparsity_type,
            "sparsity_ratio": block_mask.sparsity_ratio,
        },
        "dense": {
            "qk_flops": dense_flops["qk_flops"],
            "softmax_flops": dense_flops["softmax_flops"],
            "av_flops": dense_flops["av_flops"],
            "total_flops": dense_flops["total_flops"],
            "total_tflops": dense_flops["total_tflops"],
            "latency_ms": dense_mean_ms,
            "achieved_tflops": dense_achieved,
        },
        "sparse": {
            "qk_flops": sparse_flops["qk_flops"],
            "softmax_flops": sparse_flops["softmax_flops"],
            "av_flops": sparse_flops["av_flops"],
            "total_flops": sparse_flops["total_flops"],
            "total_tflops": sparse_flops["total_tflops"],
            "latency_ms": sparse_mean_ms,
            "achieved_tflops": sparse_achieved,
        },
        "reduction": {
            "flops": reduction,
            "percentage": reduction_pct,
        },
    }


def print_flops_report(profiles: List[Dict]):
    print(f"\n{'='*80}")
    print("FLOPs ANALYSIS")
    print(f"{'='*80}")

    for p in profiles:
        cfg = p["config"]
        print(f"\n--- B={cfg['batch_size']}, N={cfg['seq_len']}, "
              f"sparsity={cfg['sparsity_ratio']:.1%} ---")

        print(f"  {'Component':<20} {'Dense (GFLOPs)':>15} {'Sparse (GFLOPs)':>15} {'Saved':>10}")
        print(f"  {'-'*60}")

        for comp, name in [("qk_flops", "Q×K^T"), ("softmax_flops", "Softmax"),
                           ("av_flops", "Attn×V")]:
            d_gf = p["dense"][comp] / 1e9
            s_gf = p["sparse"][comp] / 1e9
            saved = d_gf - s_gf
            print(f"  {name:<20} {d_gf:>14.2f}  {s_gf:>14.2f}  {saved:>9.2f}")

        print(f"  {'-'*60}")
        d_total = p["dense"]["total_tflops"]
        s_total = p["sparse"]["total_tflops"]
        print(f"  {'TOTAL (TFLOPs)':<20} {d_total:>14.4f}  {s_total:>14.4f}")
        print(f"  {'Achieved (TFLOPs)':<20} {p['dense']['achieved_tflops']:>14.4f}  "
              f"{p['sparse']['achieved_tflops']:>14.4f}")
        print(f"  FLOPs Reduction: {p['reduction']['percentage']:.1f}%")

    print(f"\n{'='*80}")


if __name__ == "__main__":
    profiles = [
        profile_flops_detailed(batch_size=1, seq_len=1024),
        profile_flops_detailed(batch_size=4, seq_len=2048),
    ]
    print_flops_report(profiles)

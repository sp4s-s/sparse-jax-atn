from __future__ import annotations

from typing import Dict, List

import jax
import jax.numpy as jnp
import numpy as np

from sparse_attention.kernel import compute_theoretical_flops, sparse_attention
from sparse_attention.dense_attention import dense_attention
from sparse_attention.masks import create_block_mask
from sparse_attention.data import create_dummy_inputs
from sparse_attention.metrics import time_function


def scaling_analysis(
    seq_lengths: List[int] = [256, 512, 1024, 2048, 4096],
    batch_size: int = 1,
    n_heads: int = 8,
    d_head: int = 32,
    block_size: int = 128,
    sparsity_type: str = "combined",
    n_warmup: int = 3,
    n_iterations: int = 10,
) -> List[Dict]:
    results = []

    for seq_len in seq_lengths:
        if seq_len < block_size:
            continue

        print(f"  N={seq_len}...", end=" ", flush=True)

        block_mask = create_block_mask(seq_len, block_size, sparsity_type)
        q, k, v = create_dummy_inputs(batch_size, seq_len, n_heads, d_head)

        dense_flops = compute_theoretical_flops(batch_size, seq_len, n_heads, d_head)
        sparse_flops = compute_theoretical_flops(
            batch_size, seq_len, n_heads, d_head, block_mask
        )

        dense_ms, *_ = time_function(
            lambda q, k, v: dense_attention(q, k, v, causal=True),
            q, k, v, n_warmup=n_warmup, n_iterations=n_iterations,
        )
        sparse_ms, *_ = time_function(
            lambda q, k, v: sparse_attention(q, k, v, block_mask=block_mask),
            q, k, v, n_warmup=n_warmup, n_iterations=n_iterations,
        )

        results.append({
            "seq_len": seq_len,
            "sparsity_ratio": block_mask.sparsity_ratio,
            "dense_flops": dense_flops["total_flops"],
            "sparse_flops": sparse_flops["total_flops"],
            "dense_latency_ms": dense_ms,
            "sparse_latency_ms": sparse_ms,
            "speedup": dense_ms / sparse_ms if sparse_ms > 0 else 0,
            "flops_ratio": (sparse_flops["total_flops"] / dense_flops["total_flops"]
                           if dense_flops["total_flops"] > 0 else 0),
        })
        print(f"done (dense: {dense_ms:.2f}ms, sparse: {sparse_ms:.2f}ms)")

    return results


def print_scaling_report(results: List[Dict]):
    print(f"\n{'='*80}")
    print("SCALING ANALYSIS: Sparse vs Dense Attention")
    print(f"{'='*80}")

    print(f"\n  {'N':>6} {'Sparsity':>10} {'Dense (ms)':>12} {'Sparse (ms)':>12} "
          f"{'Speedup':>10} {'FLOPs Ratio':>12}")
    print(f"  {'-'*62}")

    for r in results:
        print(f"  {r['seq_len']:>6} {r['sparsity_ratio']:>9.1%} "
              f"{r['dense_latency_ms']:>11.2f} {r['sparse_latency_ms']:>11.2f} "
              f"{r['speedup']:>9.2f}x {r['flops_ratio']:>11.4f}")

    if len(results) >= 2:
        print(f"\n  Scaling trend (N doubling):")
        for i in range(1, len(results)):
            if results[i-1]["seq_len"] * 2 == results[i]["seq_len"]:
                dense_ratio = (results[i]["dense_latency_ms"] /
                              results[i-1]["dense_latency_ms"])
                sparse_ratio = (results[i]["sparse_latency_ms"] /
                               results[i-1]["sparse_latency_ms"])
                print(f"    N: {results[i-1]['seq_len']} -> {results[i]['seq_len']}: "
                      f"Dense {dense_ratio:.2f}x, Sparse {sparse_ratio:.2f}x "
                      f"(ideal O(N²) = 4.0x, ideal O(N) = 2.0x)")

    print(f"\n{'='*80}")


if __name__ == "__main__":
    print("Running scaling analysis...")
    results = scaling_analysis()
    print_scaling_report(results)

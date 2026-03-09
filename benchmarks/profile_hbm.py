from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from sparse_attention.kernel import (
    compute_theoretical_hbm_bytes,
    sparse_attention,
)
from sparse_attention.dense_attention import dense_attention
from sparse_attention.masks import create_block_mask, BlockMask
from sparse_attention.data import create_dummy_inputs
from sparse_attention.metrics import time_function


def profile_hbm_detailed(
    batch_size: int = 4,
    seq_len: int = 2048,
    n_heads: int = 8,
    d_head: int = 32,
    block_size: int = 128,
    sparsity_type: str = "combined",
    dtype_bytes: int = 2,
) -> Dict:
    block_mask = create_block_mask(seq_len, block_size, sparsity_type)

    dense_hbm = compute_theoretical_hbm_bytes(
        batch_size, seq_len, n_heads, d_head, block_mask=None, dtype_bytes=dtype_bytes
    )

    sparse_hbm = compute_theoretical_hbm_bytes(
        batch_size, seq_len, n_heads, d_head, block_mask=block_mask, dtype_bytes=dtype_bytes
    )

    reduction_bytes = dense_hbm["total_bytes"] - sparse_hbm["total_bytes"]
    reduction_pct = (reduction_bytes / dense_hbm["total_bytes"] * 100
                     if dense_hbm["total_bytes"] > 0 else 0)

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
            "q_bytes": dense_hbm["q_bytes"],
            "k_bytes": dense_hbm["k_bytes"],
            "v_bytes": dense_hbm["v_bytes"],
            "attn_matrix_bytes": dense_hbm["attn_matrix_bytes"],
            "output_bytes": dense_hbm["output_bytes"],
            "total_bytes": dense_hbm["total_bytes"],
            "total_mb": dense_hbm["total_mb"],
            "total_gb": dense_hbm["total_gb"],
        },
        "sparse": {
            "q_bytes": sparse_hbm["q_bytes"],
            "k_bytes": sparse_hbm["k_bytes"],
            "v_bytes": sparse_hbm["v_bytes"],
            "attn_matrix_bytes": sparse_hbm["attn_matrix_bytes"],
            "output_bytes": sparse_hbm["output_bytes"],
            "total_bytes": sparse_hbm["total_bytes"],
            "total_mb": sparse_hbm["total_mb"],
            "total_gb": sparse_hbm["total_gb"],
        },
        "reduction": {
            "bytes": reduction_bytes,
            "mb": reduction_bytes / (1024 * 1024),
            "percentage": reduction_pct,
        },
        "block_mask_info": block_mask.summary(),
    }


def profile_hbm_sweep(
    seq_lengths: List[int] = [512, 1024, 2048, 4096],
    batch_sizes: List[int] = [1, 4],
    n_heads: int = 8,
    d_head: int = 32,
    block_size: int = 128,
    sparsity_type: str = "combined",
) -> List[Dict]:
    results = []
    for seq_len in seq_lengths:
        for batch_size in batch_sizes:
            profile = profile_hbm_detailed(
                batch_size=batch_size,
                seq_len=seq_len,
                n_heads=n_heads,
                d_head=d_head,
                block_size=block_size,
                sparsity_type=sparsity_type,
            )
            results.append(profile)
    return results


def print_hbm_report(profiles: List[Dict]):
    print(f"\n{'='*80}")
    print("HBM BANDWIDTH ANALYSIS")
    print(f"{'='*80}")

    for p in profiles:
        cfg = p["config"]
        print(f"\n--- B={cfg['batch_size']}, N={cfg['seq_len']}, "
              f"sparsity={cfg['sparsity_ratio']:.1%} ---")
        print(f"  {'Component':<25} {'Dense (MB)':>12} {'Sparse (MB)':>12} {'Saved (MB)':>12}")
        print(f"  {'-'*61}")

        components = ["q_bytes", "k_bytes", "v_bytes", "attn_matrix_bytes", "output_bytes"]
        names = ["Q (read)", "K (read)", "V (read)", "Attn Matrix (R/W)", "Output (write)"]
        for name, comp in zip(names, components):
            d_mb = p["dense"][comp] / (1024*1024)
            s_mb = p["sparse"][comp] / (1024*1024)
            saved = d_mb - s_mb
            print(f"  {name:<25} {d_mb:>11.2f}  {s_mb:>11.2f}  {saved:>11.2f}")

        print(f"  {'-'*61}")
        print(f"  {'TOTAL':<25} {p['dense']['total_mb']:>11.2f}  "
              f"{p['sparse']['total_mb']:>11.2f}  "
              f"{p['reduction']['mb']:>11.2f}")
        print(f"  HBM Bandwidth Reduction: {p['reduction']['percentage']:.1f}%")

    print(f"\n{'='*80}")


if __name__ == "__main__":
    profiles = profile_hbm_sweep()
    print_hbm_report(profiles)

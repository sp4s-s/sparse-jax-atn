from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from sparse_attention.kernel import sparse_attention, sparse_attention_jax
from sparse_attention.dense_attention import dense_attention
from sparse_attention.masks import create_block_mask, BlockMask
from sparse_attention.data import create_dummy_inputs
from sparse_attention.metrics import (
    BenchmarkResult,
    ComparisonResult,
    benchmark_attention,
    compare_attention,
    format_results_table,
    format_comparison_table,
    print_device_info,
)


def run_benchmark_suite(
    seq_lengths: List[int] = [512, 1024, 2048, 4096],
    batch_sizes: List[int] = [1, 2, 4, 8],
    n_heads: int = 8,
    d_head: int = 32,
    block_size: int = 128,
    sparsity_type: str = "combined",
    n_warmup: int = 5,
    n_iterations: int = 20,
    use_pallas: bool = True,
    save_results: bool = True,
    output_dir: str = "benchmark_results",
) -> Tuple[List[ComparisonResult], Dict]:
    print_device_info()
    print(f"\n{'='*70}")
    print(f"BENCHMARK SUITE: Sparse vs Dense Attention")
    print(f"{'='*70}")
    print(f"Sparsity pattern: {sparsity_type}")
    print(f"Block size:       {block_size}")
    print(f"Heads:            {n_heads} × {d_head}d")
    print(f"Seq lengths:      {seq_lengths}")
    print(f"Batch sizes:      {batch_sizes}")
    print(f"Warmup/Iters:     {n_warmup}/{n_iterations}")
    print(f"Use Pallas:       {use_pallas}")
    print(f"{'='*70}\n")

    all_sparse_results = []
    all_dense_results = []
    all_comparisons = []

    for seq_len in seq_lengths:
        for batch_size in batch_sizes:
            print(f"\n--- B={batch_size}, N={seq_len} ---")

            try:
                q, k, v = create_dummy_inputs(
                    batch_size=batch_size,
                    seq_len=seq_len,
                    n_heads=n_heads,
                    d_head=d_head,
                )
            except Exception as e:
                print(f"  [SKIP] Failed to create inputs: {e}")
                continue

            block_mask = create_block_mask(seq_len, block_size, sparsity_type)
            print(f"  {block_mask.summary()}")

            try:
                print(f"  Benchmarking dense attention...", end=" ", flush=True)
                dense_result = benchmark_attention(
                    attention_fn=lambda q, k, v: dense_attention(q, k, v, causal=True),
                    query=q, key=k, value=v,
                    name=f"dense_B{batch_size}_N{seq_len}",
                    attention_type="dense",
                    block_mask=None,
                    n_warmup=n_warmup,
                    n_iterations=n_iterations,
                )
                from sparse_attention.kernel import compute_theoretical_hbm_bytes
                dense_hbm = compute_theoretical_hbm_bytes(
                    batch_size, seq_len, n_heads, d_head, block_mask=None
                )
                dense_result.theoretical_hbm_bytes = dense_hbm["total_bytes"]
                dense_result.theoretical_hbm_mb = dense_hbm["total_mb"]

                print(f"done ({dense_result.latency_mean_ms:.2f} ms)")
                all_dense_results.append(dense_result)
            except Exception as e:
                print(f"FAILED: {e}")
                continue

            try:
                print(f"  Benchmarking sparse attention...", end=" ", flush=True)

                def sparse_fn(q, k, v):
                    return sparse_attention(q, k, v, block_mask=block_mask,
                                            use_pallas=use_pallas)

                sparse_result = benchmark_attention(
                    attention_fn=sparse_fn,
                    query=q, key=k, value=v,
                    name=f"sparse_B{batch_size}_N{seq_len}",
                    attention_type="sparse",
                    block_mask=block_mask,
                    n_warmup=n_warmup,
                    n_iterations=n_iterations,
                )
                print(f"done ({sparse_result.latency_mean_ms:.2f} ms)")
                all_sparse_results.append(sparse_result)
            except Exception as e:
                print(f"FAILED: {e}")
                continue

            comparison = compare_attention(sparse_result, dense_result)
            all_comparisons.append(comparison)

            print(f"  HBM Reduction:  {comparison.hbm_reduction_pct:.1f}%")
            print(f"  FLOPs Reduction: {comparison.flops_reduction_pct:.1f}%")
            print(f"  Speedup:        {comparison.speedup:.2f}x")

    print(f"\n")
    print("RESULTS SUMMARY")
    print(f"{'='*70}")

    if all_comparisons:
        print("\n--- Comparison Table ---")
        print(format_comparison_table(all_comparisons))

        avg_hbm_reduction = np.mean([c.hbm_reduction_pct for c in all_comparisons])
        avg_flops_reduction = np.mean([c.flops_reduction_pct for c in all_comparisons])
        avg_speedup = np.mean([c.speedup for c in all_comparisons])

        print(f"\n{'='*70}")
        print("HEADLINE METRICS")
        print(f"{'='*70}")
        print(f"  Average HBM Bandwidth Reduction:  {avg_hbm_reduction:.1f}%")
        print(f"  Average FLOPs Reduction:           {avg_flops_reduction:.1f}%")
        print(f"  Average Speedup:                   {avg_speedup:.2f}x")
        print(f"{'='*70}")

    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": str(jax.devices()),
        "jax_version": jax.__version__,
        "config": {
            "sparsity_type": sparsity_type,
            "block_size": block_size,
            "n_heads": n_heads,
            "d_head": d_head,
            "n_warmup": n_warmup,
            "n_iterations": n_iterations,
        },
        "comparisons": [c.summary_dict() for c in all_comparisons],
        "headline": {
            "avg_hbm_reduction_pct": float(avg_hbm_reduction) if all_comparisons else 0,
            "avg_flops_reduction_pct": float(avg_flops_reduction) if all_comparisons else 0,
            "avg_speedup": float(avg_speedup) if all_comparisons else 0,
        },
    }

    if save_results:
        os.makedirs(output_dir, exist_ok=True)
        results_file = os.path.join(output_dir, "benchmark_results.json")
        with open(results_file, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"\nResults saved to {results_file}")

    return all_comparisons, summary


def run_quick_benchmark(
    n_heads: int = 8,
    d_head: int = 32,
    block_size: int = 128,
    sparsity_type: str = "combined",
    use_pallas: bool = True,
) -> Tuple[List[ComparisonResult], Dict]:
    return run_benchmark_suite(
        seq_lengths=[512, 1024, 2048],
        batch_sizes=[1, 4],
        n_heads=n_heads,
        d_head=d_head,
        block_size=block_size,
        sparsity_type=sparsity_type,
        n_warmup=3,
        n_iterations=10,
        use_pallas=use_pallas,
    )

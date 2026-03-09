from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import jax
import jax.numpy as jnp
import numpy as np

from sparse_attention.kernel import (
    sparse_attention,
    compute_theoretical_flops,
    compute_theoretical_hbm_bytes,
)
from sparse_attention.dense_attention import dense_attention
from sparse_attention.masks import create_block_mask
from sparse_attention.data import create_dummy_inputs
from sparse_attention.live_viz import LiveNotebookDisplay, render_roofline_live
from sparse_attention.metrics import time_function
from sparse_attention.runtime_backend import require_tpu


TPU_V5E_SPECS = {
    "name": "TPU v5e-1",
    "peak_tflops_bf16": 197.0,
    "peak_tflops_fp32": 98.5,
    "hbm_bandwidth_gb_s": 820.0,
    "hbm_capacity_gb": 16.0,
    "vmem_mb": 32.0,
}

GPU_A100_SPECS = {
    "name": "A100 80GB",
    "peak_tflops_bf16": 312.0,
    "peak_tflops_fp32": 156.0,
    "hbm_bandwidth_gb_s": 2039.0,
    "hbm_capacity_gb": 80.0,
}


def compute_arithmetic_intensity(
    batch_size: int,
    seq_len: int,
    n_heads: int,
    d_head: int,
    block_mask=None,
) -> float:
    flops = compute_theoretical_flops(batch_size, seq_len, n_heads, d_head, block_mask)
    hbm = compute_theoretical_hbm_bytes(batch_size, seq_len, n_heads, d_head, block_mask)

    if hbm["total_bytes"] == 0:
        return 0.0
    return flops["total_flops"] / hbm["total_bytes"]


def compute_mfu(
    achieved_tflops: float,
    theoretical_flops: int,
    latency_s: float,
) -> float:
    if latency_s == 0 or theoretical_flops == 0:
        return 0.0
    theoretical_tflops_per_sec = theoretical_flops / latency_s / 1e12
    if theoretical_tflops_per_sec == 0:
        return 0.0
    return (achieved_tflops / theoretical_tflops_per_sec) * 100


def compute_hfu(
    achieved_tflops: float,
    peak_tflops: float = TPU_V5E_SPECS["peak_tflops_bf16"],
) -> float:
    if peak_tflops == 0:
        return 0.0
    return (achieved_tflops / peak_tflops) * 100


def compute_memory_efficiency(
    batch_size: int,
    seq_len: int,
    n_heads: int,
    d_head: int,
    block_mask=None,
    dtype_bytes: int = 2,
) -> Dict:
    B, N, H, D = batch_size, seq_len, n_heads, d_head

    useful_bytes = 4 * B * N * H * D * dtype_bytes

    hbm = compute_theoretical_hbm_bytes(B, N, H, D, block_mask, dtype_bytes)
    total_bytes = hbm["total_bytes"]

    efficiency = useful_bytes / total_bytes * 100 if total_bytes > 0 else 0

    return {
        "useful_bytes": useful_bytes,
        "total_bytes": total_bytes,
        "efficiency_pct": efficiency,
        "overhead_bytes": total_bytes - useful_bytes,
        "overhead_pct": 100 - efficiency,
    }


def roofline_analysis(
    seq_lengths: List[int] = [256, 512, 1024, 2048, 4096],
    batch_size: int = 4,
    n_heads: int = 8,
    d_head: int = 32,
    block_size: int = 128,
    sparsity_type: str = "combined",
    n_warmup: int = 3,
    n_iterations: int = 10,
    use_pallas: bool = True,
    hw_specs: Dict = TPU_V5E_SPECS,
    output_dir: str = "benchmark_results",
) -> Dict:
    require_tpu("Roofline analysis")
    print(f"\n")
    print(f"ROOFLINE MODEL ANALYSIS")
    print(f"{'='*70}")
    print(f"Hardware: {hw_specs['name']}")
    print(f"Peak compute: {hw_specs['peak_tflops_bf16']:.0f} TFLOPs (bf16)")
    print(f"Peak bandwidth: {hw_specs['hbm_bandwidth_gb_s']:.0f} GB/s")
    ridge = hw_specs['peak_tflops_bf16'] * 1000 / hw_specs['hbm_bandwidth_gb_s']
    print(f"Ridge point: {ridge:.1f} FLOPs/byte")
    live = LiveNotebookDisplay("roofline", os.path.join(output_dir, "live"), "roofline_live.html", min_interval_steps=1)

    results = []
    roofline_points = []

    for seq_len in seq_lengths:
        if seq_len < block_size:
            continue

        print(f"\n  N={seq_len}:")
        q, k, v = create_dummy_inputs(batch_size, seq_len, n_heads, d_head)
        block_mask = create_block_mask(seq_len, block_size, sparsity_type)

        for attn_type in ["dense", "sparse"]:
            if attn_type == "sparse":
                fn = lambda q, k, v: sparse_attention(q, k, v, block_mask, use_pallas=use_pallas)
                bm = block_mask
            else:
                fn = lambda q, k, v: dense_attention(q, k, v, causal=True)
                bm = None

            mean_ms, std_ms, min_ms, max_ms, all_times = time_function(
                fn, q, k, v, n_warmup=n_warmup, n_iterations=n_iterations,
            )
            latency_s = mean_ms / 1000

            flops = compute_theoretical_flops(batch_size, seq_len, n_heads, d_head, bm)
            hbm = compute_theoretical_hbm_bytes(batch_size, seq_len, n_heads, d_head, bm)
            hbm_mb = hbm["total_bytes"] / (1024 ** 2)

            ai = compute_arithmetic_intensity(batch_size, seq_len, n_heads, d_head, bm)
            achieved_tflops = flops["total_flops"] / latency_s / 1e12 if latency_s > 0 else 0
            mfu = compute_mfu(achieved_tflops, flops["total_flops"], latency_s)
            hfu = compute_hfu(achieved_tflops, hw_specs["peak_tflops_bf16"])
            mem_eff = compute_memory_efficiency(batch_size, seq_len, n_heads, d_head, bm)
            bw_gb_s = hbm["total_bytes"] / latency_s / 1e9 if latency_s > 0 else 0
            bw_util = bw_gb_s / hw_specs["hbm_bandwidth_gb_s"] * 100

            tokens_per_sec = batch_size * seq_len / latency_s if latency_s > 0 else 0

            p50 = float(np.percentile(all_times, 50))
            p99 = float(np.percentile(all_times, 99))

            entry = {
                "type": attn_type,
                "seq_len": seq_len,
                "batch_size": batch_size,
                "sparsity_ratio": block_mask.sparsity_ratio if attn_type == "sparse" else 0,
                "latency_mean_ms": mean_ms,
                "latency_std_ms": std_ms,
                "latency_p50_ms": p50,
                "latency_p99_ms": p99,
                "tokens_per_sec": tokens_per_sec,
                "theoretical_flops": flops["total_flops"],
                "theoretical_tflops": flops["total_tflops"],
                "achieved_tflops": achieved_tflops,
                "arithmetic_intensity": ai,
                "mfu_pct": mfu,
                "hfu_pct": hfu,
                "memory_efficiency_pct": mem_eff["efficiency_pct"],
                "hbm_bytes": hbm["total_bytes"],
                "hbm_mb": hbm_mb,
                "bandwidth_gb_s": bw_gb_s,
                "bandwidth_utilization_pct": bw_util,
                "is_memory_bound": ai < ridge,
            }
            results.append(entry)

            roofline_points.append({
                "name": f"{attn_type} N={seq_len}",
                "type": attn_type,
                "arithmetic_intensity": ai,
                "achieved_tflops": achieved_tflops,
            })

            bound = "MEM-BOUND" if ai < ridge else "COMPUTE-BOUND"
            print(f"    [{attn_type:>6}] AI={ai:.1f} FLOPs/B, "
                  f"{achieved_tflops:.4f} TFLOPs, "
                  f"MFU={mfu:.1f}%, HFU={hfu:.1f}%, "
                  f"BW={bw_gb_s:.0f} GB/s ({bw_util:.0f}%), "
                  f"Tok/s={tokens_per_sec:.0f}, "
                  f"P99={p99:.2f}ms "
                  f"[{bound}]")
            render_roofline_live(
                results,
                roofline_points,
                ridge,
                hw_specs["peak_tflops_bf16"],
                hw_specs["hbm_bandwidth_gb_s"],
                os.path.join(output_dir, "live"),
                len(results),
                live,
            )

    print(f"\n{'='*70}")
    print("ROOFLINE SUMMARY")
    print(f"{'='*70}")
    print(f"{'Type':>8} {'N':>6} {'AI':>8} {'TFLOPs':>8} {'MFU%':>6} "
          f"{'HFU%':>6} {'BW GB/s':>8} {'Tok/s':>10} {'P99ms':>8} {'Bound':>12}")
    print("-" * 90)
    for r in results:
        bound = "MEM" if r["is_memory_bound"] else "COMPUTE"
        print(f"{r['type']:>8} {r['seq_len']:>6} {r['arithmetic_intensity']:>7.1f} "
              f"{r['achieved_tflops']:>7.4f} {r['mfu_pct']:>5.1f} "
              f"{r['hfu_pct']:>5.1f} {r['bandwidth_gb_s']:>7.0f} "
              f"{r['tokens_per_sec']:>9.0f} {r['latency_p99_ms']:>7.2f} "
              f"{bound:>12}")

    try:
        from sparse_attention.visualize import plot_roofline
        plot_roofline(
            roofline_points,
            peak_tflops=hw_specs["peak_tflops_bf16"],
            peak_bw_gb_s=hw_specs["hbm_bandwidth_gb_s"],
            output_path=os.path.join(output_dir, "plots", "roofline.png"),
        )
    except Exception as e:
        print(f"  [WARN] Could not generate roofline plot: {e}")

    summary = {
        "hardware": hw_specs,
        "results": results,
        "roofline_points": roofline_points,
    }
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "roofline_analysis.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nResults saved to {path}")

    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Roofline Model Analysis")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-pallas", action="store_true")
    args = parser.parse_args()

    seq_lens = [256, 512, 1024, 2048] if args.quick else [256, 512, 1024, 2048, 4096]
    roofline_analysis(seq_lengths=seq_lens, use_pallas=not args.no_pallas)

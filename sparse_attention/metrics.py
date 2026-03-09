from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from sparse_attention.masks import BlockMask
from sparse_attention.kernel import compute_theoretical_flops, compute_theoretical_hbm_bytes


@dataclass
class BenchmarkResult:
    name: str
    attention_type: str
    batch_size: int
    seq_len: int
    n_heads: int
    d_head: int
    block_size: int = 0
    sparsity_ratio: float = 0.0

    latency_mean_ms: float = 0.0
    latency_std_ms: float = 0.0
    latency_min_ms: float = 0.0
    latency_max_ms: float = 0.0

    latency_p50_ms: float = 0.0
    latency_p90_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_p99_ms: float = 0.0
    latency_jitter_ms: float = 0.0
    latency_cv: float = 0.0
    all_latencies: List[float] = field(default_factory=list)

    tokens_per_second: float = 0.0

    theoretical_hbm_bytes: int = 0
    theoretical_hbm_mb: float = 0.0
    peak_memory_bytes: int = 0
    peak_memory_mb: float = 0.0

    theoretical_flops: int = 0
    theoretical_tflops: float = 0.0
    achieved_tflops: float = 0.0

    theoretical_bandwidth_gb_s: float = 0.0

    mfu_pct: float = 0.0

    hfu_pct: float = 0.0

    arithmetic_intensity: float = 0.0

    bandwidth_utilization_pct: float = 0.0

    memory_efficiency_pct: float = 0.0

    def summary_dict(self) -> dict:
        return {
            "Type": self.attention_type,
            "B": self.batch_size,
            "N": self.seq_len,
            "Sparsity": f"{self.sparsity_ratio:.1%}",
            "Latency (ms)": f"{self.latency_mean_ms:.2f} ± {self.latency_std_ms:.2f}",
            "P99 (ms)": f"{self.latency_p99_ms:.2f}",
            "Tokens/s": f"{self.tokens_per_second:.0f}",
            "HBM (MB)": f"{self.theoretical_hbm_mb:.1f}",
            "TFLOPs": f"{self.theoretical_tflops:.4f}",
            "Achieved TFLOPs": f"{self.achieved_tflops:.4f}",
            "MFU%": f"{self.mfu_pct:.1f}",
            "HFU%": f"{self.hfu_pct:.1f}",
            "AI (F/B)": f"{self.arithmetic_intensity:.1f}",
            "BW (GB/s)": f"{self.theoretical_bandwidth_gb_s:.1f}",
        }

    def full_dict(self) -> dict:
        return {
            "name": self.name,
            "type": self.attention_type,
            "batch_size": self.batch_size,
            "seq_len": self.seq_len,
            "n_heads": self.n_heads,
            "d_head": self.d_head,
            "block_size": self.block_size,
            "sparsity_ratio": self.sparsity_ratio,
            "latency_mean_ms": self.latency_mean_ms,
            "latency_std_ms": self.latency_std_ms,
            "latency_min_ms": self.latency_min_ms,
            "latency_max_ms": self.latency_max_ms,
            "latency_p50_ms": self.latency_p50_ms,
            "latency_p90_ms": self.latency_p90_ms,
            "latency_p95_ms": self.latency_p95_ms,
            "latency_p99_ms": self.latency_p99_ms,
            "latency_jitter_ms": self.latency_jitter_ms,
            "latency_cv": self.latency_cv,
            "tokens_per_second": self.tokens_per_second,
            "theoretical_hbm_bytes": self.theoretical_hbm_bytes,
            "theoretical_hbm_mb": self.theoretical_hbm_mb,
            "peak_memory_mb": self.peak_memory_mb,
            "theoretical_flops": self.theoretical_flops,
            "theoretical_tflops": self.theoretical_tflops,
            "achieved_tflops": self.achieved_tflops,
            "mfu_pct": self.mfu_pct,
            "hfu_pct": self.hfu_pct,
            "arithmetic_intensity": self.arithmetic_intensity,
            "bandwidth_gb_s": self.theoretical_bandwidth_gb_s,
            "bandwidth_utilization_pct": self.bandwidth_utilization_pct,
            "memory_efficiency_pct": self.memory_efficiency_pct,
        }


@dataclass
class ComparisonResult:
    sparse: BenchmarkResult
    dense: BenchmarkResult

    @property
    def hbm_reduction_pct(self) -> float:
        if self.dense.theoretical_hbm_bytes == 0:
            return 0.0
        return (
            (self.dense.theoretical_hbm_bytes - self.sparse.theoretical_hbm_bytes)
            / self.dense.theoretical_hbm_bytes
            * 100
        )

    @property
    def speedup(self) -> float:
        if self.sparse.latency_mean_ms == 0:
            return 0.0
        return self.dense.latency_mean_ms / self.sparse.latency_mean_ms

    @property
    def flops_reduction_pct(self) -> float:
        if self.dense.theoretical_flops == 0:
            return 0.0
        return (
            (self.dense.theoretical_flops - self.sparse.theoretical_flops)
            / self.dense.theoretical_flops
            * 100
        )

    @property
    def memory_efficiency_gain(self) -> float:
        if self.dense.memory_efficiency_pct == 0:
            return 0.0
        return self.sparse.memory_efficiency_pct - self.dense.memory_efficiency_pct

    def summary_dict(self) -> dict:
        return {
            "B": self.sparse.batch_size,
            "N": self.sparse.seq_len,
            "Sparsity": f"{self.sparse.sparsity_ratio:.1%}",
            "Dense Latency (ms)": f"{self.dense.latency_mean_ms:.2f}",
            "Sparse Latency (ms)": f"{self.sparse.latency_mean_ms:.2f}",
            "Dense P99 (ms)": f"{self.dense.latency_p99_ms:.2f}",
            "Sparse P99 (ms)": f"{self.sparse.latency_p99_ms:.2f}",
            "Speedup": f"{self.speedup:.2f}x",
            "Dense Tok/s": f"{self.dense.tokens_per_second:.0f}",
            "Sparse Tok/s": f"{self.sparse.tokens_per_second:.0f}",
            "HBM Reduction": f"{self.hbm_reduction_pct:.1f}%",
            "FLOPs Reduction": f"{self.flops_reduction_pct:.1f}%",
            "Dense HBM (MB)": f"{self.dense.theoretical_hbm_mb:.1f}",
            "Sparse HBM (MB)": f"{self.sparse.theoretical_hbm_mb:.1f}",
            "Dense MFU%": f"{self.dense.mfu_pct:.1f}",
            "Sparse MFU%": f"{self.sparse.mfu_pct:.1f}",
        }


def time_function(
    fn: Callable,
    *args,
    n_warmup: int = 5,
    n_iterations: int = 20,
    **kwargs,
) -> Tuple[float, float, float, float, List[float]]:
    for _ in range(n_warmup):
        result = fn(*args, **kwargs)
        if isinstance(result, jnp.ndarray):
            result.block_until_ready()
        elif isinstance(result, (tuple, list)):
            for r in result:
                if isinstance(r, jnp.ndarray):
                    r.block_until_ready()

    times = []
    for _ in range(n_iterations):
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        if isinstance(result, jnp.ndarray):
            result.block_until_ready()
        elif isinstance(result, (tuple, list)):
            for r in result:
                if isinstance(r, jnp.ndarray):
                    r.block_until_ready()
        end = time.perf_counter()
        times.append((end - start) * 1000)

    times = np.array(times)
    return float(times.mean()), float(times.std()), float(times.min()), float(times.max()), times.tolist()


def benchmark_attention(
    attention_fn: Callable,
    query: jnp.ndarray,
    key: jnp.ndarray,
    value: jnp.ndarray,
    name: str,
    attention_type: str,
    block_mask: Optional[BlockMask] = None,
    n_warmup: int = 5,
    n_iterations: int = 20,
    **attn_kwargs,
) -> BenchmarkResult:
    B, N, H, D = query.shape
    PEAK_TFLOPS = 197.0
    PEAK_BW_GB_S = 820.0

    mean_ms, std_ms, min_ms, max_ms, all_times = time_function(
        attention_fn, query, key, value,
        n_warmup=n_warmup, n_iterations=n_iterations,
        **attn_kwargs,
    )

    times_arr = np.array(all_times)
    p50 = float(np.percentile(times_arr, 50))
    p90 = float(np.percentile(times_arr, 90))
    p95 = float(np.percentile(times_arr, 95))
    p99 = float(np.percentile(times_arr, 99))
    jitter = float(max_ms - min_ms)
    cv = float(std_ms / mean_ms) if mean_ms > 0 else 0

    flops_info = compute_theoretical_flops(B, N, H, D, block_mask)

    hbm_info = compute_theoretical_hbm_bytes(B, N, H, D, block_mask)

    total_tokens = B * N
    tokens_per_sec = total_tokens / (mean_ms / 1000) if mean_ms > 0 else 0

    latency_s = mean_ms / 1000
    achieved_tflops = (
        flops_info["total_flops"] / latency_s / 1e12
        if latency_s > 0 else 0
    )

    bw_gb_s = (
        hbm_info["total_bytes"] / latency_s / 1e9
        if latency_s > 0 else 0
    )

    ai = (
        flops_info["total_flops"] / hbm_info["total_bytes"]
        if hbm_info["total_bytes"] > 0 else 0
    )

    theoretical_tflops_rate = flops_info["total_flops"] / latency_s / 1e12 if latency_s > 0 else 0
    mfu = (achieved_tflops / theoretical_tflops_rate * 100) if theoretical_tflops_rate > 0 else 0

    hfu = (achieved_tflops / PEAK_TFLOPS * 100) if PEAK_TFLOPS > 0 else 0

    bw_util = (bw_gb_s / PEAK_BW_GB_S * 100) if PEAK_BW_GB_S > 0 else 0

    useful_bytes = 4 * B * N * H * D * 2
    mem_eff = (useful_bytes / hbm_info["total_bytes"] * 100
               if hbm_info["total_bytes"] > 0 else 0)

    peak_mem = 0
    try:
        devices = jax.local_devices()
        if devices and hasattr(devices[0], 'memory_stats'):
            stats = devices[0].memory_stats()
            if stats:
                peak_mem = stats.get('peak_bytes_in_use', 0)
    except Exception:
        pass

    return BenchmarkResult(
        name=name,
        attention_type=attention_type,
        batch_size=B,
        seq_len=N,
        n_heads=H,
        d_head=D,
        block_size=block_mask.block_size if block_mask else 0,
        sparsity_ratio=block_mask.sparsity_ratio if block_mask else 0.0,
        latency_mean_ms=mean_ms,
        latency_std_ms=std_ms,
        latency_min_ms=min_ms,
        latency_max_ms=max_ms,
        latency_p50_ms=p50,
        latency_p90_ms=p90,
        latency_p95_ms=p95,
        latency_p99_ms=p99,
        latency_jitter_ms=jitter,
        latency_cv=cv,
        all_latencies=all_times,
        tokens_per_second=tokens_per_sec,
        theoretical_hbm_bytes=hbm_info["total_bytes"],
        theoretical_hbm_mb=hbm_info["total_mb"],
        peak_memory_bytes=peak_mem,
        peak_memory_mb=peak_mem / (1024 * 1024),
        theoretical_flops=flops_info["total_flops"],
        theoretical_tflops=flops_info["total_tflops"],
        achieved_tflops=achieved_tflops,
        theoretical_bandwidth_gb_s=bw_gb_s,
        mfu_pct=mfu,
        hfu_pct=hfu,
        arithmetic_intensity=ai,
        bandwidth_utilization_pct=bw_util,
        memory_efficiency_pct=mem_eff,
    )


def compare_attention(
    sparse_result: BenchmarkResult,
    dense_result: BenchmarkResult,
) -> ComparisonResult:
    return ComparisonResult(sparse=sparse_result, dense=dense_result)


def format_results_table(results: List[BenchmarkResult]) -> str:
    try:
        from tabulate import tabulate
        rows = [r.summary_dict() for r in results]
        return tabulate(rows, headers="keys", tablefmt="grid")
    except ImportError:
        lines = []
        for r in results:
            d = r.summary_dict()
            lines.append(" | ".join(f"{k}: {v}" for k, v in d.items()))
        return "\n".join(lines)


def format_comparison_table(comparisons: List[ComparisonResult]) -> str:
    try:
        from tabulate import tabulate
        rows = [c.summary_dict() for c in comparisons]
        return tabulate(rows, headers="keys", tablefmt="grid")
    except ImportError:
        lines = []
        for c in comparisons:
            d = c.summary_dict()
            lines.append(" | ".join(f"{k}: {v}" for k, v in d.items()))
        return "\n".join(lines)


def print_device_info():
    print("=" * 60)
    print("JAX Device Information")
    print("=" * 60)
    print(f"JAX version:    {jax.__version__}")
    print(f"Devices:        {jax.devices()}")
    print(f"Device count:   {jax.device_count()}")
    print(f"Default backend: {jax.default_backend()}")

    for i, dev in enumerate(jax.local_devices()):
        print(f"\nDevice {i}: {dev}")
        print(f"  Platform:   {dev.platform}")
        print(f"  Device kind: {dev.device_kind}")
        try:
            stats = dev.memory_stats()
            if stats:
                for key, val in stats.items():
                    if 'bytes' in key.lower():
                        print(f"  {key}: {val / 1024**2:.1f} MB")
        except Exception:
            print("  Memory stats: unavailable")
    print("=" * 60)

from __future__ import annotations

import gc
import json
import os
import time
from typing import Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from sparse_attention.kernel import sparse_attention
from sparse_attention.dense_attention import dense_attention
from sparse_attention.masks import create_block_mask
from sparse_attention.data import create_dummy_inputs
from sparse_attention.metrics import time_function


def oom_boundary_test(
    n_heads: int = 8,
    d_head: int = 32,
    block_size: int = 128,
    sparsity_type: str = "combined",
    seq_lengths: List[int] = [256, 512, 1024, 2048, 4096, 8192],
    batch_sizes: List[int] = [1, 2, 4, 8, 16, 32],
    attention_type: str = "sparse",
    use_pallas: bool = True,
    output_dir: str = "benchmark_results",
) -> Dict:
    print(f"\n{'='*70}")
    print(f"STRESS TEST: OOM Boundary Detection ({attention_type})")
    print(f"{'='*70}")
    print(f"Seq lengths: {seq_lengths}")
    print(f"Batch sizes: {batch_sizes}")

    status_matrix = np.zeros((len(seq_lengths), len(batch_sizes)), dtype=int)
    latency_matrix = np.zeros((len(seq_lengths), len(batch_sizes)))
    memory_matrix = np.zeros((len(seq_lengths), len(batch_sizes)))
    results = []

    import subprocess
    import sys

    for i, seq_len in enumerate(seq_lengths):
        for j, batch_size in enumerate(batch_sizes):
            label = f"B={batch_size}, N={seq_len}"
            print(f"  Testing {label}...", end=" ", flush=True)

            try:
                cmd = [
                    sys.executable, "-m", "benchmarks.oom_runner",
                    "--b", str(batch_size),
                    "--n", str(seq_len),
                    "--target", attention_type,
                    "--sparsity", sparsity_type,
                ]
                if not use_pallas:
                    cmd.append("--no-pallas")
                
                start = time.perf_counter()
                
                result = subprocess.run(
                    cmd, 
                    capture_output=True, 
                    text=True, 
                    timeout=30
                )
                
                mean_ms = (time.perf_counter() - start) * 1000
                tokens_per_sec = batch_size * seq_len / (mean_ms / 1000) if mean_ms > 0 else 0

                if result.returncode == 0:
                    status_matrix[i, j] = 1
                    latency_matrix[i, j] = mean_ms
                    print(f"✓ {mean_ms:.1f}ms ({tokens_per_sec:.0f} tok/s) [Subprocess Auto-Recover]")
                    
                    results.append({
                        "batch_size": batch_size,
                        "seq_len": seq_len,
                        "status": "OK",
                        "latency_ms": mean_ms,
                        "tokens_per_sec": tokens_per_sec,
                    })
                else:
                    err = result.stderr.strip()[-100:] if result.stderr else "Subprocess Crash (OOM)"
                    print(f"✗ FAIL: {err.replace(chr(10), ' ')}")
                    status_matrix[i, j] = 0
                    results.append({
                        "batch_size": batch_size,
                        "seq_len": seq_len,
                        "status": "FAIL",
                        "error": err,
                    })

            except subprocess.TimeoutExpired:
                print("✗ TIMEOUT")
                status_matrix[i, j] = 0
                results.append({
                    "batch_size": batch_size,
                    "seq_len": seq_len,
                    "status": "TIMEOUT",
                })
            except Exception as e:
                err = str(e)[:80]
                print(f"✗ FAIL: {err}")
                status_matrix[i, j] = 0
                results.append({
                    "batch_size": batch_size,
                    "seq_len": seq_len,
                    "status": "FAIL",
                    "error": str(e)[:200],
                })

    max_tokens = 0
    max_config = {}
    for r in results:
        if r["status"] == "OK":
            total = r["batch_size"] * r["seq_len"]
            if total > max_tokens:
                max_tokens = total
                max_config = r

    print(f"\n{'='*70}")
    print(f"OOM BOUNDARY ({attention_type})")
    print(f"{'='*70}")
    print(f"  Max successful: B={max_config.get('batch_size')}, "
          f"N={max_config.get('seq_len')} "
          f"({max_tokens:,} tokens)")
    if max_config.get("peak_memory_mb"):
        print(f"  Peak memory:    {max_config.get('peak_memory_mb', 0):.1f} MB")

    summary = {
        "attention_type": attention_type,
        "seq_lengths": seq_lengths,
        "batch_sizes": batch_sizes,
        "status_matrix": status_matrix.tolist(),
        "latency_matrix": latency_matrix.tolist(),
        "memory_matrix": memory_matrix.tolist(),
        "max_tokens": max_tokens,
        "max_config": max_config,
        "results": results,
    }

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"stress_test_{attention_type}.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Results saved to {path}")

    return summary


def sustained_throughput_test(
    batch_size: int = 4,
    seq_len: int = 1024,
    n_heads: int = 8,
    d_head: int = 32,
    block_size: int = 128,
    sparsity_type: str = "combined",
    duration_seconds: float = 30.0,
    use_pallas: bool = True,
    output_dir: str = "benchmark_results",
) -> Dict:
    print(f"\n{'='*70}")
    print(f"SUSTAINED THROUGHPUT TEST")
    print(f"{'='*70}")
    print(f"Config: B={batch_size}, N={seq_len}, duration={duration_seconds}s")

    q, k, v = create_dummy_inputs(batch_size, seq_len, n_heads, d_head)
    block_mask = create_block_mask(seq_len, block_size, sparsity_type)

    print("  Warmup...", end=" ", flush=True)
    for _ in range(5):
        out = sparse_attention(q, k, v, block_mask, use_pallas=use_pallas)
        out.block_until_ready()
    print("done")

    print(f"  Running for {duration_seconds}s...", end=" ", flush=True)
    throughputs = []
    latencies = []
    tokens_per_iter = batch_size * seq_len
    start_time = time.perf_counter()
    iteration = 0

    while time.perf_counter() - start_time < duration_seconds:
        iter_start = time.perf_counter()
        out = sparse_attention(q, k, v, block_mask, use_pallas=use_pallas)
        out.block_until_ready()
        iter_end = time.perf_counter()

        iter_ms = (iter_end - iter_start) * 1000
        latencies.append(iter_ms)
        throughputs.append(tokens_per_iter / (iter_ms / 1000))
        iteration += 1

    total_time = time.perf_counter() - start_time
    print(f"done ({iteration} iterations)")

    latencies = np.array(latencies)
    throughputs = np.array(throughputs)

    summary = {
        "config": {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "duration_s": duration_seconds,
        },
        "iterations": iteration,
        "total_time_s": total_time,
        "latency": {
            "mean_ms": float(latencies.mean()),
            "std_ms": float(latencies.std()),
            "p50_ms": float(np.percentile(latencies, 50)),
            "p90_ms": float(np.percentile(latencies, 90)),
            "p95_ms": float(np.percentile(latencies, 95)),
            "p99_ms": float(np.percentile(latencies, 99)),
            "min_ms": float(latencies.min()),
            "max_ms": float(latencies.max()),
            "jitter_ms": float(latencies.max() - latencies.min()),
            "cv": float(latencies.std() / latencies.mean()) if latencies.mean() > 0 else 0,
        },
        "throughput": {
            "mean_tokens_per_sec": float(throughputs.mean()),
            "std_tokens_per_sec": float(throughputs.std()),
            "min_tokens_per_sec": float(throughputs.min()),
            "max_tokens_per_sec": float(throughputs.max()),
        },
        "time_series": {
            "latencies_ms": latencies.tolist(),
            "throughputs": throughputs.tolist(),
        },
    }

    print(f"\n  Results:")
    print(f"    Latency:    {summary['latency']['mean_ms']:.2f} ± "
          f"{summary['latency']['std_ms']:.2f} ms")
    print(f"    P50/P90/P99: {summary['latency']['p50_ms']:.2f} / "
          f"{summary['latency']['p90_ms']:.2f} / "
          f"{summary['latency']['p99_ms']:.2f} ms")
    print(f"    Jitter:     {summary['latency']['jitter_ms']:.2f} ms "
          f"(CV={summary['latency']['cv']:.3f})")
    print(f"    Throughput: {summary['throughput']['mean_tokens_per_sec']:.0f} ± "
          f"{summary['throughput']['std_tokens_per_sec']:.0f} tok/s")

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "sustained_throughput.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"    Saved to {path}")

    return summary


def numerical_stability_test(
    seq_len: int = 1024,
    n_heads: int = 8,
    d_head: int = 32,
    block_size: int = 128,
    sparsity_type: str = "combined",
) -> Dict:
    print(f"\n{'='*70}")
    print("NUMERICAL STABILITY TEST")
    print(f"{'='*70}")

    block_mask = create_block_mask(seq_len, block_size, sparsity_type)
    batch_size = 1
    rng = jax.random.PRNGKey(42)
    shape = (batch_size, seq_len, n_heads, d_head)
    results = {}

    scenarios = {
        "normal": lambda k: jax.random.normal(k, shape) * 0.1,
        "large_values": lambda k: jax.random.normal(k, shape) * 100.0,
        "small_values": lambda k: jax.random.normal(k, shape) * 1e-6,
        "mixed": lambda k: jax.random.normal(k, shape) * jax.random.choice(k, jnp.array([0.001, 100.0]), shape),
        "uniform": lambda k: jnp.ones(shape) * 0.5,
        "near_zero": lambda k: jax.random.normal(k, shape) * 1e-10,
    }

    for name, gen_fn in scenarios.items():
        print(f"  Testing '{name}'...", end=" ", flush=True)
        try:
            k1, k2, k3, rng = jax.random.split(rng, 4)
            q = gen_fn(k1).astype(jnp.float32)
            k = gen_fn(k2).astype(jnp.float32)
            v = gen_fn(k3).astype(jnp.float32)

            from sparse_attention.kernel import sparse_attention_jax
            out = sparse_attention_jax(q, k, v, block_mask)
            out.block_until_ready()

            has_nan = bool(jnp.any(jnp.isnan(out)))
            has_inf = bool(jnp.any(jnp.isinf(out)))
            max_val = float(jnp.max(jnp.abs(out)))

            status = "PASS" if not has_nan and not has_inf else "FAIL"
            results[name] = {
                "status": status,
                "has_nan": has_nan,
                "has_inf": has_inf,
                "max_abs_output": max_val,
            }
            symbol = "✓" if status == "PASS" else "✗"
            print(f"{symbol} (max_abs={max_val:.4g}, nan={has_nan}, inf={has_inf})")

        except Exception as e:
            results[name] = {"status": "ERROR", "error": str(e)[:100]}
            print(f"✗ ERROR: {str(e)[:60]}")

    passed = sum(1 for r in results.values() if r["status"] == "PASS")
    total = len(results)
    print(f"\n  Result: {passed}/{total} scenarios passed")

    return results


def compilation_time_test(
    seq_lengths: List[int] = [256, 512, 1024, 2048],
    batch_size: int = 1,
    n_heads: int = 8,
    d_head: int = 32,
    block_size: int = 128,
    sparsity_type: str = "combined",
) -> Dict:
    print(f"\n{'='*70}")
    print("COMPILATION TIME ANALYSIS")
    print(f"{'='*70}")

    results = {}
    for seq_len in seq_lengths:
        print(f"  N={seq_len}...", end=" ", flush=True)

        jax.clear_caches()
        gc.collect()

        q, k, v = create_dummy_inputs(batch_size, seq_len, n_heads, d_head)
        block_mask = create_block_mask(seq_len, block_size, sparsity_type)

        start = time.perf_counter()
        out = sparse_attention(q, k, v, block_mask, use_pallas=False)
        out.block_until_ready()
        first_call_ms = (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        out = sparse_attention(q, k, v, block_mask, use_pallas=False)
        out.block_until_ready()
        cached_call_ms = (time.perf_counter() - start) * 1000

        compile_overhead_ms = max(0, first_call_ms - cached_call_ms)
        results[seq_len] = {
            "first_call_ms": first_call_ms,
            "cached_call_ms": cached_call_ms,
            "compile_overhead_ms": compile_overhead_ms,
        }
        print(f"compile={compile_overhead_ms:.0f}ms, "
              f"cached={cached_call_ms:.2f}ms")

    return results


def run_all_stress_tests(
    output_dir: str = "benchmark_results",
    use_pallas: bool = True,
    quick: bool = False,
) -> Dict:
    all_results = {}

    seq_lens = [256, 512, 1024, 2048] if quick else [256, 512, 1024, 2048, 4096, 8192]
    batch_szs = [1, 2, 4, 8] if quick else [1, 2, 4, 8, 16, 32]

    all_results["oom_sparse"] = oom_boundary_test(
        seq_lengths=seq_lens, batch_sizes=batch_szs,
        attention_type="sparse", use_pallas=use_pallas, output_dir=output_dir,
    )
    all_results["oom_dense"] = oom_boundary_test(
        seq_lengths=seq_lens, batch_sizes=batch_szs,
        attention_type="dense", output_dir=output_dir,
    )

    duration = 10.0 if quick else 30.0
    all_results["sustained"] = sustained_throughput_test(
        duration_seconds=duration, use_pallas=use_pallas, output_dir=output_dir,
    )

    all_results["numerical"] = numerical_stability_test()

    all_results["compilation"] = compilation_time_test()

    try:
        from sparse_attention.visualize import plot_stress_results
        sparse_oom = all_results["oom_sparse"]
        status = np.array(sparse_oom["status_matrix"])
        latency = np.array(sparse_oom["latency_matrix"])
        plot_stress_results(
            sparse_oom["seq_lengths"], sparse_oom["batch_sizes"],
            status, latency,
            output_path=os.path.join(output_dir, "plots", "stress_test_sparse.png"),
        )
        dense_oom = all_results["oom_dense"]
        status_d = np.array(dense_oom["status_matrix"])
        latency_d = np.array(dense_oom["latency_matrix"])
        plot_stress_results(
            dense_oom["seq_lengths"], dense_oom["batch_sizes"],
            status_d, latency_d,
            output_path=os.path.join(output_dir, "plots", "stress_test_dense.png"),
        )
    except Exception as e:
        print(f"  [WARN] Could not generate stress test plots: {e}")

    path = os.path.join(output_dir, "stress_test_all.json")
    with open(path, "w") as f:
        json.dump({
            k: (v if not isinstance(v, dict) or "time_series" not in v.get("", {})
                 else {kk: vv for kk, vv in v.items() if kk != "time_series"})
            for k, v in all_results.items()
        }, f, indent=2, default=str)
    print(f"\nAll stress test results saved to {path}")

    return all_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Stress Tests")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-pallas", action="store_true")
    args = parser.parse_args()
    run_all_stress_tests(quick=args.quick, use_pallas=not args.no_pallas)

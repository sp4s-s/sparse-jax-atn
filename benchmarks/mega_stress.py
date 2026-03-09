from __future__ import annotations

import gc
import json
import os
import time
from typing import Dict, List, Tuple

import numpy as np

try:
    import psutil
except ImportError:
    psutil = None

from sparse_attention.runtime_telemetry import BenchmarkTelemetry


def _import_jax():
    import jax
    import jax.numpy as jnp

    return jax, jnp


def _true_available_gb() -> float:
    available_gb = 100.0
    if psutil is not None:
        available_gb = psutil.virtual_memory().available / (1024 ** 3)
    try:
        if os.path.exists("/sys/fs/cgroup/memory.max"):
            with open("/sys/fs/cgroup/memory.max", encoding="utf-8") as handle:
                raw_limit = handle.read().strip()
            if raw_limit != "max":
                with open("/sys/fs/cgroup/memory.current", encoding="utf-8") as handle:
                    used = int(handle.read().strip())
                available_gb = min(available_gb, (int(raw_limit) - used) / (1024 ** 3))
        elif os.path.exists("/sys/fs/cgroup/memory/memory.limit_in_bytes"):
            with open("/sys/fs/cgroup/memory/memory.limit_in_bytes", encoding="utf-8") as handle:
                raw_limit = int(handle.read().strip())
            if raw_limit < 100 * 1024 ** 3:
                with open("/sys/fs/cgroup/memory/memory.usage_in_bytes", encoding="utf-8") as handle:
                    used = int(handle.read().strip())
                available_gb = min(available_gb, (raw_limit - used) / (1024 ** 3))
    except Exception:
        pass
    return max(0.0, available_gb)


def _sleep_for_ui() -> None:
    time.sleep(0.02)


def _write_json(path: str, payload: Dict | List) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)


def host_ram_saturation(
    target_gb: float = 40.0,
    output_dir: str = "results",
    telemetry: BenchmarkTelemetry | None = None,
    use_pallas: bool = True,
) -> Dict:
    jax, jnp = _import_jax()
    from sparse_attention.kernel import sparse_attention
    from sparse_attention.masks import create_block_mask

    stage = "host_ram"
    chunk_mb = 64
    chunk_bytes = chunk_mb * 1024 * 1024
    n_elements = chunk_bytes // np.dtype(np.float16).itemsize
    safety_headroom_gb = 4.5
    buffers: List[np.ndarray] = []
    allocated_gb = 0.0
    target_gb = float(target_gb)
    next_report_gb = 0.5
    report_interval_s = 1.0
    last_report_at = time.perf_counter()

    if telemetry is not None:
        telemetry.start_stage(
            stage,
            total=target_gb,
            unit="GB",
            message=f"ramping host RAM toward {target_gb:.1f}GB with {chunk_mb}MB chunks",
        )

    try:
        while allocated_gb + (chunk_mb / 1024) <= target_gb:
            available_gb = _true_available_gb()
            if available_gb <= safety_headroom_gb:
                break
            buffers.append(np.full(n_elements, 1.0, dtype=np.float16))
            allocated_gb += chunk_mb / 1024
            now = time.perf_counter()
            should_report = allocated_gb >= next_report_gb or (now - last_report_at) >= report_interval_s
            if telemetry is not None and should_report:
                telemetry.update_stage(
                    stage,
                    progress=allocated_gb,
                    message="host buffers staged safely",
                    metrics={
                        "allocated_gb": allocated_gb,
                        "available_gb": available_gb,
                        "safety_headroom_gb": safety_headroom_gb,
                    },
                )
                next_report_gb = allocated_gb + 0.5
                last_report_at = now
            if int(allocated_gb * 1024) % 512 == 0:
                _sleep_for_ui()

        block_mask = create_block_mask(2048, 128, "combined")
        shape = (4, 2048, 8, 64)
        rng = jax.random.PRNGKey(0)
        k1, k2, k3 = jax.random.split(rng, 3)
        q = jax.random.normal(k1, shape, dtype=jnp.bfloat16)
        k = jax.random.normal(k2, shape, dtype=jnp.bfloat16)
        v = jax.random.normal(k3, shape, dtype=jnp.bfloat16)

        started = time.perf_counter()
        out = sparse_attention(q, k, v, block_mask, use_pallas=use_pallas)
        out.block_until_ready()
        latency_ms = (time.perf_counter() - started) * 1000.0

        result = {
            "status": "PASS" if allocated_gb >= target_gb * 0.85 else "PARTIAL",
            "target_gb": target_gb,
            "achieved_gb": round(allocated_gb, 2),
            "attention_under_pressure_ms": round(latency_ms, 2),
            "available_after_gb": round(_true_available_gb(), 2),
            "chunk_mb": chunk_mb,
            "safety_headroom_gb": safety_headroom_gb,
        }
        if telemetry is not None:
            telemetry.end_stage(stage, status=result["status"], message="host pressure sweep finished")
            telemetry.record_summary(stage, result)
    except KeyboardInterrupt:
        result = {
            "status": "INTERRUPTED",
            "target_gb": target_gb,
            "achieved_gb": round(allocated_gb, 2),
        }
        if telemetry is not None:
            telemetry.end_stage(stage, status="interrupted", message="host pressure sweep interrupted")
            telemetry.record_summary(stage, result)
        raise
    except Exception as exc:
        result = {
            "status": "FAIL",
            "target_gb": target_gb,
            "achieved_gb": round(allocated_gb, 2),
            "error": str(exc)[:240],
        }
        if telemetry is not None:
            telemetry.end_stage(stage, status="fail", message=str(exc)[:120])
            telemetry.record_summary(stage, result)
    finally:
        del buffers
        gc.collect()

    _write_json(os.path.join(output_dir, "mega_stress_host_ram.json"), result)
    return result


def progressive_oom_ceiling(
    output_dir: str = "results",
    use_pallas: bool = True,
    telemetry: BenchmarkTelemetry | None = None,
) -> Dict:
    jax, jnp = _import_jax()
    from sparse_attention.dense_attention import dense_attention
    from sparse_attention.kernel import sparse_attention
    from sparse_attention.masks import create_block_mask

    configs: List[Tuple[int, int, int, int]] = [
        (1, 1024, 8, 64),
        (2, 2048, 8, 64),
        (4, 4096, 8, 64),
        (8, 4096, 8, 64),
        (8, 8192, 8, 64),
        (8, 8192, 16, 64),
        (16, 8192, 16, 64),
        (16, 16384, 16, 64),
        (32, 16384, 16, 64),
        (16, 32768, 16, 64),
    ]
    results = {"sparse": [], "dense": []}
    total_steps = len(configs) * 2
    processed = 0

    if telemetry is not None:
        telemetry.start_stage("ceiling", total=float(total_steps), unit="cfg", message="finding the stable OOM ceiling")

    for attention_type in ("sparse", "dense"):
        for idx, (batch_size, seq_len, n_heads, d_head) in enumerate(configs):
            estimated_gb = (3 * batch_size * seq_len * n_heads * d_head * 2) / 1e9
            if attention_type == "dense":
                estimated_gb += (batch_size * n_heads * seq_len * seq_len * 2) / 1e9
            try:
                shape = (batch_size, seq_len, n_heads, d_head)
                rng = jax.random.PRNGKey(idx)
                k1, k2, k3 = jax.random.split(rng, 3)
                q = jax.random.normal(k1, shape, dtype=jnp.bfloat16)
                k = jax.random.normal(k2, shape, dtype=jnp.bfloat16)
                v = jax.random.normal(k3, shape, dtype=jnp.bfloat16)
                block_mask = create_block_mask(seq_len, 128, "combined")
                started = time.perf_counter()
                if attention_type == "sparse":
                    out = sparse_attention(q, k, v, block_mask, use_pallas=use_pallas)
                else:
                    out = dense_attention(q, k, v, causal=True)
                out.block_until_ready()
                latency_ms = (time.perf_counter() - started) * 1000.0
                tokens_per_sec = batch_size * seq_len / (latency_ms / 1000.0)
                row = {
                    "config": [batch_size, seq_len, n_heads, d_head],
                    "est_gb": round(estimated_gb, 2),
                    "status": "PASS",
                    "latency_ms": round(latency_ms, 2),
                    "tok_s": round(tokens_per_sec, 2),
                }
                results[attention_type].append(row)
                processed += 1
                if telemetry is not None:
                    telemetry.update_stage(
                        "ceiling",
                        progress=float(processed),
                        message=f"{attention_type} passed at B={batch_size} N={seq_len}",
                        metrics={
                            "estimated_gb": estimated_gb,
                            "latency_ms": latency_ms,
                            "tok_s": tokens_per_sec,
                        },
                    )
            except Exception as exc:
                row = {
                    "config": [batch_size, seq_len, n_heads, d_head],
                    "est_gb": round(estimated_gb, 2),
                    "status": "OOM",
                    "error": str(exc)[:160],
                }
                results[attention_type].append(row)
                processed += 1
                if telemetry is not None:
                    telemetry.update_stage(
                        "ceiling",
                        progress=float(processed),
                        message=f"{attention_type} stopped at B={batch_size} N={seq_len}",
                        metrics={"estimated_gb": estimated_gb},
                    )
                gc.collect()
                break
            gc.collect()
            _sleep_for_ui()

    if telemetry is not None:
        telemetry.end_stage("ceiling", status="done", message="ceiling analysis finished")
        telemetry.record_summary("ceiling", results)
    _write_json(os.path.join(output_dir, "mega_stress_ceiling.json"), results)
    return results


def gradient_accumulation_stress(
    output_dir: str = "results",
    use_pallas: bool = True,
    telemetry: BenchmarkTelemetry | None = None,
) -> List[Dict]:
    jax, jnp = _import_jax()
    from sparse_attention.kernel import sparse_attention
    from sparse_attention.masks import create_block_mask

    configs = [
        (4, 4096, 16),
        (8, 2048, 32),
        (8, 4096, 32),
        (16, 2048, 64),
    ]
    results: List[Dict] = []
    if telemetry is not None:
        telemetry.start_stage("accumulation", total=float(len(configs)), unit="jobs", message="testing effective batch scaling")

    for job_idx, (micro_batch, seq_len, steps) in enumerate(configs, start=1):
        try:
            mask = create_block_mask(seq_len, 128, "combined")
            started = time.perf_counter()
            micro_latencies = []
            for step in range(steps):
                shape = (micro_batch, seq_len, 8, 64)
                rng = jax.random.PRNGKey(step)
                k1, k2, k3 = jax.random.split(rng, 3)
                q = jax.random.normal(k1, shape, dtype=jnp.bfloat16)
                k = jax.random.normal(k2, shape, dtype=jnp.bfloat16)
                v = jax.random.normal(k3, shape, dtype=jnp.bfloat16)
                iter_started = time.perf_counter()
                out = sparse_attention(q, k, v, mask, use_pallas=use_pallas)
                out.block_until_ready()
                micro_latencies.append((time.perf_counter() - iter_started) * 1000.0)
                if telemetry is not None and step % max(1, steps // 6) == 0:
                    tokens_done = micro_batch * seq_len * (step + 1)
                    telemetry.update_stage(
                        "accumulation",
                        progress=float(job_idx - 1) + (step + 1) / steps,
                        message=f"micro_batch={micro_batch}, seq_len={seq_len}, step={step + 1}/{steps}",
                        metrics={
                            "tokens_done": float(tokens_done),
                            "latency_ms": micro_latencies[-1],
                        },
                    )
                _sleep_for_ui()
            elapsed = time.perf_counter() - started
            total_tokens = micro_batch * seq_len * steps
            result = {
                "micro_batch": micro_batch,
                "seq_len": seq_len,
                "n_steps": steps,
                "effective_batch": micro_batch * steps,
                "total_tokens": total_tokens,
                "elapsed_s": round(elapsed, 2),
                "tok_s": round(total_tokens / elapsed, 2),
                "mean_latency_ms": round(sum(micro_latencies) / len(micro_latencies), 2),
                "jitter_ms": round(max(micro_latencies) - min(micro_latencies), 2),
                "status": "PASS",
            }
        except Exception as exc:
            result = {
                "micro_batch": micro_batch,
                "seq_len": seq_len,
                "effective_batch": micro_batch * steps,
                "status": "FAIL",
                "error": str(exc)[:200],
            }
        results.append(result)

    if telemetry is not None:
        telemetry.end_stage("accumulation", status="done", message="gradient accumulation sweep finished")
        telemetry.record_summary("accumulation", {"results": results})
    _write_json(os.path.join(output_dir, "mega_stress_accumulation.json"), results)
    return results


def mixed_pressure_test(
    target_host_gb: float = 30.0,
    output_dir: str = "results",
    use_pallas: bool = True,
    telemetry: BenchmarkTelemetry | None = None,
) -> Dict:
    jax, jnp = _import_jax()
    from sparse_attention.kernel import sparse_attention
    from sparse_attention.masks import create_block_mask

    dataset: List[np.ndarray] = []
    chunk_mb = 64
    n_elements = (chunk_mb * 1024 * 1024) // np.dtype(np.float16).itemsize
    target_host_gb = max(1.0, float(target_host_gb))
    host_gb = 0.0
    next_report_gb = 0.5
    report_interval_s = 1.0
    last_report_at = time.perf_counter()

    if telemetry is not None:
        telemetry.start_stage("mixed", total=target_host_gb, unit="GB", message="holding host dataset while kernels run")

    try:
        while host_gb + (chunk_mb / 1024) <= target_host_gb:
            available_gb = _true_available_gb()
            if available_gb <= 5.0:
                break
            dataset.append(np.full(n_elements, 1.0, dtype=np.float16))
            host_gb += chunk_mb / 1024
            now = time.perf_counter()
            should_report = host_gb >= next_report_gb or (now - last_report_at) >= report_interval_s
            if telemetry is not None and should_report:
                telemetry.update_stage(
                    "mixed",
                    progress=host_gb,
                    message="dataset residency increasing",
                    metrics={"host_dataset_gb": host_gb, "available_gb": available_gb},
                )
                next_report_gb = host_gb + 0.5
                last_report_at = now
            if int(host_gb * 1024) % 512 == 0:
                _sleep_for_ui()

        batch_size, seq_len = 8, 4096
        mask = create_block_mask(seq_len, 128, "combined")
        latencies = []
        for step in range(20):
            shape = (batch_size, seq_len, 8, 64)
            rng = jax.random.PRNGKey(step)
            k1, k2, k3 = jax.random.split(rng, 3)
            q = jax.random.normal(k1, shape, dtype=jnp.bfloat16)
            k = jax.random.normal(k2, shape, dtype=jnp.bfloat16)
            v = jax.random.normal(k3, shape, dtype=jnp.bfloat16)
            started = time.perf_counter()
            out = sparse_attention(q, k, v, mask, use_pallas=use_pallas)
            out.block_until_ready()
            latency_ms = (time.perf_counter() - started) * 1000.0
            latencies.append(latency_ms)
            if telemetry is not None:
                telemetry.update_stage(
                    "mixed",
                    progress=host_gb,
                    message=f"device iteration {step + 1}/20 under mixed pressure",
                    metrics={
                        "latency_ms": latency_ms,
                        "tok_s": (batch_size * seq_len) / (latency_ms / 1000.0),
                    },
                )
            _sleep_for_ui()

        result = {
            "status": "PASS",
            "host_gb": round(host_gb, 2),
            "avg_ms": round(sum(latencies) / len(latencies), 2),
            "p99_ms": round(sorted(latencies)[min(len(latencies) - 1, int(len(latencies) * 0.99))], 2),
            "jitter_ms": round(max(latencies) - min(latencies), 2),
            "tok_s": round((batch_size * seq_len) / ((sum(latencies) / len(latencies)) / 1000.0), 2),
        }
    except Exception as exc:
        result = {"status": "FAIL", "host_gb": round(host_gb, 2), "error": str(exc)[:200]}
    finally:
        del dataset
        gc.collect()

    if telemetry is not None:
        telemetry.end_stage("mixed", status=result["status"].lower(), message="mixed host and device pressure finished")
        telemetry.record_summary("mixed", result)
    _write_json(os.path.join(output_dir, "mega_stress_mixed.json"), result)
    return result


def run_mega_stress(output_dir: str = "results", use_pallas: bool = True, target_ram_gb: float = 40.0) -> Dict:
    os.makedirs(output_dir, exist_ok=True)
    telemetry = BenchmarkTelemetry(
        run_name="mega_stress",
        output_dir=output_dir,
        config={"target_ram_gb": target_ram_gb, "use_pallas": use_pallas},
    )
    all_results: Dict[str, Dict | List] = {}
    try:
        print(f"Mega stress suite is live. Artifacts will be saved under {output_dir}.", flush=True)
        print(f"TensorBoard logdir: {telemetry.tensorboard_dir}", flush=True)
        print(f"Live run notes: {telemetry.links_path}", flush=True)
        print("In Colab, open TensorBoard from a new cell with:", flush=True)
        print(f"%load_ext tensorboard", flush=True)
        print(f"%tensorboard --logdir {telemetry.tensorboard_dir}", flush=True)
        all_results["host_ram"] = host_ram_saturation(target_ram_gb, output_dir, telemetry, use_pallas)
        all_results["ceiling"] = progressive_oom_ceiling(output_dir, use_pallas, telemetry)
        all_results["accumulation"] = gradient_accumulation_stress(output_dir, use_pallas, telemetry)
        all_results["mixed"] = mixed_pressure_test(min(target_ram_gb - 10.0, 32.0), output_dir, use_pallas, telemetry)
        _write_json(os.path.join(output_dir, "mega_stress_all.json"), all_results)
        telemetry.record_summary("final", all_results)
        return all_results
    finally:
        telemetry.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Mega Stress Test")
    parser.add_argument("--target-gb", type=float, default=40.0)
    parser.add_argument("--no-pallas", action="store_true")
    parser.add_argument("--output-dir", type=str, default="results")
    args = parser.parse_args()
    run_mega_stress(args.output_dir, not args.no_pallas, args.target_gb)

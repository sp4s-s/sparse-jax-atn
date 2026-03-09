import pytest
import jax
import jax.numpy as jnp

from sparse_attention.metrics import (
    time_function,
    benchmark_attention,
    BenchmarkResult,
    ComparisonResult,
    compare_attention,
    format_results_table,
    format_comparison_table,
)
from sparse_attention.dense_attention import dense_attention
from sparse_attention.kernel import sparse_attention_jax
from sparse_attention.masks import causal_block_mask
from sparse_attention.data import create_dummy_inputs


class TestTimeFunction:

    def test_basic_timing(self):
        def dummy_fn(x):
            return x + 1

        x = jnp.ones(100)
        mean, std, min_t, max_t, all_times = time_function(
            dummy_fn, x, n_warmup=2, n_iterations=5
        )

        assert mean >= 0
        assert std >= 0
        assert min_t <= mean
        assert max_t >= mean
        assert len(all_times) == 5


class TestBenchmarkResult:

    def test_summary_dict(self):
        result = BenchmarkResult(
            name="test", attention_type="dense",
            batch_size=1, seq_len=1024, n_heads=8, d_head=32,
            latency_mean_ms=10.0, latency_std_ms=1.0,
        )
        d = result.summary_dict()
        assert "Type" in d
        assert "Latency (ms)" in d
        assert d["Type"] == "dense"


class TestComparisonResult:

    def test_hbm_reduction(self):
        sparse_r = BenchmarkResult(
            name="sparse", attention_type="sparse",
            batch_size=1, seq_len=1024, n_heads=8, d_head=32,
            theoretical_hbm_bytes=6_000_000,
        )
        dense_r = BenchmarkResult(
            name="dense", attention_type="dense",
            batch_size=1, seq_len=1024, n_heads=8, d_head=32,
            theoretical_hbm_bytes=10_000_000,
        )
        comp = compare_attention(sparse_r, dense_r)
        assert abs(comp.hbm_reduction_pct - 40.0) < 0.1

    def test_speedup(self):
        sparse_r = BenchmarkResult(
            name="sparse", attention_type="sparse",
            batch_size=1, seq_len=1024, n_heads=8, d_head=32,
            latency_mean_ms=5.0,
        )
        dense_r = BenchmarkResult(
            name="dense", attention_type="dense",
            batch_size=1, seq_len=1024, n_heads=8, d_head=32,
            latency_mean_ms=10.0,
        )
        comp = compare_attention(sparse_r, dense_r)
        assert abs(comp.speedup - 2.0) < 0.01


class TestFormatting:

    def test_format_results_table(self):
        results = [
            BenchmarkResult(
                name="test", attention_type="dense",
                batch_size=1, seq_len=1024, n_heads=8, d_head=32,
                latency_mean_ms=10.0, latency_std_ms=1.0,
            )
        ]
        table = format_results_table(results)
        assert isinstance(table, str)
        assert len(table) > 0

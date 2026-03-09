import pytest
import jax
import jax.numpy as jnp
import numpy as np

from sparse_attention.kernel import (
    sparse_attention,
    sparse_attention_jax,
    compute_theoretical_flops,
    compute_theoretical_hbm_bytes,
)
from sparse_attention.dense_attention import dense_attention, dense_attention_with_mask
from sparse_attention.masks import (
    causal_block_mask,
    strided_block_mask,
    combined_block_mask,
    create_block_mask,
)
from sparse_attention.data import create_dummy_inputs


TEST_DTYPE = jnp.float32
ATOL = 1e-2
RTOL = 1e-2


def make_test_inputs(batch_size=2, seq_len=256, n_heads=4, d_head=32, seed=42):
    rng = jax.random.PRNGKey(seed)
    k1, k2, k3 = jax.random.split(rng, 3)
    shape = (batch_size, seq_len, n_heads, d_head)
    q = jax.random.normal(k1, shape, dtype=TEST_DTYPE) * 0.1
    k = jax.random.normal(k2, shape, dtype=TEST_DTYPE) * 0.1
    v = jax.random.normal(k3, shape, dtype=TEST_DTYPE) * 0.1
    return q, k, v


class TestSparseVsDense:

    def test_causal_mask_correctness(self):
        seq_len = 256
        block_size = 64
        q, k, v = make_test_inputs(seq_len=seq_len)
        block_mask = causal_block_mask(seq_len, block_size)

        sparse_out = sparse_attention_jax(q, k, v, block_mask)

        dense_mask = block_mask.dense_mask
        dense_out = dense_attention_with_mask(q, k, v, dense_mask)

        np.testing.assert_allclose(
            np.array(sparse_out),
            np.array(dense_out),
            atol=ATOL, rtol=RTOL,
        )

    def test_combined_mask_correctness(self):
        seq_len = 256
        block_size = 64
        q, k, v = make_test_inputs(seq_len=seq_len)
        block_mask = combined_block_mask(seq_len, block_size)

        sparse_out = sparse_attention_jax(q, k, v, block_mask)
        dense_mask = block_mask.dense_mask
        dense_out = dense_attention_with_mask(q, k, v, dense_mask)

        np.testing.assert_allclose(
            np.array(sparse_out),
            np.array(dense_out),
            atol=ATOL, rtol=RTOL,
        )

    def test_output_shape(self):
        seq_len = 256
        q, k, v = make_test_inputs(seq_len=seq_len)
        block_mask = causal_block_mask(seq_len, 64)

        out = sparse_attention_jax(q, k, v, block_mask)
        assert out.shape == q.shape

    def test_output_not_nan(self):
        seq_len = 256
        q, k, v = make_test_inputs(seq_len=seq_len)
        block_mask = causal_block_mask(seq_len, 64)

        out = sparse_attention_jax(q, k, v, block_mask)
        assert not jnp.any(jnp.isnan(out))

    def test_output_not_inf(self):
        seq_len = 256
        q, k, v = make_test_inputs(seq_len=seq_len)
        block_mask = causal_block_mask(seq_len, 64)

        out = sparse_attention_jax(q, k, v, block_mask)
        assert not jnp.any(jnp.isinf(out))

    def test_batch_independence(self):
        seq_len = 128
        q, k, v = make_test_inputs(batch_size=2, seq_len=seq_len)
        block_mask = causal_block_mask(seq_len, 64)

        full_out = sparse_attention_jax(q, k, v, block_mask)

        out1 = sparse_attention_jax(q[0:1], k[0:1], v[0:1], block_mask)
        out2 = sparse_attention_jax(q[1:2], k[1:2], v[1:2], block_mask)

        np.testing.assert_allclose(np.array(full_out[0]), np.array(out1[0]),
                                    atol=1e-5)
        np.testing.assert_allclose(np.array(full_out[1]), np.array(out2[0]),
                                    atol=1e-5)


class TestGradients:

    def test_gradient_exists(self):
        seq_len = 128
        q, k, v = make_test_inputs(batch_size=1, seq_len=seq_len)
        block_mask = causal_block_mask(seq_len, 64)

        def loss_fn(q, k, v):
            out = sparse_attention_jax(q, k, v, block_mask)
            return jnp.mean(out ** 2)

        grads = jax.grad(loss_fn, argnums=(0, 1, 2))(q, k, v)

        for i, g in enumerate(grads):
            assert g.shape == q.shape, f"Gradient {i} shape mismatch"
            assert not jnp.any(jnp.isnan(g)), f"Gradient {i} contains NaN"
            assert jnp.any(g != 0), f"Gradient {i} is all zeros"

    def test_gradient_magnitudes_reasonable(self):
        seq_len = 128
        q, k, v = make_test_inputs(batch_size=1, seq_len=seq_len)
        block_mask = causal_block_mask(seq_len, 64)

        def loss_fn(q, k, v):
            out = sparse_attention_jax(q, k, v, block_mask)
            return jnp.mean(out)

        grads = jax.grad(loss_fn, argnums=(0, 1, 2))(q, k, v)

        for i, g in enumerate(grads):
            max_abs = jnp.max(jnp.abs(g))
            assert max_abs < 100, f"Gradient {i} max abs {max_abs} is too large"


class TestTheoreticalMetrics:

    def test_sparse_fewer_flops_than_dense(self):
        block_mask = combined_block_mask(1024, 128)
        dense = compute_theoretical_flops(1, 1024, 8, 32)
        sparse = compute_theoretical_flops(1, 1024, 8, 32, block_mask)
        assert sparse["total_flops"] < dense["total_flops"]

    def test_sparse_fewer_hbm_bytes_than_dense(self):
        block_mask = combined_block_mask(1024, 128)
        dense = compute_theoretical_hbm_bytes(1, 1024, 8, 32)
        sparse = compute_theoretical_hbm_bytes(1, 1024, 8, 32, block_mask)
        assert sparse["total_bytes"] < dense["total_bytes"]

    def test_hbm_reduction_matches_sparsity(self):
        block_mask = combined_block_mask(2048, 128)
        dense = compute_theoretical_hbm_bytes(4, 2048, 8, 32)
        sparse = compute_theoretical_hbm_bytes(4, 2048, 8, 32, block_mask)

        reduction = (dense["total_bytes"] - sparse["total_bytes"]) / dense["total_bytes"]
        assert reduction > 0.1, f"HBM reduction {reduction:.1%} too small"


class TestDispatch:

    def test_dispatch_non_tpu(self):
        seq_len = 128
        q, k, v = make_test_inputs(batch_size=1, seq_len=seq_len)
        block_mask = causal_block_mask(seq_len, 64)

        out = sparse_attention(q, k, v, block_mask, use_pallas=False)
        assert out.shape == q.shape

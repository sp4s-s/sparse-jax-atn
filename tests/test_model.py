import pytest
import jax
import jax.numpy as jnp
import numpy as np

from sparse_attention.model import (
    SparseTransformer,
    create_model,
    init_model,
    count_parameters,
)
from sparse_attention.masks import causal_block_mask


class TestSparseTransformer:

    def test_forward_sparse(self):
        block_mask = causal_block_mask(128, 64)
        model = create_model(
            attention_type="sparse",
            block_mask=block_mask,
            use_pallas=False,
            d_model=64, n_heads=4, n_layers=2, d_ff=128,
            max_seq_len=128, vocab_size=1000,
        )
        rng = jax.random.PRNGKey(0)
        variables = init_model(model, rng, batch_size=2, seq_len=128)

        input_ids = jnp.ones((2, 128), dtype=jnp.int32)
        logits = model.apply(variables, input_ids, deterministic=True)

        assert logits.shape == (2, 128, 1000)
        assert not jnp.any(jnp.isnan(logits))

    def test_forward_dense(self):
        model = create_model(
            attention_type="dense",
            use_pallas=False,
            d_model=64, n_heads=4, n_layers=2, d_ff=128,
            max_seq_len=128, vocab_size=1000,
        )
        rng = jax.random.PRNGKey(0)
        variables = init_model(model, rng, batch_size=2, seq_len=128)

        input_ids = jnp.ones((2, 128), dtype=jnp.int32)
        logits = model.apply(variables, input_ids, deterministic=True)

        assert logits.shape == (2, 128, 1000)

    def test_parameter_count(self):
        model = create_model(
            attention_type="dense",
            d_model=256, n_heads=8, n_layers=4, d_ff=1024,
            max_seq_len=1024, vocab_size=50257,
        )
        rng = jax.random.PRNGKey(0)
        variables = init_model(model, rng, batch_size=1, seq_len=64)
        n_params = count_parameters(variables["params"])
        assert n_params > 1_000_000
        assert n_params < 100_000_000

    def test_gradient_flow(self):
        block_mask = causal_block_mask(64, 32)
        model = create_model(
            attention_type="sparse",
            block_mask=block_mask,
            use_pallas=False,
            d_model=32, n_heads=2, n_layers=1, d_ff=64,
            max_seq_len=64, vocab_size=100,
        )
        rng = jax.random.PRNGKey(0)
        variables = init_model(model, rng, batch_size=1, seq_len=64)

        input_ids = jax.random.randint(rng, (1, 64), 0, 100)

        def loss_fn(params):
            logits = model.apply({"params": params}, input_ids,
                                deterministic=True)
            return jnp.mean(logits)

        grads = jax.grad(loss_fn)(variables["params"])
        grad_leaves = jax.tree_util.tree_leaves(grads)
        assert all(not jnp.any(jnp.isnan(g)) for g in grad_leaves)
        assert any(jnp.any(g != 0) for g in grad_leaves)

    def test_different_seq_lengths(self):
        model = create_model(
            attention_type="dense",
            d_model=64, n_heads=4, n_layers=1, d_ff=128,
            max_seq_len=256, vocab_size=1000,
        )
        rng = jax.random.PRNGKey(0)
        variables = init_model(model, rng, batch_size=1, seq_len=64)

        for seq_len in [32, 64, 128]:
            input_ids = jnp.ones((1, seq_len), dtype=jnp.int32)
            logits = model.apply(variables, input_ids, deterministic=True)
            assert logits.shape == (1, seq_len, 1000)

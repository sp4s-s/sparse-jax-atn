import pytest
import jax.numpy as jnp
import numpy as np

from sparse_attention.masks import (
    causal_block_mask,
    strided_block_mask,
    fixed_block_mask,
    random_block_mask,
    combined_block_mask,
    create_block_mask,
    BlockMask,
)


class TestCausalBlockMask:

    def test_shape(self):
        mask = causal_block_mask(seq_len=512, block_size=128)
        assert mask.mask.shape == (4, 4)

    def test_lower_triangular(self):
        mask = causal_block_mask(seq_len=512, block_size=128)
        n = mask.num_q_blocks
        for i in range(n):
            for j in range(n):
                if j > i:
                    assert not mask.mask[i, j], f"Position ({i},{j}) should be False"
                else:
                    assert mask.mask[i, j], f"Position ({i},{j}) should be True"

    def test_diagonal_active(self):
        mask = causal_block_mask(seq_len=1024, block_size=128)
        n = mask.num_q_blocks
        for i in range(n):
            assert mask.mask[i, i], f"Diagonal block ({i},{i}) should be active"

    def test_sparsity_ratio(self):
        mask = causal_block_mask(seq_len=1024, block_size=128)
        n = mask.num_q_blocks
        expected_active = n * (n + 1) // 2
        expected_sparsity = 1.0 - expected_active / (n * n)
        assert abs(mask.sparsity_ratio - expected_sparsity) < 1e-6

    def test_dtype(self):
        mask = causal_block_mask(seq_len=512, block_size=128)
        assert mask.mask.dtype == jnp.bool_


class TestStridedBlockMask:

    def test_shape(self):
        mask = strided_block_mask(seq_len=1024, block_size=128)
        assert mask.mask.shape == (8, 8)

    def test_local_window(self):
        mask = strided_block_mask(seq_len=1024, block_size=128,
                                   local_window_blocks=2)
        n = mask.num_q_blocks
        for i in range(n):
            for j in range(n):
                if abs(i - j) <= 2:
                    assert mask.mask[i, j], (
                        f"Local window block ({i},{j}) should be active"
                    )

    def test_global_stride(self):
        stride = 4
        mask = strided_block_mask(seq_len=1024, block_size=128,
                                   global_stride=stride)
        n = mask.num_q_blocks
        for i in range(n):
            for j in range(n):
                if j % stride == 0:
                    assert mask.mask[i, j], (
                        f"Global column {j} should be active for row {i}"
                    )

    def test_has_some_sparsity(self):
        mask = strided_block_mask(seq_len=2048, block_size=128)
        assert mask.sparsity_ratio > 0.1


class TestFixedBlockMask:

    def test_first_and_last_columns_active(self):
        mask = fixed_block_mask(seq_len=1024, block_size=128)
        n = mask.num_q_blocks
        for i in range(n):
            assert mask.mask[i, 0], f"First column should be active for row {i}"
            assert mask.mask[i, n-1], f"Last column should be active for row {i}"


class TestRandomBlockMask:

    def test_diagonal_always_active(self):
        mask = random_block_mask(seq_len=1024, block_size=128, density=0.1)
        n = mask.num_q_blocks
        for i in range(n):
            assert mask.mask[i, i], f"Diagonal ({i},{i}) should be active"

    def test_density_approximate(self):
        mask = random_block_mask(seq_len=2048, block_size=128, density=0.3)
        assert 0.5 < mask.sparsity_ratio < 0.85

    def test_reproducibility(self):
        m1 = random_block_mask(seq_len=1024, block_size=128, seed=42)
        m2 = random_block_mask(seq_len=1024, block_size=128, seed=42)
        assert jnp.array_equal(m1.mask, m2.mask)


class TestCombinedBlockMask:

    def test_is_subset_of_causal(self):
        causal = causal_block_mask(seq_len=1024, block_size=128)
        combined = combined_block_mask(seq_len=1024, block_size=128)
        assert jnp.all(~combined.mask | causal.mask)

    def test_sparser_than_causal(self):
        causal = causal_block_mask(seq_len=2048, block_size=128)
        combined = combined_block_mask(seq_len=2048, block_size=128)
        assert combined.sparsity_ratio >= causal.sparsity_ratio


class TestDenseMask:

    def test_dense_mask_shape(self):
        mask = causal_block_mask(seq_len=512, block_size=128)
        dense = mask.dense_mask
        assert dense.shape == (512, 512)

    def test_dense_mask_values(self):
        mask = causal_block_mask(seq_len=256, block_size=128)
        dense = mask.dense_mask
        assert dense[0, 128] == 0.0
        assert dense[128, 0] == 1.0


class TestCreateBlockMask:

    def test_all_patterns(self):
        for pattern in ["causal", "strided", "fixed", "random", "combined"]:
            mask = create_block_mask(1024, 128, pattern)
            assert isinstance(mask, BlockMask)
            assert mask.pattern_name in [pattern, f"random(d=0.50)"]

    def test_invalid_pattern(self):
        with pytest.raises(ValueError):
            create_block_mask(1024, 128, "nonexistent")


class TestBlockMaskProperties:

    def test_summary(self):
        mask = causal_block_mask(seq_len=512, block_size=128)
        s = mask.summary()
        assert isinstance(s, str)
        assert len(s) > 0
        assert "causal" in s

    def test_num_active_blocks(self):
        mask = causal_block_mask(seq_len=512, block_size=128)
        total = mask.num_total_blocks
        active = mask.num_active_blocks
        expected_sparsity = 1.0 - active / total
        assert abs(mask.sparsity_ratio - expected_sparsity) < 1e-6

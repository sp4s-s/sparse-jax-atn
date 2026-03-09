from sparse_attention.kernel import sparse_attention, sparse_attention_jax
from sparse_attention.dense_attention import dense_attention
from sparse_attention.masks import (
    create_block_mask,
    BlockMask,
    causal_block_mask,
    strided_block_mask,
    fixed_block_mask,
    random_block_mask,
    combined_block_mask,
)

__version__ = "1.0.0"

__all__ = [
    "sparse_attention",
    "sparse_attention_jax",
    "dense_attention",
    "create_block_mask",
    "BlockMask",
    "causal_block_mask",
    "strided_block_mask",
    "fixed_block_mask",
    "random_block_mask",
    "combined_block_mask",
]

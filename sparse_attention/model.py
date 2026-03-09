from __future__ import annotations
import math
from typing import Literal, Optional
import flax.linen as nn
import jax
import jax.numpy as jnp

from sparse_attention.kernel import sparse_attention, sparse_attention_jax
from sparse_attention.dense_attention import dense_attention
from sparse_attention.masks import BlockMask, create_block_mask

class MultiHeadAttention(nn.Module):
    n_heads: int
    d_head: int
    attention_type: str = "sparse"
    block_mask: Optional[BlockMask] = None
    use_pallas: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        B, N, D = x.shape
        d_model = self.n_heads * self.d_head

        q = nn.Dense(d_model, name="query")(x)
        k = nn.Dense(d_model, name="key")(x)
        v = nn.Dense(d_model, name="value")(x)

        q = q.reshape(B, N, self.n_heads, self.d_head)
        k = k.reshape(B, N, self.n_heads, self.d_head)
        v = v.reshape(B, N, self.n_heads, self.d_head)

        if self.attention_type == "sparse" and self.block_mask is not None:
            attn_out = sparse_attention(q, k, v, block_mask=self.block_mask, use_pallas=self.use_pallas)
        else:
            attn_out = dense_attention(q, k, v, causal=True)

        attn_out = attn_out.reshape(B, N, d_model)
        return nn.Dense(d_model, name="output")(attn_out)

class TransformerBlock(nn.Module):
    n_heads: int
    d_head: int
    d_ff: int
    dropout_rate: float = 0.1
    attention_type: str = "sparse"
    block_mask: Optional[BlockMask] = None
    use_pallas: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        d_model = self.n_heads * self.d_head

        res = x
        x = nn.LayerNorm(name="attn_norm")(x)
        x = MultiHeadAttention(
            n_heads=self.n_heads, d_head=self.d_head, attention_type=self.attention_type,
            block_mask=self.block_mask, use_pallas=self.use_pallas, name="attention"
        )(x, deterministic=deterministic)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
        x = x + res

        res = x
        x = nn.LayerNorm(name="ffn_norm")(x)
        x = nn.Dense(self.d_ff, name="ffn_up")(x)
        x = nn.gelu(x)
        x = nn.Dense(d_model, name="ffn_down")(x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
        x = x + res
        return x

class SparseTransformer(nn.Module):
    vocab_size: int = 50257
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 4
    d_ff: int = 1024
    max_seq_len: int = 2048
    dropout_rate: float = 0.1
    attention_type: str = "sparse"
    block_mask: Optional[BlockMask] = None
    use_pallas: bool = True

    @nn.compact
    def __call__(self, input_ids: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        B, N = input_ids.shape
        d_head = self.d_model // self.n_heads

        x = nn.Embed(num_embeddings=self.vocab_size, features=self.d_model, name="token_embed")(input_ids)
        pos = self.param("pos_embed", nn.initializers.normal(stddev=0.02), (1, self.max_seq_len, self.d_model))
        x = x + pos[:, :N, :]
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)

        for i in range(self.n_layers):
            x = TransformerBlock(
                n_heads=self.n_heads, d_head=d_head, d_ff=self.d_ff, dropout_rate=self.dropout_rate,
                attention_type=self.attention_type, block_mask=self.block_mask,
                use_pallas=self.use_pallas, name=f"block_{i}"
            )(x, deterministic=deterministic)

        x = nn.LayerNorm(name="final_norm")(x)
        return nn.Dense(self.vocab_size, name="lm_head")(x)

def create_model(**kwargs):
    return SparseTransformer(**kwargs)

def init_model(model, rng, batch_size=1, seq_len=128):
    dummy = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
    return model.init(rng, dummy, deterministic=True)

def count_parameters(params):
    return sum(p.size for p in jax.tree_util.tree_leaves(params))

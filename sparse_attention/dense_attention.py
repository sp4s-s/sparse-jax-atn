from __future__ import annotations

import math
from typing import Optional

import jax
import jax.numpy as jnp


def dense_attention(
    query: jnp.ndarray,
    key: jnp.ndarray,
    value: jnp.ndarray,
    causal: bool = False,
    scale: Optional[float] = None,
) -> jnp.ndarray:
    B, N, H, D = query.shape
    if scale is None:
        scale = 1.0 / math.sqrt(D)

    q = jnp.transpose(query, (0, 2, 1, 3))
    k = jnp.transpose(key, (0, 2, 1, 3))
    v = jnp.transpose(value, (0, 2, 1, 3))

    scores = jnp.matmul(q, jnp.transpose(k, (0, 1, 3, 2))) * scale

    if causal:
        mask = jnp.tril(jnp.ones((N, N), dtype=jnp.bool_))
        scores = jnp.where(mask[None, None, :, :], scores, jnp.finfo(scores.dtype).min)

    attn_weights = jax.nn.softmax(scores, axis=-1)

    output = jnp.matmul(attn_weights, v)

    output = jnp.transpose(output, (0, 2, 1, 3))
    return output


def dense_attention_with_mask(
    query: jnp.ndarray,
    key: jnp.ndarray,
    value: jnp.ndarray,
    mask: jnp.ndarray,
    scale: Optional[float] = None,
) -> jnp.ndarray:
    B, N, H, D = query.shape
    if scale is None:
        scale = 1.0 / math.sqrt(D)

    q = jnp.transpose(query, (0, 2, 1, 3))
    k = jnp.transpose(key, (0, 2, 1, 3))
    v = jnp.transpose(value, (0, 2, 1, 3))

    scores = jnp.matmul(q, jnp.transpose(k, (0, 1, 3, 2))) * scale

    if mask.ndim == 2:
        mask = mask[None, None, :, :]

    scores = jnp.where(mask, scores, jnp.finfo(scores.dtype).min)
    attn_weights = jax.nn.softmax(scores, axis=-1)
    attn_weights = jnp.where(mask, attn_weights, 0.0)

    output = jnp.matmul(attn_weights, v)
    output = jnp.transpose(output, (0, 2, 1, 3))
    return output

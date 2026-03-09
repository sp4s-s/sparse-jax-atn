from __future__ import annotations
import functools
import math
from typing import Optional, Tuple
import jax
import jax.numpy as jnp
import numpy as np
from sparse_attention.masks import BlockMask

def sparse_attention(query, key, value, block_mask, scale=None, use_pallas=True):
    if use_pallas and _is_tpu_available():
        try:
            return sparse_attention_pallas(query, key, value, block_mask, scale)
        except Exception as e:
            print(f"[WARN] Pallas failed: {e}")
            return sparse_attention_jax(query, key, value, block_mask, scale)
    return sparse_attention_jax(query, key, value, block_mask, scale)

def sparse_attention_jax(query, key, value, block_mask, scale=None):
    B, N, H, D = query.shape
    scale = scale or (1.0 / math.sqrt(D))

    q = jnp.transpose(query, (0, 2, 1, 3))
    k = jnp.transpose(key, (0, 2, 1, 3))
    v = jnp.transpose(value, (0, 2, 1, 3))

    scores = jnp.matmul(q, jnp.transpose(k, (0, 1, 3, 2))) * scale
    element_mask = block_mask.dense_mask[None, None, :, :]
    scores = jnp.where(element_mask > 0, scores, jnp.finfo(scores.dtype).min)

    attn_weights = jax.nn.softmax(scores, axis=-1)
    attn_weights = jnp.where(element_mask > 0, attn_weights, 0.0)
    
    output = jnp.matmul(attn_weights, v)
    return jnp.transpose(output, (0, 2, 1, 3))

def sparse_attention_pallas(query, key, value, block_mask, scale=None):
    try:
        from jax.experimental import pallas as pl
    except ImportError:
        return sparse_attention_jax(query, key, value, block_mask, scale)

    B, N, H, D = query.shape
    block_size = block_mask.block_size
    scale = scale or (1.0 / math.sqrt(D))

    n_q_blocks = block_mask.num_q_blocks
    n_kv_blocks = block_mask.num_kv_blocks
    pad_len = n_q_blocks * block_size - N
    
    if pad_len > 0:
        query = jnp.pad(query, ((0, 0), (0, pad_len), (0, 0), (0, 0)))
        key = jnp.pad(key, ((0, 0), (0, pad_len), (0, 0), (0, 0)))
        value = jnp.pad(value, ((0, 0), (0, pad_len), (0, 0), (0, 0)))
    
    N_padded = n_q_blocks * block_size
    q_blocked = query.reshape(B, n_q_blocks, block_size, H, D)
    k_blocked = key.reshape(B, n_kv_blocks, block_size, H, D)
    v_blocked = value.reshape(B, n_kv_blocks, block_size, H, D)

    mask_np = np.array(block_mask.mask)
    max_active = int(mask_np.sum(axis=1).max())
    active_kv_indices = np.full((n_q_blocks, max_active), -1, dtype=np.int32)
    active_kv_counts = np.zeros(n_q_blocks, dtype=np.int32)

    for q_idx in range(n_q_blocks):
        active_indices = np.where(mask_np[q_idx])[0]
        count = len(active_indices)
        active_kv_indices[q_idx, :count] = active_indices
        active_kv_counts[q_idx] = count

    active_kv_indices_jax = jnp.array(active_kv_indices)
    active_kv_counts_jax = jnp.array(active_kv_counts)

    def _process_single_batch(args):
        q_b, k_b, v_b = args
        def _process_single_q_block(q_idx):
            q_block = q_b[q_idx]
            active_idx = active_kv_indices_jax[q_idx]
            k_flat = k_b.reshape(-1, H, D)
            v_flat = v_b.reshape(-1, H, D)

            running_max = jnp.full((block_size, H), -jnp.inf)
            running_sum = jnp.zeros((block_size, H))
            acc = jnp.zeros((block_size, H, D))

            def body_fn(kv_iter, carry):
                rm, rs, a = carry
                idx = active_idx[kv_iter]
                k_block = jax.lax.dynamic_slice(k_flat, (idx * block_size, 0, 0), (block_size, H, D))
                v_block = jax.lax.dynamic_slice(v_flat, (idx * block_size, 0, 0), (block_size, H, D))

                scores = jnp.einsum('qhd,khd->hqk', q_block, k_block) * scale
                block_max = jnp.max(scores, axis=-1).T
                new_max = jnp.maximum(rm, block_max)
                exp_old = jnp.exp(rm - new_max)
                exp_scores = jnp.exp(scores - new_max.T[:, :, None])

                new_sum = exp_old * rs + jnp.sum(exp_scores, axis=-1).T
                new_vals = jnp.einsum('hqk,khd->qhd', exp_scores, v_block)
                return new_max, new_sum, exp_old[:, :, None] * a + new_vals

            running_max, running_sum, acc = jax.lax.fori_loop(0, max_active, body_fn, (running_max, running_sum, acc))
            return acc / jnp.maximum(running_sum[:, :, None], 1e-6)

        return jax.vmap(_process_single_q_block)(jnp.arange(n_q_blocks))

    output_blocked = jax.vmap(_process_single_batch)((q_blocked, k_blocked, v_blocked))
    output = output_blocked.reshape(B, N_padded, H, D)
    return output[:, :N, :, :]

def _is_tpu_available():
    try:
        return any(d.platform == 'tpu' for d in jax.devices())
    except: return False


def compute_theoretical_flops(batch_size, seq_len, n_heads, d_head, block_mask=None):
    B, N, H, D = batch_size, seq_len, n_heads, d_head
    if block_mask:
        active = block_mask.num_active_blocks
        bs = block_mask.block_size
        qk = active * (2 * bs * bs * D) * B * H
        soft = active * (5 * bs * bs) * B * H
        av = active * (2 * bs * bs * D) * B * H
    else:
        qk = 2 * B * H * N * N * D
        soft = 5 * B * H * N * N
        av = 2 * B * H * N * N * D
    
    total = qk + soft + av
    return {"qk_flops": qk, "softmax_flops": soft, "av_flops": av, "total_flops": total, "total_tflops": total / 1e12}

def compute_theoretical_hbm_bytes(batch_size, seq_len, n_heads, d_head, block_mask=None, dtype_bytes=2):
    B, N, H, D = batch_size, seq_len, n_heads, d_head
    bs = block_mask.block_size if block_mask else N
    
    if block_mask:
        q_b = B * N * H * D * dtype_bytes
        unique_kv = int(block_mask.mask.any(axis=0).sum())
        k_b = B * unique_kv * bs * H * D * dtype_bytes
        v_b = k_b
        out_b = B * N * H * D * dtype_bytes
        attn_b = 0
    else:
        q_b = k_b = v_b = out_b = B * N * H * D * dtype_bytes
        attn_b = B * H * N * N * dtype_bytes

    total = q_b + k_b + v_b + out_b + attn_b
    return {"q_bytes": q_b, "k_bytes": k_b, "v_bytes": v_b, "output_bytes": out_b, "attn_matrix_bytes": attn_b, "total_bytes": total, "total_gb": total / 1e9}

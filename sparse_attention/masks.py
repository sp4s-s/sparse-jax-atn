from __future__ import annotations
import functools
from dataclasses import dataclass
from typing import Literal, Optional
import jax
import jax.numpy as jnp
import numpy as np

@dataclass
class BlockMask:
    mask: jnp.ndarray
    block_size: int
    sparsity_ratio: float
    pattern_name: str
    seq_len: int

    @property
    def num_q_blocks(self): return self.mask.shape[0]

    @property
    def num_kv_blocks(self): return self.mask.shape[1]

    @property
    def num_active_blocks(self): return int(jnp.sum(self.mask).item())

    @property
    def num_total_blocks(self): return self.num_q_blocks * self.num_kv_blocks

    @property
    def dense_mask(self):
        return jnp.kron(self.mask.astype(jnp.float32), 
                        jnp.ones((self.block_size, self.block_size)))[:self.seq_len, :self.seq_len]

    def summary(self):
        return (f"BlockMask(pattern={self.pattern_name}, seq_len={self.seq_len}, "
                f"block_size={self.block_size}, blocks={self.num_active_blocks}/{self.num_total_blocks}, "
                f"sparsity={self.sparsity_ratio:.1%})")


def causal_block_mask(seq_len, block_size):
    n = _num_blocks(seq_len, block_size)
    rows, cols = jnp.arange(n)[:, None], jnp.arange(n)[None, :]
    return _make_block_mask(cols <= rows, block_size, "causal", seq_len)

def strided_block_mask(seq_len, block_size, local_window_blocks=3, global_stride=4):
    n = _num_blocks(seq_len, block_size)
    rows, cols = jnp.arange(n)[:, None], jnp.arange(n)[None, :]
    local = jnp.abs(rows - cols) <= local_window_blocks
    glbl = (cols % global_stride) == 0
    return _make_block_mask(local | glbl, block_size, "strided", seq_len)

def fixed_block_mask(seq_len, block_size, local_window_blocks=3):
    n = _num_blocks(seq_len, block_size)
    rows, cols = jnp.arange(n)[:, None], jnp.arange(n)[None, :]
    local = jnp.abs(rows - cols) <= local_window_blocks
    glbl = (cols == 0) | (cols == n - 1)
    return _make_block_mask(local | glbl, block_size, "fixed", seq_len)

def random_block_mask(seq_len, block_size, density=0.5, seed=42):
    n = _num_blocks(seq_len, block_size)
    rng = np.random.RandomState(seed)
    mask = jnp.array(rng.random((n, n)) < density) | jnp.eye(n, dtype=jnp.bool_)
    return _make_block_mask(mask, block_size, f"random(d={density:.2f})", seq_len)

def combined_block_mask(seq_len, block_size, local_window_blocks=3, global_stride=4):
    causal = causal_block_mask(seq_len, block_size)
    strided = strided_block_mask(seq_len, block_size, local_window_blocks, global_stride)
    return _make_block_mask(causal.mask & strided.mask, block_size, "combined", seq_len)

def create_block_mask(seq_len, block_size, pattern="combined", **kwargs):
    factories = {
        "causal": causal_block_mask, "strided": strided_block_mask,
        "fixed": fixed_block_mask, "random": random_block_mask, "combined": combined_block_mask
    }
    if pattern not in factories: raise ValueError(f"Unknown pattern {pattern}")
    return factories[pattern](seq_len, block_size, **kwargs)


def _num_blocks(seq_len, block_size):
    return (seq_len + block_size - 1) // block_size

def _make_block_mask(mask, block_size, pattern_name, seq_len):
    m = mask.astype(jnp.bool_)
    active = int(jnp.sum(m).item())
    sparsity = 1.0 - (active / m.size) if m.size > 0 else 0.0
    return BlockMask(m, block_size, sparsity, pattern_name, seq_len)

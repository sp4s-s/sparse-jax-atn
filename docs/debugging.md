# Debugging & Troubleshooting Guide

Common errors, debugging strategies, and fixes for the Sparse Attention project on Colab TPU.

## Common Errors & Fixes

### 1. "No TPU device found" / JAX shows CPU backend

**Symptom:**
```python
jax.devices()  
# [CpuDevice(id=0)]  ← Wrong! Should show TpuDevice
```

**Fix:**
```python
# Check runtime type
import jax
print(jax.default_backend())  # Should print 'tpu'

# If 'cpu': Go to Runtime → Change runtime type → TPU
# Then restart runtime (Runtime → Restart runtime)
```

**If TPU is selected but still shows CPU:**
```bash
# Reinstall JAX for TPU
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

### 2. Out of Memory (OOM) Errors

**Symptom:**
```
RESOURCE_EXHAUSTED: Out of memory
```

**Fix — reduce dimensions:**
```bash
# Use smaller config
python main.py --quick --batch-size 1 --seq-len 512

# Or in benchmarks
python run_benchmarks.py --quick
```

**Fix — clear JAX cache:**
```python
import jax
jax.clear_caches()
```

**Fix — check what's using memory:**
```python
for dev in jax.local_devices():
    stats = dev.memory_stats()
    if stats:
        print(f"Peak: {stats.get('peak_bytes_in_use', 0) / 1e9:.2f} GB")
        print(f"Current: {stats.get('bytes_in_use', 0) / 1e9:.2f} GB")
```

### 3. Pallas Import Error

**Symptom:**
```
ImportError: cannot import name 'pallas' from 'jax.experimental'
```

**Fix:**
```bash
# Ensure JAX version is recent enough
pip install --upgrade jax[tpu]>=0.4.30

# Verify
python -c "from jax.experimental import pallas; print('Pallas OK')"
```

**Note:** Pallas is only needed on TPU. On CPU/GPU, use `--no-pallas`:
```bash
python main.py --quick --no-pallas
```

### 4. XLA Compilation Timeout

**Symptom:**
```
Slow first run, or timeout during warmup
```

**Explanation:** XLA compiles the JAX program on first call. Large models/seq_lens take longer to compile.

**Fix:**
- Be patient on first run (30-60s for large configs)
- Use `--quick` flag to reduce model size
- Reduce `n_warmup` in benchmarks

### 5. NaN in Outputs

**Symptom:**
```
Output contains NaN values
```

**Debug:**
```python
import jax
jax.config.update("jax_debug_nans", True)
# This will raise an error at the exact operation that produced NaN
```

**Common causes:**
- Very large sequence lengths with bf16 precision
- Numerical overflow in softmax
- Division by zero in normalization

**Fix:**
```python
# Use float32 instead of bfloat16
from config import ProjectConfig
cfg = ProjectConfig()
cfg.model.dtype = "float32"
```

### 6. Test Failures

**Symptom:**
```
FAILED tests/test_kernel.py::TestSparseVsDense::test_causal_mask_correctness
```

**Debug:**
```bash
# Run with verbose output
python -m pytest tests/test_kernel.py -v -s

# Run specific test
python -m pytest tests/test_kernel.py::TestSparseVsDense::test_causal_mask_correctness -v -s
```

**Common cause:** Numerical tolerance — bf16 has limited precision:
```python
# tests compare with atol=1e-2 which is appropriate for bf16
# If using float32, you can use tighter tolerance: atol=1e-4
```

### 7. tiktoken Installation Error

**Symptom:**
```
ImportError: No module named 'tiktoken'
```

**Fix:**
```bash
pip install tiktoken
```

**Alternative (without tiktoken):**
```python
# The code falls back to random tokens:
from sparse_attention.data import create_random_token_batch
tokens = create_random_token_batch(batch_size=4, seq_len=1024)
```

### 8. Block Size / Sequence Length Mismatch

**Symptom:**
```
ValueError: seq_len must be divisible by block_size
```

**Fix:** Ensure seq_len is a multiple of block_size:
```bash
# Good: 1024 / 128 = 8 blocks
python main.py --seq-len 1024

# Bad: 1000 / 128 = 7.8 blocks
python main.py --seq-len 1000  # Will be auto-adjusted
```

---

## Debugging Strategies

### 1. Check JAX Configuration

```python
import jax
print(f"JAX version: {jax.__version__}")
print(f"Backend: {jax.default_backend()}")
print(f"Devices: {jax.devices()}")
print(f"Device count: {jax.device_count()}")
```

### 2. Enable JAX Debug Logging

```python
import jax
jax.config.update("jax_log_compiles", True)   # Log XLA compilations
jax.config.update("jax_debug_nans", True)      # Catch NaN immediately
```

```bash
# Set environment variable for more verbose output
export JAX_LOG_LEVEL=DEBUG
```

### 3. Profile Memory Usage

```python
import jax

def check_memory():
    for i, dev in enumerate(jax.local_devices()):
        try:
            stats = dev.memory_stats()
            if stats:
                print(f"Device {i}:")
                print(f"  Peak:    {stats.get('peak_bytes_in_use', 0)/1e9:.2f} GB")
                print(f"  Current: {stats.get('bytes_in_use', 0)/1e9:.2f} GB")
                print(f"  Limit:   {stats.get('bytes_limit', 0)/1e9:.2f} GB")
        except:
            print(f"Device {i}: Memory stats unavailable")

check_memory()
```

### 4. Test Individual Components

```python
# Test mask generation
from sparse_attention.masks import create_block_mask
mask = create_block_mask(1024, 128, "combined")
print(mask.summary())

# Test kernel
from sparse_attention.kernel import sparse_attention_jax
from sparse_attention.data import create_dummy_inputs
q, k, v = create_dummy_inputs(1, 256, 4, 32, dtype=jax.numpy.float32)
mask = create_block_mask(256, 64, "causal")
out = sparse_attention_jax(q, k, v, mask)
print(f"Output shape: {out.shape}, NaN: {jax.numpy.any(jax.numpy.isnan(out))}")
```

### 5. Verify Correctness

```python
from sparse_attention.kernel import sparse_attention_jax
from sparse_attention.dense_attention import dense_attention_with_mask
from sparse_attention.masks import causal_block_mask
from sparse_attention.data import create_dummy_inputs
import jax.numpy as jnp

# Create inputs
q, k, v = create_dummy_inputs(1, 256, 4, 32, dtype=jnp.float32)
mask = causal_block_mask(256, 64)

# Compare
sparse_out = sparse_attention_jax(q, k, v, mask)
dense_out = dense_attention_with_mask(q, k, v, mask.dense_mask)

diff = jnp.abs(sparse_out - dense_out)
print(f"Max diff: {jnp.max(diff):.6f}")
print(f"Mean diff: {jnp.mean(diff):.6f}")
print(f"Close enough: {jnp.allclose(sparse_out, dense_out, atol=1e-2)}")
```

---

## Colab-Specific Tips

1. **Save results early**: Colab may disconnect. Save benchmark JSON frequently.
2. **Use `--quick` first**: Verify everything works before running full benchmarks.
3. **Monitor runtime**: Check Runtime → Manage sessions for time remaining.
4. **Download results**: `from google.colab import files; files.download('benchmark_results/benchmark_results.json')`
5. **Re-run fast**: After first compilation, subsequent runs are much faster.

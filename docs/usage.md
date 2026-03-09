# Full Usage Guide

Complete guide for running the Sparse Attention Transformer project on Google Colab TPU v5e-1.

## Colab TPU First Rules

Read this before running anything on Colab TPU.

- This project is TPU-only.
- A Colab TPU can be owned by one Python process at a time.
- If the notebook kernel already touched TPU through `jax.devices()`, `jax.default_backend()`, or any JAX TPU array creation, a later `!python ...` command may fail because it runs in a different process.
- Your earlier sequence did exactly that: you imported JAX in the notebook process, queried TPU state, then launched `!python run_benchmarks.py ...` in a subprocess.

### Safe usage patterns

Choose one pattern and stay with it for that runtime.

### Pattern A: CLI subprocess owns TPU

Use this when you want to run `!python ...` commands.

Do:

```python
%cd ./sparse-attention-jax/
!pip install .
```

Then run TPU workloads directly:

```python
!python3 run_benchmarks.py --quick
!python3 run_benchmarks.py --mega-stress --target-gb 40
!python3 run_benchmarks.py --roofline
!python3 train.py --dense --steps 500
```

Do not do this before those commands:

```python
import jax
print(jax.devices())
print(jax.default_backend())
```

If you already ran those lines, restart the Colab runtime before using `!python3 ...`.

### Pattern B: Notebook process owns TPU

Use this when you want to inspect devices and run code interactively.

Do:

```python
%cd ./sparse-attention-jax/
!pip install .

import jax
from train import train
from benchmarks.roofline import roofline_analysis
from benchmarks.mega_stress import run_mega_stress

print(jax.devices())
print(jax.default_backend())
```

Then keep running inside the same notebook process:

```python
train(attention_type="dense", max_steps=500, log_dir="tensorboard_logs")
roofline_analysis(output_dir="results")
run_mega_stress(output_dir="results", target_ram_gb=40.0)
```

Do not switch to `!python3 ...` after that unless you restart the runtime.

### TPU memory inspection

Use this only in Pattern B, because it initializes TPU in the notebook process.

```python
import jax

for device in jax.devices():
    print(device)
    try:
        print(device.memory_stats())
    except Exception as exc:
        print(f"memory_stats unavailable: {exc}")
```

For this project:

- TPU HBM is device memory and is much smaller than host RAM.
- `--target-gb 40` or `45` in mega stress refers to host RAM staging pressure, not TPU HBM.

### Recommended Colab command combinations

Fresh runtime, CLI mode:

```python
%cd ./sparse-attention-jax/
!pip install .
!python3 main.py --quick
!python3 run_benchmarks.py --quick
!python3 run_benchmarks.py --stress
!python3 run_benchmarks.py --mega-stress --target-gb 40
!python3 train.py --dense --steps 500
```

Fresh runtime, notebook mode:

```python
%cd ./sparse-attention-jax/
!pip install .

import jax
from train import train
from benchmarks.roofline import roofline_analysis
from benchmarks.mega_stress import run_mega_stress

print(jax.devices())
print(jax.default_backend())

train(attention_type="dense", max_steps=500, log_dir="tensorboard_logs")
roofline_analysis(output_dir="results")
run_mega_stress(output_dir="results", target_ram_gb=40.0)
```

### TensorBoard in Colab

If a run writes logs under `tensorboard_logs/` or `results/tensorboard/`, open TensorBoard in a new notebook cell:

```python
%load_ext tensorboard
%tensorboard --logdir tensorboard_logs
```

Or for mega stress:

```python
%load_ext tensorboard
%tensorboard --logdir results/tensorboard/mega_stress
```

### Automated pre/post training generation

When you run:

```python
train(attention_type="dense", max_steps=200, log_dir="tensorboard_logs")
```

the training run now also writes:

- `tensorboard_logs/run_*/generation_before_training.txt`
- `tensorboard_logs/run_*/generation_after_training.txt`

Each file contains a `10,000` token sample generated from the same fixed prompt so you can compare model behavior before and after training.

The same training run now also reports and logs:

- validation loss
- perplexity
- token-level accuracy on a fixed held-out batch
- 4-gram repetition score for generated text

So each run gives both:

- qualitative comparison from the saved text files
- quantitative before/after improvement from the evaluation metrics

### Your previous cell sequence

This sequence is fine for setup:

```python
%cd ./sparse-attention-jax/
!pip install .
!cat /sys/kernel/mm/transparent_hugepage/enabled
!mount | grep sysfs
!sudo mount -o remount,rw /sys
!echo always | sudo tee /sys/kernel/mm/transparent_hugepage/enabled
!cat /sys/kernel/mm/transparent_hugepage/enabled
```

This sequence changes the ownership model:

```python
import jax
import jax.numpy as jnp
import sparse_attention

print(f"JAX version: {jax.__version__}")
print(f"Backend: {jax.default_backend()}")
print(f"Devices: {jax.devices()}")
print(f"Package version: {sparse_attention.__version__}")
```

After that, the TPU is typically owned by the notebook kernel. At that point, this may fail:

```python
!python3 run_benchmarks.py --mega-stress --target-gb 40
```

because it starts a second process.

## Table of Contents

1. [Colab Setup](#1-colab-setup)
2. [Installation](#2-installation)
3. [Running the Demo](#3-running-the-demo)
4. [Running Benchmarks](#4-running-benchmarks)
5. [Running Tests](#5-running-tests)
6. [Interpreting Results](#6-interpreting-results)
7. [All Commands Reference](#7-all-commands-reference)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. Colab Setup

### Step-by-step:

1. **Open Google Colab**: Go to [colab.research.google.com](https://colab.research.google.com)
2. **Change Runtime**:
   - Click `Runtime` → `Change runtime type`
   - Select **TPU** under Hardware accelerator
   - Choose **TPU v5e** if available (free tier gets v5e-1)
   - Click **Save**
3. **Verify TPU**:
   ```python
   import jax
   print(jax.devices())
   # Should show: [TpuDevice(...)]
   print(jax.default_backend())
   # Should show: 'tpu'
   ```

### Upload Project:

```python
# Option A: Upload zip
from google.colab import files
uploaded = files.upload()  # Upload sparse-attention-jax.zip
!unzip sparse-attention-jax.zip
%cd sparse-attention-jax

# Option B: Clone from GitHub (if hosted)
# !git clone https://github.com/yourusername/sparse-attention-jax.git
# %cd sparse-attention-jax
```

## 2. Installation

```bash
# Install the package and all dependencies
pip install -e .

# Or install dependencies directly
pip install -r requirements.txt
```

### Verify Installation:

```python
import jax
import jax.numpy as jnp
import sparse_attention

print(f"JAX version: {jax.__version__}")
print(f"Backend: {jax.default_backend()}")
print(f"Devices: {jax.devices()}")
print(f"Package version: {sparse_attention.__version__}")
```

## 3. Running the Demo

The demo tokenizes text, creates sparse masks, runs forward passes through both sparse and dense models, and compares them.

```bash
# Quick demo (smaller model, faster)
python main.py --quick

# Full demo (default config)
python main.py

# Custom settings
python main.py --seq-len 2048 --batch-size 4

# CPU/GPU mode (no Pallas kernel)
python main.py --quick --no-pallas
```

### What the demo shows:

1. **Tokenization**: Encodes text using GPT-2 BPE tokenizer
2. **Block Mask**: Generates and visualizes the sparsity pattern
3. **HBM Analysis**: Theoretical bandwidth comparison
4. **Forward Pass**: Runs both sparse and dense models
5. **Timing**: Measures and compares latency

## 4. Running Benchmarks

### Quick Benchmark (recommended for 1-hour Colab session):

```bash
python run_benchmarks.py --quick
```

Runs a reduced sweep (seq_len=[512,1024,2048], batch=[1,4]) and completes in ~5-10 minutes.

### Full Benchmark Suite:

```bash
python run_benchmarks.py
```

Sweeps all configurations. Takes ~30-45 minutes. Results saved to `benchmark_results/benchmark_results.json`.

### Specific Profiling:

```bash
# HBM bandwidth analysis
python run_benchmarks.py --hbm
python run_benchmarks.py --hbm --quick

# FLOPs analysis
python run_benchmarks.py --flops
python run_benchmarks.py --flops --quick

# Scaling analysis (how performance scales with seq_len)
python run_benchmarks.py --scaling
python run_benchmarks.py --scaling --quick
```

### Custom Sparsity Patterns:

```bash
# Try different patterns
python run_benchmarks.py --quick --sparsity causal
python run_benchmarks.py --quick --sparsity strided
python run_benchmarks.py --quick --sparsity combined

# Adjust block size
python run_benchmarks.py --quick --block-size 64
python run_benchmarks.py --quick --block-size 256
```

## 5. Running Tests

```bash
# All tests
python run_tests.py

# Or using pytest directly
python -m pytest tests/ -v

# Specific test files
python -m pytest tests/test_masks.py -v     # Mask tests
python -m pytest tests/test_kernel.py -v    # Kernel correctness
python -m pytest tests/test_model.py -v     # Model tests
python -m pytest tests/test_metrics.py -v   # Metrics tests

# Stop on first failure
python -m pytest tests/ -x

# Run with output
python -m pytest tests/ -v -s
```

## 6. Interpreting Results

### HBM Bandwidth Reduction

The headline metric. Calculated as:

```
reduction = (dense_hbm - sparse_hbm) / dense_hbm × 100%
```

**Expected values:**
- `combined` pattern, N=2048: **~40-50%** reduction
- `causal` pattern: lower reduction (~25-30%) since causal alone isn't very sparse
- `strided` pattern: moderate reduction (~35-45%)

### Why Sparse Uses Less HBM:

1. **No N×N attention matrix**: Online softmax avoids materializing the full attention matrix
2. **Skipped K/V blocks**: Masked blocks are never read from HBM
3. **Proportional savings**: With 50% sparsity, ~50% of K/V reads are skipped

### FLOPs Reduction

Calculated from theoretical operation counts:

| Operation | Dense | Sparse |
|-----------|-------|--------|
| Q×K^T     | 2·B·H·N²·D | 2·B·H·(active_blocks·bs²)·D |
| Softmax   | 5·B·H·N²   | 5·B·H·(active_blocks·bs²) |
| Attn×V    | 2·B·H·N²·D | 2·B·H·(active_blocks·bs²)·D |

### Speedup

```
speedup = dense_latency / sparse_latency
```

Expected: 1.3-2.0x on TPU for typical configurations.

## 7. All Commands Reference

### Demo Commands

| Command | Description |
|---------|-------------|
| `python main.py` | Full demo |
| `python main.py --quick` | Quick demo |
| `python main.py --seq-len 2048` | Custom sequence length |
| `python main.py --batch-size 8` | Custom batch size |
| `python main.py --no-pallas` | CPU/GPU mode |

### Benchmark Commands

| Command | Description |
|---------|-------------|
| `python run_benchmarks.py` | Full benchmark suite |
| `python run_benchmarks.py --quick` | Quick benchmark |
| `python run_benchmarks.py --hbm` | HBM profiling |
| `python run_benchmarks.py --flops` | FLOPs profiling |
| `python run_benchmarks.py --scaling` | Scaling analysis |
| `python run_benchmarks.py --sparsity TYPE` | Choose pattern |
| `python run_benchmarks.py --block-size N` | Set block size |
| `python run_benchmarks.py --output-dir DIR` | Save location |

### Test Commands

| Command | Description |
|---------|-------------|
| `python run_tests.py` | All tests |
| `python -m pytest tests/ -v` | Verbose output |
| `python -m pytest tests/test_kernel.py -v` | Kernel tests |
| `python -m pytest tests/ -x` | Stop on first fail |
| `python -m pytest tests/ -v -s` | Show print output |

### Individual Profiler Commands

| Command | Description |
|---------|-------------|
| `python -m benchmarks.profile_hbm` | HBM report |
| `python -m benchmarks.profile_flops` | FLOPs report |
| `python -m benchmarks.scaling_analysis` | Scaling report |

### Python API

```python
from sparse_attention import sparse_attention, dense_attention, create_block_mask
from sparse_attention.data import create_dummy_inputs

# Create inputs
q, k, v = create_dummy_inputs(batch_size=4, seq_len=1024)

# Create mask
mask = create_block_mask(1024, 128, "combined")

# Run sparse attention
output = sparse_attention(q, k, v, mask)

# Run dense attention
output_dense = dense_attention(q, k, v, causal=True)
```

## 8. Troubleshooting

See [debugging.md](debugging.md) for detailed troubleshooting guide.

### Quick fixes:

- **"No TPU found"**: Ensure Colab runtime is set to TPU
- **OOM errors**: Reduce `batch_size` or `seq_len`
- **Pallas import error**: Ensure `jax[tpu]>=0.4.30` is installed
- **Test failures**: Run `python -m pytest tests/ -v -s` for detailed output

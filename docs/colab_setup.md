# Google Colab TPU Setup Guide

Step-by-step guide for setting up and running on Google Colab's free TPU runtime.

## Prerequisites

- Google account
- Access to Google Colab ([colab.research.google.com](https://colab.research.google.com))
- The `sparse-attention-jax.zip` project file

## Step 1: Create New Notebook

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Click **"New notebook"**

## Step 2: Enable TPU Runtime

1. Click **Runtime** → **Change runtime type**
2. Under **Hardware accelerator**, select **TPU**
3. Select **TPU v5e** if available
4. Click **Save**

> **Note**: Free Colab gives you 1 hour of TPU time. Plan your experiments accordingly.

## Step 3: Verify TPU Access

Run in the first cell:

```python
import jax
print(f"JAX version: {jax.__version__}")
print(f"Backend: {jax.default_backend()}")
print(f"Devices: {jax.devices()}")
print(f"Device count: {jax.device_count()}")

# Should output:
# JAX version: 0.4.xx
# Backend: tpu
# Devices: [TpuDevice(id=0, ...)]
# Device count: 1
```

If it shows `cpu` instead of `tpu`, try:
```bash
!pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```
Then restart the runtime.

## Step 4: Upload and Install Project

**Option A — Upload zip:**
```python
# Upload the zip file
from google.colab import files
uploaded = files.upload()  # Select sparse-attention-jax.zip

# Extract
!unzip sparse-attention-jax.zip
%cd sparse-attention-jax

# Install
!pip install -e .
```

**Option B — Direct upload to folder:**
```python
# Create directory and upload individual files
# (Less convenient but works if zip upload fails)
!mkdir -p sparse-attention-jax
# Then use the file browser on the left to upload files
```

## Step 5: Run the Project

### Quick Demo (5 minutes):
```python
!python main.py --quick
```

### Quick Benchmarks (10 minutes):
```python
!python run_benchmarks.py --quick
```

### Tests (2 minutes):
```python
!python run_tests.py
```

### Full Benchmarks (30 minutes):
```python
!python run_benchmarks.py
```

### Individual Profilers:
```python
# HBM bandwidth profiling
!python run_benchmarks.py --hbm --quick

# FLOPs profiling
!python run_benchmarks.py --flops --quick

# Scaling analysis
!python run_benchmarks.py --scaling --quick
```

## Step 6: Save Results

```python
# Download benchmark results
from google.colab import files
files.download('benchmark_results/benchmark_results.json')
```

## Recommended Colab Workflow (1-hour session)

| Time | Activity | Command |
|------|----------|---------|
| 0:00 | Setup & install | Upload, `pip install -e .` |
| 0:05 | Verify TPU | Check `jax.devices()` |
| 0:07 | Quick demo | `python main.py --quick` |
| 0:12 | Run tests | `python run_tests.py` |
| 0:15 | HBM profiling | `python run_benchmarks.py --hbm --quick` |
| 0:20 | FLOPs profiling | `python run_benchmarks.py --flops --quick` |
| 0:25 | Scaling analysis | `python run_benchmarks.py --scaling --quick` |
| 0:30 | Quick benchmarks | `python run_benchmarks.py --quick` |
| 0:40 | Full demo | `python main.py` |
| 0:50 | Download results | Save JSON & screenshots |
| 0:55 | Buffer time | Fix any issues |

## TPU Memory Limits

TPU v5e-1 has ~16 GB HBM. Safe configurations:

| Batch Size | Seq Length | Estimated Memory |
|-----------|-----------|-----------------|
| 1 | 4096 | ~4 GB |
| 4 | 2048 | ~6 GB |
| 8 | 1024 | ~4 GB |
| 4 | 4096 | ~12 GB |
| 8 | 4096 | ~15 GB (close to limit!) |

If you get OOM, reduce batch_size or seq_len.

## Troubleshooting

See [debugging.md](debugging.md) for comprehensive troubleshooting.

### Quick Fixes:
- **Can't select TPU v5e**: Use whatever TPU is available (v2, v3, v4 also work)
- **OOM**: Use `--quick` or reduce `--batch-size`
- **Slow compilation**: First run is slow (XLA compilation). Subsequent runs are fast.
- **Disconnection**: Colab may disconnect if idle. Re-run setup cells.

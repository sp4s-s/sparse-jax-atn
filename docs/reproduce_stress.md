# Reproducing Stress Test Results

## Overview

The mega stress suite targets **40+ GB host RAM** on Colab TPU v5e-1.

**Key insight**: the TPU has 16GB HBM (device memory) and 47GB host RAM.
JAX device arrays sit in HBM; numpy arrays and data staging use host RAM.
The suite tests both limits separately and together.

---

## Quick Run

```bash
# default 40GB target
python run_benchmarks.py --mega-stress

# custom target
python run_benchmarks.py --mega-stress --target-gb 45

# standard stress (OOM boundary, sustained throughput, stability)
python run_benchmarks.py --stress
```

---

## Test Breakdown

### 1. Host RAM Saturation

Allocates **numpy float16 arrays** (which live in host RAM, not HBM)
in 512MB chunks until hitting the target. Then runs sparse attention
on-device under that memory pressure.

- **Why numpy?** JAX's `jnp.ones()` allocates on HBM (16GB cap). Real
  training pipelines hold datasets, tokenized batches, and checkpoints
  in host RAM — this simulates that.
- **Expected**: reaches ~40GB on Colab's 47GB host RAM.

### 2. Progressive OOM Ceiling Finder

Starts with a tiny config `B=1, N=1024` and **doubles the problem size**
until the TPU OOMs. Reports the exact ceiling for both sparse and dense.

Sparse attention's ceiling will be **much higher** than dense because it
never materializes the N×N attention matrix.

### 3. Gradient Accumulation

Runs micro-batches through sparse attention and accumulates results,
simulating effective batch sizes up to **4096**. Each step stays within
HBM, but total token throughput is massive.

### 4. Mixed Host+Device Pressure

Holds a large numpy "dataset" in host RAM (e.g. 30GB) while
simultaneously running attention kernels on the TPU. Tests whether
host memory pressure degrades device performance.

---

## Output Files

```
results/
├── mega_stress_all.json           # combined summary
├── mega_stress_host_ram.json      # host RAM saturation results
├── mega_stress_ceiling.json       # OOM ceiling per attention type
├── mega_stress_accumulation.json  # gradient accumulation throughput
└── mega_stress_mixed.json         # mixed pressure latency/throughput
```

---

## Generating Visualizations

```bash
python run_benchmarks.py --viz-all
```

Outputs to `results/viz_stress/`:
- `mega_stress_scatter_plotly.html` — estimated RAM vs execution time
- `grad_accum_throughput.png` — bar chart of tok/s per effective batch
- `oom_heatmap.png` — pass/fail matrix (from `--stress` runs)

---

## Key Metrics to Report

| Metric | Where |
|---|---|
| Host RAM saturated | `mega_stress_host_ram.json` → `achieved_gb` |
| Sparse OOM ceiling | `mega_stress_ceiling.json` → last PASS config |
| Dense OOM ceiling | `mega_stress_ceiling.json` → last PASS config |
| Sparse vs Dense advantage | ceiling ratio (sparse / dense est_gb) |
| Throughput under pressure | `mega_stress_mixed.json` → `tok_s` |
| Best gradient accum throughput | `mega_stress_accumulation.json` → max `tok_s` |

---

## Expected Results on TPU v5e-1

- **Host RAM**: should reach 35-42 GB before OOM
- **Sparse ceiling**: can handle configs up to ~25 GB estimated (since no N×N matrix)
- **Dense ceiling**: OOMs around 8-16 GB estimated (N×N matrix dominates)
- **Mixed pressure**: attention latency stays stable even under 30GB host load

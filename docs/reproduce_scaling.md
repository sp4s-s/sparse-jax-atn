# Reproducing Scaling & Roofline Analysis

## What is Generated

| Plot | File | Description |
|------|------|-------------|
| Scaling (log-log) | `viz_scaling/scaling_loglog.png` | O(N²) dense vs sub-quadratic sparse |
| Roofline model | `viz_scaling/roofline.png` | Compute vs memory-bound regions |
| MFU / HFU bars | `viz_scaling/mfu_hfu.png` | Model & hardware FLOPs utilization |
| Arithmetic intensity | `viz_scaling/arithmetic_intensity.png` | FLOPs/Byte with ridge point |

Interactive Plotly versions (`.html`) are generated alongside every PNG.

## How to Generate

### Step 1: Run roofline analysis

```bash
python run_benchmarks.py --roofline

# Quick version
python run_benchmarks.py --roofline --quick
```

This produces `benchmark_results/roofline_analysis.json`.

### Step 2: Generate scaling visuals

```bash
python -m sparse_attention.viz_scaling

# Or as part of the full viz suite
python run_benchmarks.py --viz-all
```

## Understanding the Roofline Model

The **roofline model** shows the maximum achievable performance given either:
- The **compute ceiling** (peak TFLOPs of the hardware), or
- The **memory bandwidth ceiling** (peak HBM bandwidth × arithmetic intensity)

The **ridge point** is where these two ceilings meet:

$$\text{Ridge Point} = \frac{\text{Peak TFLOPs} \times 1000}{\text{Peak BW (GB/s)}}$$

For TPU v5e: Ridge = 197 × 1000 / 820 ≈ 240 FLOPs/Byte.

- **Below the ridge**: Your kernel is **memory-bound**. Dense attention is typically here because it moves a lot of data (the full N×N attention matrix) relative to its compute.
- **Above the ridge**: Your kernel is **compute-bound**. This is rare for attention but can happen at very long sequences.

## Key Metrics

- **MFU (Model FLOPs Utilization)**: What fraction of the model's theoretical FLOPs you're actually achieving. Used in the PaLM and Chinchilla papers.
- **HFU (Hardware FLOPs Utilization)**: What fraction of the hardware's peak compute capability you're using.
- **Arithmetic Intensity**: FLOPs per byte of HBM traffic. Higher AI = more compute-efficient.

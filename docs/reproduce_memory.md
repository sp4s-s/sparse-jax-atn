# Reproducing Memory & HBM Bandwidth Results

## What is Generated

| Plot | File | Description |
|------|------|-------------|
| HBM waterfall | `viz_memory/hbm_waterfall.png` | Per-component HBM savings |
| Memory usage | `viz_memory/memory_usage.png` | Stacked bar dense vs sparse |
| HBM reduction % | `viz_memory/hbm_reduction.png` | Bar chart with 40% target line |
| Memory efficiency | `viz_memory/memory_pie_plotly.html` | Pie charts by config |
| Bandwidth util | `viz_memory/bandwidth_utilization.png` | Achieved vs peak BW |

## How to Generate

```bash
# Run core benchmarks first
python run_benchmarks.py --quick

# Generate memory visuals only
python -m sparse_attention.viz_memory

# Or all visuals at once
python run_benchmarks.py --viz-all
```

## Key Metrics

- **HBM Bandwidth Reduction (~40%)**: The headline metric. Sparse attention avoids materializing the full N×N attention matrix, saving ~40% HBM traffic.
- **Memory Efficiency**: Ratio of useful Q/K/V/O bytes to total bytes transferred. Higher = less wasted bandwidth.
- **Bandwidth Utilization**: What fraction of the 820 GB/s TPU v5e peak HBM bandwidth is achieved.

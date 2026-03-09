# Reproducing Throughput & Latency Results

## What is Generated

| Plot | File | Description |
|------|------|-------------|
| Throughput bar chart | `viz_throughput/throughput_bar.png` | Tokens/sec for sparse vs dense |
| Latency comparison | `viz_throughput/latency_comparison.png` | Grouped bars + speedup overlay |
| Latency distribution | `viz_throughput/latency_distribution.png` | Box plot + histogram with P99 |
| Latency CDF | `viz_throughput/latency_cdf.png` | Cumulative distribution |
| P99 heatmap | `viz_throughput/p99_heatmap_plotly.html` | Interactive tail latency matrix |

**Interactive versions** (`.html`) are also generated via Plotly alongside every PNG.

## How to Generate

### Step 1: Run the benchmarks (if not already done)

```bash
python run_benchmarks.py           # Full suite
python run_benchmarks.py --quick   # Quick version
```

### Step 2: Generate only throughput visuals

```bash
python -m sparse_attention.viz_throughput
```

Or as part of the complete visual set:
```bash
python run_benchmarks.py --viz-all
```

### Step 3: View results

All outputs land in `benchmark_results/viz_throughput/`.

Open `.html` files in a browser for interactive exploration. The PNGs are publication-quality dark-theme charts at 200 DPI.

## Key Metrics

- **Tokens/sec**: Primary throughput measure. Higher = better.
- **P99 Latency**: 99th percentile latency. Critical for SLA compliance in production.
- **Speedup (×)**: Ratio of dense/sparse latency. Target: >1.3× for seq_len ≥ 1024.

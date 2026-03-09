# Reproducing All Results & Visualizations

This project ships with an industrial-grade visualization pipeline. Every metric has its own dedicated visualization module, output directory, and reproduction guide.

## Quick Reference: One Command to Generate Everything

```bash
# 1. Run all benchmarks (creates JSON data files)
python run_benchmarks.py

# 2. Run roofline analysis
python run_benchmarks.py --roofline

# 3. Run stress tests
python run_benchmarks.py --stress

# 4. Run mega stress (~40 GB RAM target)
python run_benchmarks.py --mega-stress --target-gb 40

# 5. Run training with TensorBoard logging
python train.py --steps 500
python train.py --dense --steps 500

# 6. Generate ALL visualization sets at once
python run_benchmarks.py --viz-all
```

## Output Directory Structure

After running everything, the following directories are created:

```
benchmark_results/
├── benchmark_results.json           # Core benchmark data
├── roofline_analysis.json           # Roofline/MFU/HFU data
├── stress_test_all.json             # Stress test data
├── mega_stress_all.json             # 40 GB mega stress data
├── plots/                           # Combined matplotlib dashboard
├── plotly_plots/                    # Interactive Plotly HTML exports
├── viz_throughput/                  # Throughput & latency visuals
│   ├── throughput_bar.png
│   ├── latency_comparison.png
│   ├── latency_distribution.png
│   ├── latency_cdf.png
│   └── p99_heatmap_plotly.html
├── viz_memory/                      # Memory & HBM visuals
│   ├── hbm_waterfall.png
│   ├── memory_usage.png
│   ├── hbm_reduction.png
│   └── bandwidth_utilization.png
├── viz_stress/                      # Stress test visuals
│   ├── oom_heatmap.png
│   ├── sustained_throughput.png
│   ├── numerical_stability.png
│   ├── compilation_overhead.png
│   └── mega_stress_scatter_plotly.html
├── viz_scaling/                     # Scaling & roofline visuals
│   ├── scaling_loglog.png
│   ├── roofline.png
│   ├── mfu_hfu.png
│   └── arithmetic_intensity.png
└── viz_training/                    # Training curve visuals
    ├── loss_curve.png
    ├── grad_norm.png
    ├── lr_schedule.png
    ├── step_throughput.png
    └── training_dashboard.png

tensorboard_logs/                    # TensorBoard event files
```

## Detailed Guides (per Category)

Each visualization category has its own dedicated reproduction doc:

| Category | Guide | Module | Output Dir |
|----------|-------|--------|-----------|
| Throughput & Latency | [reproduce_throughput.md](reproduce_throughput.md) | `viz_throughput.py` | `viz_throughput/` |
| Memory & HBM | [reproduce_memory.md](reproduce_memory.md) | `viz_memory.py` | `viz_memory/` |
| Training Curves | [reproduce_training.md](reproduce_training.md) | `viz_training.py` | `viz_training/` |
| Stress Tests | [reproduce_stress.md](reproduce_stress.md) | `viz_stress.py` | `viz_stress/` |
| Scaling & Roofline | [reproduce_scaling.md](reproduce_scaling.md) | `viz_scaling.py` | `viz_scaling/` |

## TensorBoard Live Monitoring

```bash
tensorboard --logdir tensorboard_logs/
```

This gives you real-time loss, gradient norms, learning rate, and tokens/sec tracking.

## All CLI Commands

```bash
python run_benchmarks.py                     # Full benchmark suite
python run_benchmarks.py --quick             # Quick (few configs)
python run_benchmarks.py --hbm               # HBM profiling only
python run_benchmarks.py --flops             # FLOPs profiling only
python run_benchmarks.py --scaling           # Scaling analysis only
python run_benchmarks.py --roofline          # Roofline model
python run_benchmarks.py --stress            # Stress tests (OOM, throughput, stability)
python run_benchmarks.py --mega-stress       # 40 GB mega stress test
python run_benchmarks.py --viz-only          # Regenerate combined plots from JSON
python run_benchmarks.py --viz-all           # Generate ALL separate viz sets
python train.py --steps 500                  # Training with TensorBoard
python train.py --dense --steps 500          # Dense baseline training
```

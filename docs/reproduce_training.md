# Reproducing Training Curves (Loss, Gradients, LR)

## What is Generated

| Plot | File | Description |
|------|------|-------------|
| Loss curve | `viz_training/loss_curve.png` | Raw loss + EMA smoothing |
| Gradient norm | `viz_training/grad_norm.png` | Log-scale gradient norms |
| LR schedule | `viz_training/lr_schedule.png` | Warmup + cosine decay |
| Step throughput | `viz_training/step_throughput.png` | Tokens/sec stability |
| Dashboard | `viz_training/training_dashboard.png` | Combined 2×2 view |

Interactive Plotly `.html` versions are also generated.

## How to Generate

### Step 1: Run the training loop (creates TensorBoard events)

```bash
# Sparse attention training
python train.py --steps 500

# Dense baseline for comparison
python train.py --dense --steps 500
```

### Step 2: View in TensorBoard (live)

```bash
tensorboard --logdir tensorboard_logs/
```

### Step 3: Generate static plots from the events

```bash
python -m sparse_attention.viz_training

# Or as part of the full viz suite
python run_benchmarks.py --viz-all
```

## Key Metrics

- **Loss curve**: Should decrease steadily. Smoothed (EMA) line should show clear convergence.
- **Gradient norm**: Should stay below the clip threshold (1.0). If it spikes, the model is unstable.
- **LR schedule**: Warmup + cosine decay is standard for Transformer training.
- **Throughput**: Should be stable. Large jitter indicates memory pressure or GC pauses.

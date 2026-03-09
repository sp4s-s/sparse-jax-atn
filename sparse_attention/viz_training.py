from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MPL = True
except ImportError:
    MPL = False

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY = True
except ImportError:
    PLOTLY = False


DARK = {
    "figure.facecolor": "#0d1117",
    "axes.facecolor": "#161b22",
    "axes.edgecolor": "#30363d",
    "axes.labelcolor": "#c9d1d9",
    "text.color": "#c9d1d9",
    "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",
    "grid.color": "#21262d",
}
CLR = {"blue": "#58a6ff", "red": "#f85149", "green": "#3fb950",
       "purple": "#bc8cff", "cyan": "#39d2c0", "orange": "#f0883e"}


def _save(fig, path, dpi=200):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  📊 {path}")

def _save_plotly(fig, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    try:
        fig.write_image(path, scale=2)
    except Exception:
        pass
    fig.write_html(path.replace(".png", ".html"))
    print(f"  🌐 {path.replace('.png', '.html')}")


def _moving_avg(arr, window=20):
    if len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="valid")


def plot_loss_curve(
    steps: List[int],
    losses: List[float],
    title: str = "Training Loss",
    output_dir: str = "viz_training",
    window: int = 20,
):
    os.makedirs(output_dir, exist_ok=True)
    s = np.array(steps); l = np.array(losses)

    if MPL and len(l) > 2:
        with plt.rc_context(DARK):
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(s, l, alpha=0.3, color=CLR["blue"], linewidth=0.8, label="Raw")
            if len(l) > window:
                smooth = _moving_avg(l, window)
                ax.plot(s[window-1:], smooth, color=CLR["cyan"], linewidth=2,
                       label=f"EMA (w={window})")
            ax.set_xlabel("Step"); ax.set_ylabel("Loss")
            ax.set_title(title, fontsize=14, fontweight="bold")
            ax.legend()
            ax.grid(alpha=0.3)
            _save(fig, os.path.join(output_dir, "loss_curve.png"))

    if PLOTLY and len(l) > 2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=s.tolist(), y=l.tolist(), mode="lines",
                                 name="Raw", opacity=0.4, line=dict(color=CLR["blue"])))
        if len(l) > window:
            smooth = _moving_avg(l, window)
            fig.add_trace(go.Scatter(x=s[window-1:].tolist(), y=smooth.tolist(),
                                     mode="lines", name="Smoothed",
                                     line=dict(color=CLR["cyan"], width=2)))
        fig.update_layout(title=title, template="plotly_dark",
                          xaxis_title="Step", yaxis_title="Loss")
        _save_plotly(fig, os.path.join(output_dir, "loss_curve_plotly.png"))


def plot_grad_norm(
    steps: List[int],
    grad_norms: List[float],
    output_dir: str = "viz_training",
):
    os.makedirs(output_dir, exist_ok=True)
    s = np.array(steps); g = np.array(grad_norms)

    if MPL and len(g) > 2:
        with plt.rc_context(DARK):
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.semilogy(s, g, color=CLR["orange"], linewidth=1.5, alpha=0.8)
            ax.axhline(y=1.0, linestyle="--", color=CLR["red"], alpha=0.6, label="Clip threshold")
            ax.set_xlabel("Step"); ax.set_ylabel("Gradient Norm (log)")
            ax.set_title("Gradient Norm", fontsize=14, fontweight="bold")
            ax.legend(); ax.grid(alpha=0.3)
            _save(fig, os.path.join(output_dir, "grad_norm.png"))


def plot_lr_schedule(
    steps: List[int],
    lrs: List[float],
    output_dir: str = "viz_training",
):
    os.makedirs(output_dir, exist_ok=True)
    s = np.array(steps); lr = np.array(lrs)

    if MPL and len(lr) > 2:
        with plt.rc_context(DARK):
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(s, lr, color=CLR["purple"], linewidth=2)
            ax.set_xlabel("Step"); ax.set_ylabel("Learning Rate")
            ax.set_title("Learning Rate Schedule (Warmup + Cosine Decay)",
                        fontsize=14, fontweight="bold")
            ax.grid(alpha=0.3)
            _save(fig, os.path.join(output_dir, "lr_schedule.png"))


def plot_step_throughput(
    steps: List[int],
    tokens_per_sec: List[float],
    output_dir: str = "viz_training",
):
    os.makedirs(output_dir, exist_ok=True)
    s = np.array(steps); t = np.array(tokens_per_sec)

    if MPL and len(t) > 2:
        with plt.rc_context(DARK):
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(s, t, color=CLR["green"], linewidth=1, alpha=0.6)
            if len(t) > 10:
                smooth = _moving_avg(t, 10)
                ax.plot(s[9:], smooth, color=CLR["cyan"], linewidth=2, label="Smoothed")
            ax.axhline(y=np.mean(t), linestyle=":", color=CLR["orange"],
                       label=f"Mean: {np.mean(t):.0f}")
            ax.set_xlabel("Step"); ax.set_ylabel("Tokens/sec")
            ax.set_title("Training Throughput Stability", fontsize=14, fontweight="bold")
            ax.legend(); ax.grid(alpha=0.3)
            _save(fig, os.path.join(output_dir, "step_throughput.png"))


def plot_training_dashboard(
    steps: List[int],
    losses: List[float],
    grad_norms: List[float],
    lrs: List[float],
    tokens_per_sec: List[float],
    output_dir: str = "viz_training",
):
    os.makedirs(output_dir, exist_ok=True)

    if MPL and len(losses) > 2:
        with plt.rc_context(DARK):
            fig, axes = plt.subplots(2, 2, figsize=(16, 10))

            ax = axes[0, 0]
            ax.plot(steps, losses, alpha=0.3, color=CLR["blue"])
            if len(losses) > 20:
                sm = _moving_avg(np.array(losses), 20)
                ax.plot(steps[19:], sm, color=CLR["cyan"], linewidth=2)
            ax.set_title("Training Loss", fontweight="bold")
            ax.set_xlabel("Step"); ax.set_ylabel("Loss"); ax.grid(alpha=0.3)

            ax = axes[0, 1]
            ax.semilogy(steps, grad_norms, color=CLR["orange"], alpha=0.8)
            ax.axhline(y=1.0, linestyle="--", color=CLR["red"], alpha=0.5)
            ax.set_title("Gradient Norm", fontweight="bold")
            ax.set_xlabel("Step"); ax.set_ylabel("Norm (log)"); ax.grid(alpha=0.3)

            ax = axes[1, 0]
            ax.plot(steps, lrs, color=CLR["purple"], linewidth=2)
            ax.set_title("Learning Rate", fontweight="bold")
            ax.set_xlabel("Step"); ax.set_ylabel("LR"); ax.grid(alpha=0.3)

            ax = axes[1, 1]
            ax.plot(steps, tokens_per_sec, color=CLR["green"], alpha=0.6)
            if len(tokens_per_sec) > 10:
                sm = _moving_avg(np.array(tokens_per_sec), 10)
                ax.plot(steps[9:], sm, color=CLR["cyan"], linewidth=2)
            ax.set_title("Throughput (tok/s)", fontweight="bold")
            ax.set_xlabel("Step"); ax.set_ylabel("Tokens/sec"); ax.grid(alpha=0.3)

            fig.suptitle("Training Dashboard", fontsize=16, fontweight="bold", y=1.01)
            fig.tight_layout()
            _save(fig, os.path.join(output_dir, "training_dashboard.png"))

    if PLOTLY and len(losses) > 2:
        fig = make_subplots(rows=2, cols=2,
                            subplot_titles=["Loss", "Gradient Norm", "Learning Rate", "Throughput"])
        fig.add_trace(go.Scatter(x=steps, y=losses, mode="lines", name="Loss",
                                 line=dict(color=CLR["blue"])), row=1, col=1)
        fig.add_trace(go.Scatter(x=steps, y=grad_norms, mode="lines", name="Grad Norm",
                                 line=dict(color=CLR["orange"])), row=1, col=2)
        fig.add_trace(go.Scatter(x=steps, y=lrs, mode="lines", name="LR",
                                 line=dict(color=CLR["purple"])), row=2, col=1)
        fig.add_trace(go.Scatter(x=steps, y=tokens_per_sec, mode="lines", name="Tok/s",
                                 line=dict(color=CLR["green"])), row=2, col=2)
        fig.update_layout(title="Training Dashboard", template="plotly_dark", height=700)
        _save_plotly(fig, os.path.join(output_dir, "training_dashboard_plotly.png"))


def read_tensorboard_events(logdir: str) -> Dict[str, List]:
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        print("  [WARN] tensorboard not installed, cannot read events.")
        return {}

    data = {}
    for root, dirs, files in os.walk(logdir):
        for fname in files:
            if "events.out.tfevents" in fname:
                ea = EventAccumulator(os.path.join(root, fname))
                ea.Reload()
                for tag in ea.Tags().get("scalars", []):
                    events = ea.Scalars(tag)
                    data[tag] = {
                        "steps": [e.step for e in events],
                        "values": [e.value for e in events],
                    }
    return data


def generate_training_viz(
    logdir: str = "tensorboard_logs",
    output_dir: str = "benchmark_results/viz_training",
):
    print(f"\n{'='*70}")
    print("GENERATING TRAINING VISUALIZATIONS")
    print(f"{'='*70}")

    data = read_tensorboard_events(logdir)
    if not data:
        print(f"  No TensorBoard events found in {logdir}")
        print("  Run `python train.py --steps 500` first to generate data.")
        return

    loss_data = data.get("Loss/train", {})
    grad_data = data.get("Optimization/grad_norm", {})
    lr_data = data.get("Optimization/learning_rate", {})
    tok_data = data.get("Performance/tokens_per_sec", {})

    if loss_data:
        plot_loss_curve(loss_data["steps"], loss_data["values"], output_dir=output_dir)
    if grad_data:
        plot_grad_norm(grad_data["steps"], grad_data["values"], output_dir=output_dir)
    if lr_data:
        plot_lr_schedule(lr_data["steps"], lr_data["values"], output_dir=output_dir)
    if tok_data:
        plot_step_throughput(tok_data["steps"], tok_data["values"], output_dir=output_dir)

    if loss_data and grad_data and lr_data and tok_data:
        min_len = min(len(loss_data["steps"]), len(grad_data["steps"]),
                      len(lr_data["steps"]), len(tok_data["steps"]))
        plot_training_dashboard(
            loss_data["steps"][:min_len],
            loss_data["values"][:min_len],
            grad_data["values"][:min_len],
            lr_data["values"][:min_len],
            tok_data["values"][:min_len],
            output_dir=output_dir,
        )

    print(f"  ✅ Training visuals → {output_dir}")


if __name__ == "__main__":
    import sys
    logdir = sys.argv[1] if len(sys.argv) > 1 else "tensorboard_logs"
    generate_training_viz(logdir)

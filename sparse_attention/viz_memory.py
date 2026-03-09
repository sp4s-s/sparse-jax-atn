from __future__ import annotations

import json
import os
from typing import Dict, List

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


def plot_hbm_waterfall(
    components: Dict[str, tuple],
    output_dir: str = "viz_memory",
):
    os.makedirs(output_dir, exist_ok=True)
    names = list(components.keys())
    dense_vals = [v[0] for v in components.values()]
    sparse_vals = [v[1] for v in components.values()]
    savings = [d - s for d, s in zip(dense_vals, sparse_vals)]

    if MPL:
        with plt.rc_context(DARK):
            fig, ax = plt.subplots(figsize=(12, 6))
            x = np.arange(len(names))
            w = 0.30
            ax.bar(x - w, dense_vals, w, label="Dense", color=CLR["red"], alpha=0.85)
            ax.bar(x, sparse_vals, w, label="Sparse", color=CLR["blue"], alpha=0.85)
            ax.bar(x + w, savings, w, label="Savings", color=CLR["green"], alpha=0.85)
            ax.set_xticks(x); ax.set_xticklabels(names, rotation=20, ha="right")
            ax.set_ylabel("HBM (MB)")
            ax.set_title("HBM Bandwidth Waterfall", fontsize=14, fontweight="bold")
            ax.legend()
            ax.grid(axis="y", alpha=0.3)
            _save(fig, os.path.join(output_dir, "hbm_waterfall.png"))

    if PLOTLY:
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Dense", x=names, y=dense_vals, marker_color=CLR["red"]))
        fig.add_trace(go.Bar(name="Sparse", x=names, y=sparse_vals, marker_color=CLR["blue"]))
        fig.add_trace(go.Bar(name="Savings", x=names, y=savings, marker_color=CLR["green"]))
        fig.update_layout(title="HBM Waterfall", template="plotly_dark", barmode="group")
        _save_plotly(fig, os.path.join(output_dir, "hbm_waterfall_plotly.png"))


def plot_memory_stacked(
    configs: List[str],
    dense_mb: List[float],
    sparse_mb: List[float],
    output_dir: str = "viz_memory",
):
    os.makedirs(output_dir, exist_ok=True)

    if MPL:
        with plt.rc_context(DARK):
            fig, ax = plt.subplots(figsize=(12, 6))
            x = np.arange(len(configs)); w = 0.35
            ax.bar(x - w/2, dense_mb, w, label="Dense HBM", color=CLR["red"], alpha=0.85)
            ax.bar(x + w/2, sparse_mb, w, label="Sparse HBM", color=CLR["blue"], alpha=0.85)

            for i in range(len(configs)):
                if dense_mb[i] > 0:
                    pct = (1 - sparse_mb[i] / dense_mb[i]) * 100
                    ax.annotate(f"-{pct:.0f}%", (x[i] + w/2, sparse_mb[i]),
                               ha="center", va="bottom", fontsize=9,
                               color=CLR["green"], fontweight="bold")

            ax.set_xticks(x); ax.set_xticklabels(configs, rotation=25, ha="right")
            ax.set_ylabel("HBM (MB)")
            ax.set_title("Memory Usage: Dense vs Sparse", fontsize=14, fontweight="bold")
            ax.legend()
            ax.grid(axis="y", alpha=0.3)
            _save(fig, os.path.join(output_dir, "memory_usage.png"))


def plot_hbm_reduction(comparisons: List[Dict], output_dir: str = "viz_memory"):
    os.makedirs(output_dir, exist_ok=True)

    labels = [f"B={c.get('B')}\nN={c.get('N')}" for c in comparisons]
    reductions = [float(str(c.get("HBM Reduction", "0")).replace("%", "")) for c in comparisons]

    if MPL:
        with plt.rc_context(DARK):
            fig, ax = plt.subplots(figsize=(12, 6))
            colors = [CLR["green"] if r >= 35 else CLR["orange"] for r in reductions]
            ax.bar(labels, reductions, color=colors, alpha=0.85, edgecolor="#30363d")
            ax.axhline(y=40, linestyle="--", color=CLR["red"], alpha=0.8, label="40% Target")
            ax.set_ylabel("HBM Reduction (%)")
            ax.set_title("HBM Bandwidth Reduction", fontsize=14, fontweight="bold")
            ax.legend()
            ax.grid(axis="y", alpha=0.3)

            for i, v in enumerate(reductions):
                ax.text(i, v + 0.5, f"{v:.1f}%", ha="center", fontsize=9,
                       color=CLR["green"] if v >= 35 else CLR["orange"])

            _save(fig, os.path.join(output_dir, "hbm_reduction.png"))

    if PLOTLY:
        fig = go.Figure(data=go.Bar(x=labels, y=reductions, marker_color=CLR["purple"]))
        fig.add_hline(y=40, line_dash="dash", line_color=CLR["red"], annotation_text="Target 40%")
        fig.update_layout(title="HBM Bandwidth Reduction %", template="plotly_dark",
                          yaxis_title="Reduction (%)")
        _save_plotly(fig, os.path.join(output_dir, "hbm_reduction_plotly.png"))


def plot_memory_efficiency(comparisons: List[Dict], output_dir: str = "viz_memory"):
    os.makedirs(output_dir, exist_ok=True)

    labels = [f"B={c.get('B')}\nN={c.get('N')}" for c in comparisons]
    d_hbm = [float(str(c.get("Dense HBM (MB)", "1"))) for c in comparisons]
    s_hbm = [float(str(c.get("Sparse HBM (MB)", "1"))) for c in comparisons]

    if PLOTLY:
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Dense HBM", "Sparse HBM"),
                            specs=[[{"type": "pie"}, {"type": "pie"}]])
        fig.add_trace(go.Pie(labels=labels, values=d_hbm, marker_colors=["#f85149", "#d29922", "#f0883e", "#bc8cff"]),
                      row=1, col=1)
        fig.add_trace(go.Pie(labels=labels, values=s_hbm, marker_colors=["#58a6ff", "#39d2c0", "#3fb950", "#79c0ff"]),
                      row=1, col=2)
        fig.update_layout(title="Memory Allocation by Config", template="plotly_dark")
        _save_plotly(fig, os.path.join(output_dir, "memory_pie_plotly.png"))


def plot_bandwidth_utilization(
    configs: List[str],
    bw_dense: List[float],
    bw_sparse: List[float],
    peak_bw: float = 820.0,
    output_dir: str = "viz_memory",
):
    os.makedirs(output_dir, exist_ok=True)

    if MPL:
        with plt.rc_context(DARK):
            fig, ax = plt.subplots(figsize=(12, 6))
            x = np.arange(len(configs)); w = 0.35
            ax.barh(x - w/2, bw_dense, w, label="Dense", color=CLR["red"], alpha=0.85)
            ax.barh(x + w/2, bw_sparse, w, label="Sparse", color=CLR["blue"], alpha=0.85)
            ax.axvline(x=peak_bw, linestyle="--", color=CLR["green"], alpha=0.8, label=f"Peak {peak_bw:.0f} GB/s")
            ax.set_yticks(x); ax.set_yticklabels(configs)
            ax.set_xlabel("Bandwidth (GB/s)")
            ax.set_title("HBM Bandwidth Utilization", fontsize=14, fontweight="bold")
            ax.legend(); ax.grid(axis="x", alpha=0.3)
            _save(fig, os.path.join(output_dir, "bandwidth_utilization.png"))


def generate_memory_viz(
    results_json: str = "benchmark_results/benchmark_results.json",
    output_dir: str = "benchmark_results/viz_memory",
):
    print(f"\n{'='*70}")
    print("GENERATING MEMORY & HBM VISUALIZATIONS")
    print(f"{'='*70}")

    try:
        with open(results_json) as f:
            data = json.load(f)
    except Exception as e:
        print(f"  Could not load {results_json}: {e}")
        return

    comparisons = data.get("comparisons", [])
    if not comparisons:
        print("  No comparison data found.")
        return

    plot_hbm_reduction(comparisons, output_dir)
    plot_memory_efficiency(comparisons, output_dir)

    configs = [f"B={c.get('B')},N={c.get('N')}" for c in comparisons]
    d_hbm = [float(str(c.get("Dense HBM (MB)", 0))) for c in comparisons]
    s_hbm = [float(str(c.get("Sparse HBM (MB)", 0))) for c in comparisons]
    plot_memory_stacked(configs, d_hbm, s_hbm, output_dir)

    print(f"  ✅ All memory visuals → {output_dir}")


if __name__ == "__main__":
    generate_memory_viz()

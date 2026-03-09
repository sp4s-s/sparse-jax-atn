from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
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


def plot_throughput(comparisons: List[Dict], output_dir: str = "viz_throughput"):
    os.makedirs(output_dir, exist_ok=True)

    labels = [f"B={c.get('batch_size', c.get('B'))}\nN={c.get('seq_len', c.get('N'))}"
              for c in comparisons]
    dense_t = [float(c.get("dense_tokens_per_sec",
                           c.get("Dense Tok/s", 0))) for c in comparisons]
    sparse_t = [float(c.get("sparse_tokens_per_sec",
                            c.get("Sparse Tok/s", 0))) for c in comparisons]

    if MPL:
        with plt.rc_context(DARK):
            fig, ax = plt.subplots(figsize=(12, 6))
            x = np.arange(len(labels))
            w = 0.35
            ax.bar(x - w/2, dense_t, w, label="Dense", color=CLR["red"], alpha=0.85)
            ax.bar(x + w/2, sparse_t, w, label="Sparse", color=CLR["blue"], alpha=0.85)
            ax.set_xticks(x); ax.set_xticklabels(labels)
            ax.set_ylabel("Tokens / Second")
            ax.set_title("Throughput: Sparse vs Dense Attention", fontsize=14, fontweight="bold")
            ax.legend()
            ax.grid(axis="y", alpha=0.3)
            _save(fig, os.path.join(output_dir, "throughput_bar.png"))

    if PLOTLY:
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Dense", x=labels, y=dense_t, marker_color=CLR["red"]))
        fig.add_trace(go.Bar(name="Sparse", x=labels, y=sparse_t, marker_color=CLR["blue"]))
        fig.update_layout(title="Throughput: Sparse vs Dense", barmode="group",
                          template="plotly_dark", yaxis_title="Tokens/sec")
        _save_plotly(fig, os.path.join(output_dir, "throughput_bar_plotly.png"))


def plot_latency_bars(comparisons: List[Dict], output_dir: str = "viz_throughput"):
    os.makedirs(output_dir, exist_ok=True)

    labels = [f"B={c.get('batch_size', c.get('B'))}\nN={c.get('seq_len', c.get('N'))}"
              for c in comparisons]
    dense_lat = [float(c.get("Dense Latency (ms)", 0)) for c in comparisons]
    sparse_lat = [float(c.get("Sparse Latency (ms)", 0)) for c in comparisons]
    speedups = [float(str(c.get("Speedup", "1")).replace("x", "")) for c in comparisons]

    if MPL:
        with plt.rc_context(DARK):
            fig, ax1 = plt.subplots(figsize=(12, 6))
            x = np.arange(len(labels)); w = 0.35
            ax1.bar(x - w/2, dense_lat, w, label="Dense", color=CLR["red"], alpha=0.85)
            ax1.bar(x + w/2, sparse_lat, w, label="Sparse", color=CLR["blue"], alpha=0.85)
            ax1.set_ylabel("Latency (ms)")
            ax1.set_xticks(x); ax1.set_xticklabels(labels)

            ax2 = ax1.twinx()
            ax2.plot(x, speedups, "o-", color=CLR["green"], linewidth=2, markersize=8, label="Speedup")
            ax2.set_ylabel("Speedup (×)")
            ax2.tick_params(axis='y', labelcolor=CLR["green"])

            ax1.legend(loc="upper left"); ax2.legend(loc="upper right")
            ax1.set_title("Latency Comparison with Speedup Overlay",
                          fontsize=14, fontweight="bold")
            ax1.grid(axis="y", alpha=0.3)
            _save(fig, os.path.join(output_dir, "latency_comparison.png"))

    if PLOTLY:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(name="Dense", x=labels, y=dense_lat, marker_color=CLR["red"]), secondary_y=False)
        fig.add_trace(go.Bar(name="Sparse", x=labels, y=sparse_lat, marker_color=CLR["blue"]), secondary_y=False)
        fig.add_trace(go.Scatter(name="Speedup", x=labels, y=speedups, mode="lines+markers",
                                 marker_color=CLR["green"]), secondary_y=True)
        fig.update_layout(title="Latency + Speedup", template="plotly_dark", barmode="group")
        fig.update_yaxes(title_text="ms", secondary_y=False)
        fig.update_yaxes(title_text="Speedup", secondary_y=True)
        _save_plotly(fig, os.path.join(output_dir, "latency_comparison_plotly.png"))


def plot_latency_distribution(
    dense_times: List[float],
    sparse_times: List[float],
    config_label: str = "",
    output_dir: str = "viz_throughput",
):
    os.makedirs(output_dir, exist_ok=True)
    d = np.array(dense_times); s = np.array(sparse_times)

    if MPL and len(d) > 1 and len(s) > 1:
        with plt.rc_context(DARK):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

            bp = ax1.boxplot([d, s], labels=["Dense", "Sparse"], patch_artist=True,
                             boxprops=dict(facecolor="#161b22"),
                             medianprops=dict(color=CLR["green"], linewidth=2))
            bp["boxes"][0].set_facecolor(CLR["red"]); bp["boxes"][0].set_alpha(0.4)
            bp["boxes"][1].set_facecolor(CLR["blue"]); bp["boxes"][1].set_alpha(0.4)
            ax1.set_ylabel("Latency (ms)")
            ax1.set_title("Latency Distribution", fontweight="bold")

            for i, arr in enumerate([d, s]):
                p99 = np.percentile(arr, 99)
                ax1.axhline(y=p99, linestyle=":", alpha=0.5,
                           color=CLR["red"] if i == 0 else CLR["blue"])
                ax1.text(i + 1.2, p99, f"P99={p99:.2f}", fontsize=8,
                        color=CLR["red"] if i == 0 else CLR["blue"])

            ax2.hist(d, bins=30, alpha=0.6, color=CLR["red"], label="Dense", density=True)
            ax2.hist(s, bins=30, alpha=0.6, color=CLR["blue"], label="Sparse", density=True)
            ax2.set_xlabel("Latency (ms)"); ax2.set_ylabel("Density")
            ax2.set_title("Latency Histogram", fontweight="bold")
            ax2.legend()

            fig.suptitle(f"Latency Analysis {config_label}", fontsize=14, fontweight="bold", y=1.02)
            _save(fig, os.path.join(output_dir, "latency_distribution.png"))

    if PLOTLY and len(d) > 1 and len(s) > 1:
        fig = go.Figure()
        fig.add_trace(go.Box(y=d, name="Dense", marker_color=CLR["red"]))
        fig.add_trace(go.Box(y=s, name="Sparse", marker_color=CLR["blue"]))
        fig.update_layout(title="Latency Distribution", template="plotly_dark",
                          yaxis_title="Latency (ms)")
        _save_plotly(fig, os.path.join(output_dir, "latency_dist_plotly.png"))


def plot_latency_cdf(
    dense_times: List[float],
    sparse_times: List[float],
    output_dir: str = "viz_throughput",
):
    os.makedirs(output_dir, exist_ok=True)
    d = np.sort(dense_times); s = np.sort(sparse_times)

    if MPL and len(d) > 1 and len(s) > 1:
        with plt.rc_context(DARK):
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(d, np.linspace(0, 1, len(d)), color=CLR["red"], linewidth=2, label="Dense")
            ax.plot(s, np.linspace(0, 1, len(s)), color=CLR["blue"], linewidth=2, label="Sparse")
            ax.axhline(y=0.50, linestyle=":", color="#8b949e", alpha=0.5, label="P50")
            ax.axhline(y=0.99, linestyle=":", color=CLR["orange"], alpha=0.7, label="P99")
            ax.set_xlabel("Latency (ms)"); ax.set_ylabel("CDF")
            ax.set_title("Latency CDF", fontsize=14, fontweight="bold")
            ax.legend()
            ax.grid(alpha=0.3)
            _save(fig, os.path.join(output_dir, "latency_cdf.png"))


def plot_p99_heatmap(
    configs: List[str],
    dense_p99: List[float],
    sparse_p99: List[float],
    output_dir: str = "viz_throughput",
):
    os.makedirs(output_dir, exist_ok=True)

    if PLOTLY:
        fig = go.Figure(data=go.Heatmap(
            z=[dense_p99, sparse_p99],
            x=configs,
            y=["Dense P99", "Sparse P99"],
            colorscale="YlOrRd",
            text=[[f"{v:.2f}ms" for v in dense_p99],
                  [f"{v:.2f}ms" for v in sparse_p99]],
            texttemplate="%{text}",
        ))
        fig.update_layout(title="P99 Tail Latency Heatmap", template="plotly_dark")
        _save_plotly(fig, os.path.join(output_dir, "p99_heatmap_plotly.png"))


def generate_throughput_viz(
    results_json: str = "benchmark_results/benchmark_results.json",
    output_dir: str = "benchmark_results/viz_throughput",
):
    print(f"\n{'='*70}")
    print("GENERATING THROUGHPUT & LATENCY VISUALIZATIONS")
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

    plot_throughput(comparisons, output_dir)
    plot_latency_bars(comparisons, output_dir)

    details = data.get("details", {})
    dense_raws = details.get("dense_all_latencies", [])
    sparse_raws = details.get("sparse_all_latencies", [])
    if dense_raws and sparse_raws:
        plot_latency_distribution(dense_raws, sparse_raws, output_dir=output_dir)
        plot_latency_cdf(dense_raws, sparse_raws, output_dir)

    configs = [f"B={c.get('B')},N={c.get('N')}" for c in comparisons]
    dp99 = [float(c.get("Dense P99 (ms)", 0)) for c in comparisons]
    sp99 = [float(c.get("Sparse P99 (ms)", 0)) for c in comparisons]
    if any(v > 0 for v in dp99):
        plot_p99_heatmap(configs, dp99, sp99, output_dir)

    print(f"  ✅ All throughput visuals → {output_dir}")


if __name__ == "__main__":
    generate_throughput_viz()

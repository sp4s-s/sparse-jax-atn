from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
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


def plot_oom_heatmap(
    seq_lengths: List[int],
    batch_sizes: List[int],
    status_matrix,
    title: str = "OOM Boundary",
    output_dir: str = "viz_stress",
):
    os.makedirs(output_dir, exist_ok=True)
    z = np.array(status_matrix)

    if MPL:
        with plt.rc_context(DARK):
            fig, ax = plt.subplots(figsize=(max(8, len(batch_sizes)*1.5), max(6, len(seq_lengths)*0.8)))
            cmap = ListedColormap(["#f85149", "#3fb950"])
            im = ax.imshow(z, cmap=cmap, aspect="auto", vmin=0, vmax=1)

            ax.set_xticks(range(len(batch_sizes)))
            ax.set_xticklabels([f"B={b}" for b in batch_sizes])
            ax.set_yticks(range(len(seq_lengths)))
            ax.set_yticklabels([f"N={n}" for n in seq_lengths])

            for i in range(len(seq_lengths)):
                for j in range(len(batch_sizes)):
                    txt = "✓" if z[i, j] == 1 else "✗"
                    ax.text(j, i, txt, ha="center", va="center",
                           fontsize=16, fontweight="bold",
                           color="white" if z[i, j] == 0 else "#0d1117")

            ax.set_title(title, fontsize=14, fontweight="bold")
            ax.set_xlabel("Batch Size")
            ax.set_ylabel("Sequence Length")
            _save(fig, os.path.join(output_dir, "oom_heatmap.png"))

    if PLOTLY:
        colorscale = [[0, "#f85149"], [1, "#3fb950"]]
        fig = go.Figure(data=go.Heatmap(
            z=z.tolist(),
            x=[f"B={b}" for b in batch_sizes],
            y=[f"N={n}" for n in seq_lengths],
            colorscale=colorscale, showscale=False,
            text=[["✓ Pass" if v == 1 else "✗ OOM" for v in row] for row in z.tolist()],
            texttemplate="%{text}",
        ))
        fig.update_layout(title=title, template="plotly_dark")
        _save_plotly(fig, os.path.join(output_dir, "oom_heatmap_plotly.png"))


def plot_sustained_throughput(
    latencies_ms: List[float],
    throughputs: List[float],
    output_dir: str = "viz_stress",
):
    os.makedirs(output_dir, exist_ok=True)

    if MPL and len(latencies_ms) > 2:
        with plt.rc_context(DARK):
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

            x = list(range(len(latencies_ms)))

            ax1.plot(x, throughputs, color=CLR["cyan"], linewidth=1, alpha=0.7)
            mean_t = np.mean(throughputs)
            ax1.axhline(y=mean_t, linestyle=":", color=CLR["orange"],
                       label=f"Mean: {mean_t:.0f} tok/s")
            ax1.fill_between(x, np.percentile(throughputs, 5), np.percentile(throughputs, 95),
                            alpha=0.1, color=CLR["cyan"], label="P5-P95 band")
            ax1.set_ylabel("Tokens/sec")
            ax1.set_title("Sustained Throughput", fontweight="bold")
            ax1.legend(); ax1.grid(alpha=0.3)

            ax2.plot(x, latencies_ms, color=CLR["orange"], linewidth=1, alpha=0.7)
            p99 = np.percentile(latencies_ms, 99)
            ax2.axhline(y=p99, linestyle="--", color=CLR["red"],
                       label=f"P99: {p99:.2f}ms")
            ax2.axhline(y=np.median(latencies_ms), linestyle=":",
                       color=CLR["green"], label=f"Median: {np.median(latencies_ms):.2f}ms")
            ax2.set_xlabel("Iteration"); ax2.set_ylabel("Latency (ms)")
            ax2.set_title("Latency Stability", fontweight="bold")
            ax2.legend(); ax2.grid(alpha=0.3)

            fig.suptitle("Sustained Performance Analysis", fontsize=14, fontweight="bold", y=1.01)
            fig.tight_layout()
            _save(fig, os.path.join(output_dir, "sustained_throughput.png"))

    if PLOTLY and len(latencies_ms) > 2:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            subplot_titles=["Throughput", "Latency"])
        x = list(range(len(latencies_ms)))
        fig.add_trace(go.Scatter(x=x, y=throughputs, mode="lines", name="tok/s",
                                 line=dict(color=CLR["cyan"])), row=1, col=1)
        fig.add_trace(go.Scatter(x=x, y=latencies_ms, mode="lines", name="Latency",
                                 line=dict(color=CLR["orange"])), row=2, col=1)
        fig.update_layout(title="Sustained Performance", template="plotly_dark", height=600)
        _save_plotly(fig, os.path.join(output_dir, "sustained_throughput_plotly.png"))


def plot_numerical_stability(
    results: Dict[str, Dict],
    output_dir: str = "viz_stress",
):
    os.makedirs(output_dir, exist_ok=True)

    if MPL:
        with plt.rc_context(DARK):
            fig, ax = plt.subplots(figsize=(10, max(3, len(results) * 0.6 + 1)))
            ax.axis("off")

            scenarios = list(results.keys())
            statuses = [results[s].get("status", "?") for s in scenarios]
            max_vals = [f"{results[s].get('max_abs_output', 'N/A')}" for s in scenarios]
            nans = [str(results[s].get("has_nan", "?")) for s in scenarios]

            cell_text = [[s, st, mv, n] for s, st, mv, n in
                        zip(scenarios, statuses, max_vals, nans)]
            colors = [["#161b22"]*4 for _ in scenarios]
            for i, st in enumerate(statuses):
                c = "#1a3a2a" if st == "PASS" else "#3a1a1a"
                colors[i] = [c]*4

            table = ax.table(cellText=cell_text,
                           colLabels=["Scenario", "Status", "Max |Output|", "Has NaN"],
                           cellColours=colors,
                           loc="center", cellLoc="center")
            table.auto_set_font_size(False)
            table.set_fontsize(11)
            table.scale(1, 1.5)

            for key, cell in table.get_celld().items():
                cell.set_edgecolor("#30363d")
                cell.set_text_props(color="#c9d1d9")
                if key[0] == 0:
                    cell.set_facecolor("#21262d")
                    cell.set_text_props(color="#c9d1d9", fontweight="bold")

            ax.set_title("Numerical Stability Test Results",
                        fontsize=14, fontweight="bold", pad=20)
            _save(fig, os.path.join(output_dir, "numerical_stability.png"))


def plot_compilation_overhead(
    results: Dict,
    output_dir: str = "viz_stress",
):
    os.makedirs(output_dir, exist_ok=True)

    if MPL and results:
        with plt.rc_context(DARK):
            labels = [f"N={k}" for k in results.keys()]
            compile_ms = [v.get("compile_overhead_ms", 0) for v in results.values()]
            cached_ms = [v.get("cached_call_ms", 0) for v in results.values()]

            fig, ax = plt.subplots(figsize=(10, 6))
            x = np.arange(len(labels)); w = 0.35
            ax.bar(x - w/2, compile_ms, w, label="Compilation", color=CLR["orange"], alpha=0.85)
            ax.bar(x + w/2, cached_ms, w, label="Cached Exec", color=CLR["green"], alpha=0.85)
            ax.set_xticks(x); ax.set_xticklabels(labels)
            ax.set_ylabel("Time (ms)")
            ax.set_title("XLA Compilation vs Cached Execution", fontsize=14, fontweight="bold")
            ax.legend(); ax.grid(axis="y", alpha=0.3)

            for i in range(len(labels)):
                if cached_ms[i] > 0:
                    ratio = compile_ms[i] / cached_ms[i]
                    ax.text(i, compile_ms[i] + 10, f"{ratio:.0f}×",
                           ha="center", fontsize=9, color=CLR["orange"])

            _save(fig, os.path.join(output_dir, "compilation_overhead.png"))


def plot_mega_stress_summary(
    kernel_results: List[Dict],
    output_dir: str = "viz_stress",
):
    os.makedirs(output_dir, exist_ok=True)

    sparse_r = [r for r in kernel_results if r.get("type") == "sparse"]
    dense_r = [r for r in kernel_results if r.get("type") == "dense"]

    if PLOTLY:
        fig = go.Figure()
        for label, data, color in [("Sparse", sparse_r, CLR["blue"]),
                                    ("Dense", dense_r, CLR["red"])]:
            for r in data:
                cfg = r.get("config", [0, 0, 0, 0])
                est = r.get("estimated_ram_gb", 0)
                status = r.get("status", "?")
                symbol = "circle" if status == "PASS" else "x"
                fig.add_trace(go.Scatter(
                    x=[est], y=[r.get("elapsed_ms", 0) or 0],
                    mode="markers", name=f"{label} B={cfg[0]} N={cfg[1]}",
                    marker=dict(color=color if status == "PASS" else "#d29922",
                               size=12, symbol=symbol),
                    showlegend=False,
                ))

        fig.update_layout(
            title="Mega Stress: Estimated RAM vs Execution Time",
            template="plotly_dark",
            xaxis_title="Estimated RAM (GB)",
            yaxis_title="Execution Time (ms)",
        )
        _save_plotly(fig, os.path.join(output_dir, "mega_stress_scatter_plotly.png"))


def plot_grad_accum_throughput(
    results: List[Dict],
    output_dir: str = "viz_stress",
):
    os.makedirs(output_dir, exist_ok=True)

    passed = [r for r in results if r.get("status") == "PASS"]
    if not passed:
        return

    if MPL:
        with plt.rc_context(DARK):
            labels = [f"eff_B={r['effective_batch']}\nN={r['seq_len']}" for r in passed]
            tps = [r.get("tokens_per_sec", 0) for r in passed]

            fig, ax = plt.subplots(figsize=(12, 6))
            ax.bar(labels, tps, color=CLR["cyan"], alpha=0.85, edgecolor="#30363d")
            ax.set_ylabel("Tokens/sec")
            ax.set_title("Gradient Accumulation Throughput", fontsize=14, fontweight="bold")
            ax.grid(axis="y", alpha=0.3)

            for i, v in enumerate(tps):
                ax.text(i, v + max(tps)*0.02, f"{v:.0f}", ha="center", fontsize=9, color=CLR["cyan"])

            _save(fig, os.path.join(output_dir, "grad_accum_throughput.png"))


def generate_stress_viz(
    stress_json: str = "benchmark_results/stress_test_all.json",
    mega_json: str = "benchmark_results/mega_stress_all.json",
    output_dir: str = "benchmark_results/viz_stress",
):
    print(f"\n{'='*70}")
    print("GENERATING STRESS TEST VISUALIZATIONS")
    print(f"{'='*70}")

    try:
        with open(stress_json) as f:
            stress = json.load(f)
    except Exception:
        stress = {}

    if "oom_sparse" in stress:
        oom = stress["oom_sparse"]
        plot_oom_heatmap(oom["seq_lengths"], oom["batch_sizes"],
                        oom["status_matrix"], "Sparse OOM Boundary", output_dir)

    if "oom_dense" in stress:
        oom = stress["oom_dense"]
        plot_oom_heatmap(oom["seq_lengths"], oom["batch_sizes"],
                        oom["status_matrix"], "Dense OOM Boundary", output_dir)

    if "sustained" in stress:
        ts = stress["sustained"].get("time_series", {})
        if ts.get("latencies_ms") and ts.get("throughputs"):
            plot_sustained_throughput(ts["latencies_ms"], ts["throughputs"], output_dir)

    if "numerical" in stress:
        plot_numerical_stability(stress["numerical"], output_dir)

    if "compilation" in stress:
        plot_compilation_overhead(stress["compilation"], output_dir)

    try:
        with open(mega_json) as f:
            mega = json.load(f)
    except Exception:
        mega = {}

    if "ceiling" in mega:
        for attn_type in ["sparse", "dense"]:
            entries = mega["ceiling"].get(attn_type, [])
            if entries:
                adapted = []
                for e in entries:
                    adapted.append({
                        "config": e.get("config", [0,0,0,0]),
                        "type": attn_type,
                        "status": e.get("status", "?"),
                        "elapsed_ms": e.get("ms", 0),
                        "estimated_ram_gb": e.get("est_gb", 0),
                    })
                plot_mega_stress_summary(adapted, output_dir)

    if "kernel" in mega:
        plot_mega_stress_summary(mega["kernel"].get("results", []), output_dir)

    if "accumulation" in mega:
        results = mega["accumulation"]
        if isinstance(results, dict):
            results = results.get("results", [])
        plot_grad_accum_throughput(results, output_dir)

    print(f"  All stress visuals -> {output_dir}")


if __name__ == "__main__":
    generate_stress_viz()

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


def plot_scaling(
    seq_lengths: List[int],
    dense_latencies: List[float],
    sparse_latencies: List[float],
    output_dir: str = "viz_scaling",
):
    os.makedirs(output_dir, exist_ok=True)

    if MPL and len(seq_lengths) > 1:
        with plt.rc_context(DARK):
            fig, ax = plt.subplots(figsize=(10, 7))
            ax.loglog(seq_lengths, dense_latencies, "o-", color=CLR["red"],
                     linewidth=2, markersize=8, label="Dense O(N²)")
            ax.loglog(seq_lengths, sparse_latencies, "s-", color=CLR["blue"],
                     linewidth=2, markersize=8, label="Sparse (sub-quadratic)")

            base = dense_latencies[0] if dense_latencies[0] > 0 else 1
            ref_n2 = [base * (n / seq_lengths[0]) ** 2 for n in seq_lengths]
            ax.loglog(seq_lengths, ref_n2, "--", color="#8b949e", alpha=0.5, label="O(N²) reference")

            ax.set_xlabel("Sequence Length (N)")
            ax.set_ylabel("Latency (ms)")
            ax.set_title("Scaling: Dense vs Sparse Attention", fontsize=14, fontweight="bold")
            ax.legend(fontsize=11); ax.grid(True, alpha=0.3, which="both")
            _save(fig, os.path.join(output_dir, "scaling_loglog.png"))

    if PLOTLY and len(seq_lengths) > 1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=seq_lengths, y=dense_latencies, mode="lines+markers",
                                 name="Dense", line=dict(color=CLR["red"])))
        fig.add_trace(go.Scatter(x=seq_lengths, y=sparse_latencies, mode="lines+markers",
                                 name="Sparse", line=dict(color=CLR["blue"])))
        fig.update_xaxes(type="log"); fig.update_yaxes(type="log")
        fig.update_layout(title="Scaling Analysis (log-log)", template="plotly_dark",
                          xaxis_title="Seq Length", yaxis_title="Latency (ms)")
        _save_plotly(fig, os.path.join(output_dir, "scaling_loglog_plotly.png"))


def plot_roofline(
    points: List[Dict],
    peak_tflops: float = 197.0,
    peak_bw_gb_s: float = 820.0,
    output_dir: str = "viz_scaling",
):
    os.makedirs(output_dir, exist_ok=True)
    ridge = peak_tflops * 1000 / peak_bw_gb_s

    if MPL and points:
        with plt.rc_context(DARK):
            fig, ax = plt.subplots(figsize=(12, 7))

            ai_range = np.logspace(-2, 4, 500)
            roofline = np.minimum(peak_tflops, ai_range * peak_bw_gb_s / 1000)
            ax.loglog(ai_range, roofline, "-", color=CLR["green"], linewidth=2.5,
                     label=f"Roofline ({peak_tflops:.0f} TF, {peak_bw_gb_s:.0f} GB/s)")

            ax.axvline(x=ridge, linestyle=":", color="#8b949e", alpha=0.5)
            ax.text(ridge * 1.1, peak_tflops * 0.5, f"Ridge\n{ridge:.1f} F/B",
                   fontsize=9, color="#8b949e")

            for p in points:
                ai = p.get("arithmetic_intensity", 0)
                tf = p.get("achieved_tflops", 0)
                name = p.get("name", "")
                is_sparse = "sparse" in p.get("type", "").lower()
                color = CLR["blue"] if is_sparse else CLR["red"]
                marker = "s" if is_sparse else "o"

                if ai > 0 and tf > 0:
                    ax.plot(ai, tf, marker, color=color, markersize=10, alpha=0.9)
                    ax.annotate(name, (ai, tf), textcoords="offset points",
                               xytext=(8, 8), fontsize=7, color=color)

            ax.fill_between(ai_range[ai_range < ridge], 0, roofline[ai_range < ridge],
                          alpha=0.05, color=CLR["orange"])
            ax.fill_between(ai_range[ai_range >= ridge], 0, roofline[ai_range >= ridge],
                          alpha=0.05, color=CLR["cyan"])
            ax.text(ridge * 0.1, peak_tflops * 0.02, "MEMORY\nBOUND",
                   fontsize=11, color=CLR["orange"], alpha=0.7, fontweight="bold")
            ax.text(ridge * 10, peak_tflops * 0.02, "COMPUTE\nBOUND",
                   fontsize=11, color=CLR["cyan"], alpha=0.7, fontweight="bold")

            ax.set_xlabel("Arithmetic Intensity (FLOPs/Byte)")
            ax.set_ylabel("Performance (TFLOPs)")
            ax.set_title("Roofline Model — TPU v5e", fontsize=14, fontweight="bold")
            ax.legend(); ax.grid(True, alpha=0.2, which="both")
            ax.set_xlim(0.01, 10000)
            _save(fig, os.path.join(output_dir, "roofline.png"))

    if PLOTLY and points:
        ai_range_list = np.logspace(-2, 4, 200).tolist()
        roofline_list = [min(peak_tflops, ai * peak_bw_gb_s / 1000) for ai in ai_range_list]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ai_range_list, y=roofline_list, mode="lines",
                                 name="Roofline", line=dict(color=CLR["green"], width=3)))
        for p in points:
            ai = p.get("arithmetic_intensity", 0)
            tf = p.get("achieved_tflops", 0)
            is_sparse = "sparse" in p.get("type", "").lower()
            if ai > 0 and tf > 0:
                fig.add_trace(go.Scatter(
                    x=[ai], y=[tf], mode="markers+text", name=p.get("name", ""),
                    marker=dict(size=12, color=CLR["blue"] if is_sparse else CLR["red"]),
                    text=[p.get("name", "")], textposition="top right",
                ))
        fig.update_xaxes(type="log"); fig.update_yaxes(type="log")
        fig.update_layout(title="Roofline Model", template="plotly_dark",
                          xaxis_title="Arithmetic Intensity (FLOPs/B)",
                          yaxis_title="TFLOPs")
        _save_plotly(fig, os.path.join(output_dir, "roofline_plotly.png"))


def plot_mfu_hfu(
    configs: List[str],
    mfu_dense: List[float],
    mfu_sparse: List[float],
    hfu_dense: List[float],
    hfu_sparse: List[float],
    output_dir: str = "viz_scaling",
):
    os.makedirs(output_dir, exist_ok=True)

    if MPL:
        with plt.rc_context(DARK):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            x = np.arange(len(configs)); w = 0.35

            ax1.bar(x - w/2, mfu_dense, w, label="Dense", color=CLR["red"], alpha=0.85)
            ax1.bar(x + w/2, mfu_sparse, w, label="Sparse", color=CLR["blue"], alpha=0.85)
            ax1.set_xticks(x); ax1.set_xticklabels(configs, rotation=25, ha="right")
            ax1.set_ylabel("MFU (%)"); ax1.set_title("Model FLOPs Utilization", fontweight="bold")
            ax1.legend(); ax1.grid(axis="y", alpha=0.3)

            ax2.bar(x - w/2, hfu_dense, w, label="Dense", color=CLR["red"], alpha=0.85)
            ax2.bar(x + w/2, hfu_sparse, w, label="Sparse", color=CLR["blue"], alpha=0.85)
            ax2.set_xticks(x); ax2.set_xticklabels(configs, rotation=25, ha="right")
            ax2.set_ylabel("HFU (%)"); ax2.set_title("Hardware FLOPs Utilization", fontweight="bold")
            ax2.legend(); ax2.grid(axis="y", alpha=0.3)

            fig.suptitle("Compute Efficiency", fontsize=14, fontweight="bold", y=1.01)
            fig.tight_layout()
            _save(fig, os.path.join(output_dir, "mfu_hfu.png"))


def plot_arithmetic_intensity(
    configs: List[str],
    ai_dense: List[float],
    ai_sparse: List[float],
    ridge_point: float = 240.0,
    output_dir: str = "viz_scaling",
):
    os.makedirs(output_dir, exist_ok=True)

    if MPL:
        with plt.rc_context(DARK):
            fig, ax = plt.subplots(figsize=(12, 6))
            x = np.arange(len(configs)); w = 0.35
            ax.bar(x - w/2, ai_dense, w, label="Dense", color=CLR["red"], alpha=0.85)
            ax.bar(x + w/2, ai_sparse, w, label="Sparse", color=CLR["blue"], alpha=0.85)
            ax.axhline(y=ridge_point, linestyle="--", color=CLR["green"],
                      label=f"Ridge Point ({ridge_point:.0f})")
            ax.set_xticks(x); ax.set_xticklabels(configs, rotation=25, ha="right")
            ax.set_ylabel("Arithmetic Intensity (FLOPs/Byte)")
            ax.set_title("Arithmetic Intensity", fontsize=14, fontweight="bold")
            ax.legend(); ax.grid(axis="y", alpha=0.3)
            _save(fig, os.path.join(output_dir, "arithmetic_intensity.png"))


def generate_scaling_viz(
    roofline_json: str = "benchmark_results/roofline_analysis.json",
    output_dir: str = "benchmark_results/viz_scaling",
):
    print(f"\n{'='*70}")
    print("GENERATING SCALING & ROOFLINE VISUALIZATIONS")
    print(f"{'='*70}")

    try:
        with open(roofline_json) as f:
            data = json.load(f)
    except Exception as e:
        print(f"  Could not load {roofline_json}: {e}")
        return

    results = data.get("results", [])
    roofline_pts = data.get("roofline_points", [])
    hw = data.get("hardware", {})

    dense_r = [r for r in results if r["type"] == "dense"]
    sparse_r = [r for r in results if r["type"] == "sparse"]

    if dense_r and sparse_r:
        seq_lens = sorted(set(r["seq_len"] for r in dense_r))
        d_lat = [next((r["latency_mean_ms"] for r in dense_r if r["seq_len"] == n), 0) for n in seq_lens]
        s_lat = [next((r["latency_mean_ms"] for r in sparse_r if r["seq_len"] == n), 0) for n in seq_lens]
        plot_scaling(seq_lens, d_lat, s_lat, output_dir)

    if roofline_pts:
        plot_roofline(roofline_pts,
                     peak_tflops=hw.get("peak_tflops_bf16", 197),
                     peak_bw_gb_s=hw.get("hbm_bandwidth_gb_s", 820),
                     output_dir=output_dir)

    if dense_r and sparse_r:
        configs = [f"N={r['seq_len']}" for r in dense_r]
        plot_mfu_hfu(
            configs,
            [r.get("mfu_pct", 0) for r in dense_r],
            [r.get("mfu_pct", 0) for r in sparse_r],
            [r.get("hfu_pct", 0) for r in dense_r],
            [r.get("hfu_pct", 0) for r in sparse_r],
            output_dir,
        )
        ridge = hw.get("peak_tflops_bf16", 197) * 1000 / hw.get("hbm_bandwidth_gb_s", 820)
        plot_arithmetic_intensity(
            configs,
            [r.get("arithmetic_intensity", 0) for r in dense_r],
            [r.get("arithmetic_intensity", 0) for r in sparse_r],
            ridge, output_dir,
        )

    print(f"  ✅ All scaling visuals → {output_dir}")


if __name__ == "__main__":
    generate_scaling_viz()

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as mticker

STYLE = {
    "figure.facecolor": "#0d1117",
    "axes.facecolor": "#161b22",
    "axes.edgecolor": "#30363d",
    "axes.labelcolor": "#c9d1d9",
    "text.color": "#c9d1d9",
    "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",
    "grid.color": "#21262d",
    "grid.alpha": 0.6,
    "font.family": "sans-serif",
    "font.size": 11,
}

COLORS = {
    "sparse": "#58a6ff",
    "dense": "#f85149",
    "accent": "#3fb950",
    "warning": "#d29922",
    "purple": "#bc8cff",
    "cyan": "#39d2c0",
    "orange": "#f0883e",
    "pink": "#f778ba",
}

GRADIENT_CMAP = LinearSegmentedColormap.from_list(
    "custom_blue", ["#0d1117", "#1f6feb", "#58a6ff", "#79c0ff"], N=256
)


def _apply_style():
    plt.rcParams.update(STYLE)


def _save_fig(fig, path: str, dpi: int = 150):
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  📊 Saved: {path}")



def plot_block_mask(
    mask_array: np.ndarray,
    title: str = "Block-Sparse Attention Mask",
    sparsity: float = 0.0,
    output_path: str = "plots/block_mask.png",
):
    _apply_style()
    fig, ax = plt.subplots(figsize=(8, 7))

    cmap = LinearSegmentedColormap.from_list("mask", ["#161b22", "#58a6ff"], N=2)
    im = ax.imshow(mask_array.astype(float), cmap=cmap, aspect="equal",
                   interpolation="nearest")
    ax.set_xlabel("Key-Value Block Index", fontsize=12, fontweight="bold")
    ax.set_ylabel("Query Block Index", fontsize=12, fontweight="bold")
    ax.set_title(f"{title}\nSparsity: {sparsity:.1%} of blocks skipped",
                 fontsize=14, fontweight="bold", pad=15)

    n_blocks = mask_array.shape[0]
    if n_blocks <= 16:
        ax.set_xticks(range(n_blocks))
        ax.set_yticks(range(n_blocks))

    if n_blocks <= 16:
        for i in range(mask_array.shape[0]):
            for j in range(mask_array.shape[1]):
                color = "#0d1117" if mask_array[i, j] else "#30363d"
                ax.text(j, i, "■" if mask_array[i, j] else "·",
                        ha="center", va="center", color=color, fontsize=8)

    active = int(np.sum(mask_array))
    total = mask_array.size
    ax.text(0.02, -0.08, f"Active: {active}/{total} blocks",
            transform=ax.transAxes, fontsize=10, color=COLORS["accent"])

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_ticks([0.25, 0.75])
    cbar.set_ticklabels(["Skipped", "Computed"])

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    _save_fig(fig, output_path)



def plot_sparsity_patterns(
    masks: Dict[str, np.ndarray],
    sparsities: Dict[str, float],
    output_path: str = "plots/sparsity_patterns.png",
):
    _apply_style()
    n = len(masks)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    cmap = LinearSegmentedColormap.from_list("mask", ["#161b22", "#58a6ff"], N=2)
    for ax, (name, mask) in zip(axes, masks.items()):
        ax.imshow(mask.astype(float), cmap=cmap, aspect="equal",
                  interpolation="nearest")
        sp = sparsities.get(name, 0)
        ax.set_title(f"{name}\n({sp:.0%} sparse)", fontsize=11, fontweight="bold")
        ax.set_xlabel("KV Block")
        ax.set_ylabel("Q Block")

    fig.suptitle("Sparsity Pattern Comparison", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    _save_fig(fig, output_path)



def plot_latency_comparison(
    comparisons: List[Dict],
    output_path: str = "plots/latency_comparison.png",
):
    _apply_style()
    fig, ax = plt.subplots(figsize=(12, 6))

    labels = [f"B={c['batch_size']}\nN={c['seq_len']}" for c in comparisons]
    dense_times = [c["dense_latency_ms"] for c in comparisons]
    sparse_times = [c["sparse_latency_ms"] for c in comparisons]

    x = np.arange(len(labels))
    width = 0.35

    bars_d = ax.bar(x - width / 2, dense_times, width, label="Dense Attention",
                    color=COLORS["dense"], alpha=0.9, edgecolor="#30363d", linewidth=0.5)
    bars_s = ax.bar(x + width / 2, sparse_times, width, label="Sparse Attention",
                    color=COLORS["sparse"], alpha=0.9, edgecolor="#30363d", linewidth=0.5)

    for i, (d, s) in enumerate(zip(dense_times, sparse_times)):
        if s > 0:
            speedup = d / s
            ax.text(i, max(d, s) * 1.05, f"{speedup:.2f}×",
                    ha="center", fontsize=9, fontweight="bold", color=COLORS["accent"])

    ax.set_ylabel("Latency (ms)", fontsize=12, fontweight="bold")
    ax.set_title("Latency: Sparse vs Dense Attention", fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(framealpha=0.3, edgecolor="#30363d")
    ax.grid(axis="y", alpha=0.3)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    _save_fig(fig, output_path)



def plot_hbm_waterfall(
    components: Dict[str, Tuple[float, float]],
    output_path: str = "plots/hbm_waterfall.png",
):
    _apply_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    names = list(components.keys())
    dense_vals = [components[n][0] for n in names]
    sparse_vals = [components[n][1] for n in names]
    savings = [d - s for d, s in zip(dense_vals, sparse_vals)]

    x = np.arange(len(names))
    width = 0.3

    ax.bar(x - width / 2, dense_vals, width, label="Dense", color=COLORS["dense"], alpha=0.9)
    ax.bar(x + width / 2, sparse_vals, width, label="Sparse", color=COLORS["sparse"], alpha=0.9)

    for i, (d, s, saved) in enumerate(zip(dense_vals, sparse_vals, savings)):
        if d > 0:
            pct = saved / d * 100
            ax.annotate(f"-{pct:.0f}%", xy=(i, max(d, s) * 1.02),
                        fontsize=9, ha="center", color=COLORS["accent"], fontweight="bold")

    ax.set_ylabel("HBM Bandwidth (MB)", fontsize=12, fontweight="bold")
    ax.set_title("HBM Bandwidth: Component Breakdown", fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.legend(framealpha=0.3, edgecolor="#30363d")
    ax.grid(axis="y", alpha=0.3)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    _save_fig(fig, output_path)



def plot_throughput(
    comparisons: List[Dict],
    output_path: str = "plots/throughput.png",
):
    _apply_style()
    fig, ax = plt.subplots(figsize=(12, 6))

    labels = [f"B={c['batch_size']}\nN={c['seq_len']}" for c in comparisons]
    dense_tps = [c.get("dense_tokens_per_sec", 0) for c in comparisons]
    sparse_tps = [c.get("sparse_tokens_per_sec", 0) for c in comparisons]

    x = np.arange(len(labels))
    width = 0.35

    ax.bar(x - width / 2, [t / 1000 for t in dense_tps], width,
           label="Dense", color=COLORS["dense"], alpha=0.9)
    ax.bar(x + width / 2, [t / 1000 for t in sparse_tps], width,
           label="Sparse", color=COLORS["sparse"], alpha=0.9)

    ax.set_ylabel("Throughput (K tokens/sec)", fontsize=12, fontweight="bold")
    ax.set_title("Throughput Comparison", fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(framealpha=0.3, edgecolor="#30363d")
    ax.grid(axis="y", alpha=0.3)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    _save_fig(fig, output_path)



def plot_scaling(
    seq_lengths: List[int],
    dense_latencies: List[float],
    sparse_latencies: List[float],
    output_path: str = "plots/scaling.png",
):
    _apply_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.loglog(seq_lengths, dense_latencies, "o-", color=COLORS["dense"],
               label="Dense O(N²)", linewidth=2, markersize=8)
    ax1.loglog(seq_lengths, sparse_latencies, "s-", color=COLORS["sparse"],
               label="Sparse (block-sparse)", linewidth=2, markersize=8)

    ref_n2 = [dense_latencies[0] * (n / seq_lengths[0]) ** 2 for n in seq_lengths]
    ax1.loglog(seq_lengths, ref_n2, "--", color="#8b949e", alpha=0.5, label="O(N²) reference")

    ax1.set_xlabel("Sequence Length (N)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Latency (ms)", fontsize=12, fontweight="bold")
    ax1.set_title("Scaling: Latency vs Sequence Length", fontsize=13, fontweight="bold")
    ax1.legend(framealpha=0.3, edgecolor="#30363d")
    ax1.grid(True, alpha=0.3, which="both")

    speedups = [d / s if s > 0 else 0 for d, s in zip(dense_latencies, sparse_latencies)]
    ax2.plot(seq_lengths, speedups, "D-", color=COLORS["accent"], linewidth=2, markersize=8)
    ax2.axhline(y=1.0, color="#8b949e", linestyle="--", alpha=0.5, label="1× (no speedup)")
    ax2.set_xlabel("Sequence Length (N)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Speedup (Dense / Sparse)", fontsize=12, fontweight="bold")
    ax2.set_title("Speedup vs Sequence Length", fontsize=13, fontweight="bold")
    ax2.legend(framealpha=0.3, edgecolor="#30363d")
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Scaling Analysis", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    _save_fig(fig, output_path)



def plot_roofline(
    results: List[Dict],
    peak_tflops: float = 197.0,
    peak_bw_gb_s: float = 820.0,
    output_path: str = "plots/roofline.png",
):
    _apply_style()
    fig, ax = plt.subplots(figsize=(10, 7))

    ridge_point = peak_tflops / peak_bw_gb_s
    ai_range = np.logspace(-2, 4, 1000)
    roofline = np.minimum(peak_tflops, ai_range * peak_bw_gb_s / 1000)

    ax.loglog(ai_range, roofline, "-", color=COLORS["warning"], linewidth=2.5,
              label=f"Roofline (peak={peak_tflops:.0f} TFLOPs, BW={peak_bw_gb_s:.0f} GB/s)")

    ax.fill_between(ai_range, 0, roofline, where=(ai_range < ridge_point),
                    alpha=0.08, color=COLORS["sparse"])
    ax.fill_between(ai_range, 0, roofline, where=(ai_range >= ridge_point),
                    alpha=0.08, color=COLORS["dense"])

    ax.text(0.03, peak_tflops * 0.3, "Memory\nBound", color=COLORS["sparse"],
            fontsize=12, fontweight="bold", alpha=0.7)
    ax.text(ridge_point * 5, peak_tflops * 0.3, "Compute\nBound", color=COLORS["dense"],
            fontsize=12, fontweight="bold", alpha=0.7)

    markers = {"sparse": "s", "dense": "o"}
    colors_map = {"sparse": COLORS["sparse"], "dense": COLORS["dense"]}

    for r in results:
        ai = r.get("arithmetic_intensity", 1)
        perf = r.get("achieved_tflops", 0)
        attn_type = r.get("type", "dense")
        name = r.get("name", "")
        marker = markers.get(attn_type, "^")
        color = colors_map.get(attn_type, COLORS["purple"])
        ax.loglog(ai, perf, marker, color=color, markersize=12, markeredgecolor="white",
                  markeredgewidth=1.5, zorder=5)
        ax.annotate(name, (ai, perf), textcoords="offset points", xytext=(8, 4),
                    fontsize=8, color=color)

    ax.axvline(x=ridge_point, color="#8b949e", linestyle=":", alpha=0.5)
    ax.set_xlabel("Arithmetic Intensity (FLOPs / Byte)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Performance (TFLOPs)", fontsize=12, fontweight="bold")
    ax.set_title("Roofline Model Analysis", fontsize=14, fontweight="bold", pad=15)
    ax.legend(framealpha=0.3, edgecolor="#30363d", loc="lower right")
    ax.grid(True, alpha=0.2, which="both")
    ax.set_xlim(0.01, 10000)
    ax.set_ylim(0.001, peak_tflops * 2)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    _save_fig(fig, output_path)



def plot_memory_usage(
    configs: List[str],
    dense_mem_mb: List[float],
    sparse_mem_mb: List[float],
    output_path: str = "plots/memory_usage.png",
):
    _apply_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(configs))
    width = 0.35

    ax.bar(x - width / 2, dense_mem_mb, width, label="Dense",
           color=COLORS["dense"], alpha=0.9)
    ax.bar(x + width / 2, sparse_mem_mb, width, label="Sparse",
           color=COLORS["sparse"], alpha=0.9)

    for i, (d, s) in enumerate(zip(dense_mem_mb, sparse_mem_mb)):
        if d > 0:
            saved = (d - s) / d * 100
            ax.text(i, max(d, s) * 1.03, f"-{saved:.0f}%",
                    ha="center", fontsize=9, color=COLORS["accent"], fontweight="bold")

    ax.set_ylabel("Memory (MB)", fontsize=12, fontweight="bold")
    ax.set_title("HBM Memory: Dense vs Sparse", fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(configs)
    ax.legend(framealpha=0.3, edgecolor="#30363d")
    ax.grid(axis="y", alpha=0.3)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    _save_fig(fig, output_path)



def plot_latency_distribution(
    dense_times: List[float],
    sparse_times: List[float],
    config_label: str = "B=4, N=2048",
    output_path: str = "plots/latency_distribution.png",
):
    _apply_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    bp = ax1.boxplot(
        [dense_times, sparse_times],
        labels=["Dense", "Sparse"],
        patch_artist=True,
        widths=0.5,
        medianprops=dict(color=COLORS["accent"], linewidth=2),
        whiskerprops=dict(color="#8b949e"),
        capprops=dict(color="#8b949e"),
    )
    bp["boxes"][0].set_facecolor(COLORS["dense"])
    bp["boxes"][0].set_alpha(0.7)
    bp["boxes"][1].set_facecolor(COLORS["sparse"])
    bp["boxes"][1].set_alpha(0.7)

    for i, times in enumerate([dense_times, sparse_times]):
        p50 = np.percentile(times, 50)
        p99 = np.percentile(times, 99)
        ax1.annotate(f"P50: {p50:.2f}ms\nP99: {p99:.2f}ms",
                     xy=(i + 1, p99), textcoords="offset points", xytext=(30, 10),
                     fontsize=8, color=COLORS["accent"],
                     arrowprops=dict(arrowstyle="->", color=COLORS["accent"], lw=0.8))

    ax1.set_ylabel("Latency (ms)", fontsize=12, fontweight="bold")
    ax1.set_title(f"Latency Distribution ({config_label})", fontsize=12, fontweight="bold")
    ax1.grid(axis="y", alpha=0.3)

    bins = 20
    ax2.hist(dense_times, bins=bins, alpha=0.6, color=COLORS["dense"], label="Dense", edgecolor="#30363d")
    ax2.hist(sparse_times, bins=bins, alpha=0.6, color=COLORS["sparse"], label="Sparse", edgecolor="#30363d")

    ax2.axvline(np.percentile(dense_times, 99), color=COLORS["dense"], linestyle="--",
                alpha=0.8, label=f"Dense P99: {np.percentile(dense_times, 99):.2f}ms")
    ax2.axvline(np.percentile(sparse_times, 99), color=COLORS["sparse"], linestyle="--",
                alpha=0.8, label=f"Sparse P99: {np.percentile(sparse_times, 99):.2f}ms")

    ax2.set_xlabel("Latency (ms)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Count", fontsize=12, fontweight="bold")
    ax2.set_title("Latency Histogram", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=8, framealpha=0.3, edgecolor="#30363d")

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    _save_fig(fig, output_path)



def plot_mfu_hfu(
    configs: List[str],
    mfu_dense: List[float],
    mfu_sparse: List[float],
    hfu_dense: List[float],
    hfu_sparse: List[float],
    output_path: str = "plots/mfu_hfu.png",
):
    _apply_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    x = np.arange(len(configs))
    width = 0.35

    ax1.bar(x - width / 2, mfu_dense, width, label="Dense", color=COLORS["dense"], alpha=0.9)
    ax1.bar(x + width / 2, mfu_sparse, width, label="Sparse", color=COLORS["sparse"], alpha=0.9)
    ax1.set_ylabel("MFU (%)", fontsize=12, fontweight="bold")
    ax1.set_title("Model FLOPs Utilization (MFU)", fontsize=13, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(configs, fontsize=9)
    ax1.legend(framealpha=0.3, edgecolor="#30363d")
    ax1.grid(axis="y", alpha=0.3)
    ax1.set_ylim(0, 100)

    ax2.bar(x - width / 2, hfu_dense, width, label="Dense", color=COLORS["dense"], alpha=0.9)
    ax2.bar(x + width / 2, hfu_sparse, width, label="Sparse", color=COLORS["sparse"], alpha=0.9)
    ax2.set_ylabel("HFU (%)", fontsize=12, fontweight="bold")
    ax2.set_title("Hardware FLOPs Utilization (HFU)", fontsize=13, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(configs, fontsize=9)
    ax2.legend(framealpha=0.3, edgecolor="#30363d")
    ax2.grid(axis="y", alpha=0.3)
    ax2.set_ylim(0, 100)

    fig.suptitle("Compute Utilization Analysis", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    _save_fig(fig, output_path)



def plot_flops_breakdown(
    configs: List[str],
    dense_qk: List[float],
    dense_softmax: List[float],
    dense_av: List[float],
    sparse_qk: List[float],
    sparse_softmax: List[float],
    sparse_av: List[float],
    output_path: str = "plots/flops_breakdown.png",
):
    _apply_style()
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(configs))
    width = 0.35

    ax.bar(x - width / 2, dense_qk, width, label="Dense Q×K^T",
           color=COLORS["dense"], alpha=0.9)
    ax.bar(x - width / 2, dense_softmax, width, bottom=dense_qk,
           label="Dense Softmax", color=COLORS["orange"], alpha=0.9)
    bottom_d = [q + s for q, s in zip(dense_qk, dense_softmax)]
    ax.bar(x - width / 2, dense_av, width, bottom=bottom_d,
           label="Dense Attn×V", color=COLORS["pink"], alpha=0.9)

    ax.bar(x + width / 2, sparse_qk, width, label="Sparse Q×K^T",
           color=COLORS["sparse"], alpha=0.9)
    ax.bar(x + width / 2, sparse_softmax, width, bottom=sparse_qk,
           label="Sparse Softmax", color=COLORS["cyan"], alpha=0.9)
    bottom_s = [q + s for q, s in zip(sparse_qk, sparse_softmax)]
    ax.bar(x + width / 2, sparse_av, width, bottom=bottom_s,
           label="Sparse Attn×V", color=COLORS["purple"], alpha=0.9)

    ax.set_ylabel("GFLOPs", fontsize=12, fontweight="bold")
    ax.set_title("FLOPs Breakdown by Component", fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(configs)
    ax.legend(ncol=2, fontsize=8, framealpha=0.3, edgecolor="#30363d")
    ax.grid(axis="y", alpha=0.3)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    _save_fig(fig, output_path)



def plot_stress_results(
    seq_lengths: List[int],
    batch_sizes: List[int],
    status_matrix: np.ndarray,
    latency_matrix: Optional[np.ndarray] = None,
    output_path: str = "plots/stress_test.png",
):
    _apply_style()
    fig, axes = plt.subplots(1, 2 if latency_matrix is not None else 1,
                              figsize=(12 if latency_matrix is not None else 7, 6))
    if latency_matrix is None:
        axes = [axes]

    cmap = LinearSegmentedColormap.from_list("status", [COLORS["dense"], COLORS["accent"]], N=2)
    ax = axes[0]
    im = ax.imshow(status_matrix, cmap=cmap, aspect="auto", interpolation="nearest")
    ax.set_xticks(range(len(batch_sizes)))
    ax.set_xticklabels(batch_sizes)
    ax.set_yticks(range(len(seq_lengths)))
    ax.set_yticklabels(seq_lengths)
    ax.set_xlabel("Batch Size", fontsize=12, fontweight="bold")
    ax.set_ylabel("Sequence Length", fontsize=12, fontweight="bold")
    ax.set_title("Stress Test: OOM Boundary", fontsize=13, fontweight="bold")

    for i in range(len(seq_lengths)):
        for j in range(len(batch_sizes)):
            text = "✓" if status_matrix[i, j] else "✗"
            color = "#0d1117" if status_matrix[i, j] else "white"
            ax.text(j, i, text, ha="center", va="center", fontsize=14,
                    fontweight="bold", color=color)

    if latency_matrix is not None:
        ax2 = axes[1]
        im2 = ax2.imshow(latency_matrix, cmap=GRADIENT_CMAP, aspect="auto")
        ax2.set_xticks(range(len(batch_sizes)))
        ax2.set_xticklabels(batch_sizes)
        ax2.set_yticks(range(len(seq_lengths)))
        ax2.set_yticklabels(seq_lengths)
        ax2.set_xlabel("Batch Size", fontsize=12, fontweight="bold")
        ax2.set_ylabel("Sequence Length", fontsize=12, fontweight="bold")
        ax2.set_title("Latency Heatmap (ms)", fontsize=13, fontweight="bold")
        fig.colorbar(im2, ax=ax2, shrink=0.8)

        for i in range(len(seq_lengths)):
            for j in range(len(batch_sizes)):
                val = latency_matrix[i, j]
                if val > 0:
                    ax2.text(j, i, f"{val:.1f}", ha="center", va="center",
                             fontsize=8, color="white")

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    _save_fig(fig, output_path)



def plot_dashboard(
    summary: Dict,
    output_path: str = "plots/dashboard.png",
):
    _apply_style()
    fig = plt.figure(figsize=(20, 13))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    comparisons = summary.get("comparisons", [])
    headline = summary.get("headline", {})

    if not comparisons:
        fig.text(0.5, 0.5, "No comparison data available",
                 ha="center", va="center", fontsize=16, color="#8b949e")
        _save_fig(fig, output_path)
        return

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.axis("off")
    metrics = [
        ("HBM Reduction", f"{headline.get('avg_hbm_reduction_pct', 0):.1f}%", COLORS["accent"]),
        ("FLOPs Reduction", f"{headline.get('avg_flops_reduction_pct', 0):.1f}%", COLORS["sparse"]),
        ("Avg Speedup", f"{headline.get('avg_speedup', 0):.2f}×", COLORS["warning"]),
    ]
    ax1.set_title("Headline Metrics", fontsize=14, fontweight="bold", pad=20)
    for i, (label, value, color) in enumerate(metrics):
        y = 0.75 - i * 0.3
        ax1.text(0.5, y, value, ha="center", va="center", fontsize=32,
                 fontweight="bold", color=color, transform=ax1.transAxes)
        ax1.text(0.5, y - 0.1, label, ha="center", va="center", fontsize=11,
                 color="#8b949e", transform=ax1.transAxes)

    ax2 = fig.add_subplot(gs[0, 1])
    labels_short = []
    dense_lats = []
    sparse_lats = []
    for c in comparisons:
        labels_short.append(f"N={c.get('N', '?')}")
        dl = c.get("Dense Latency (ms)", "0")
        sl = c.get("Sparse Latency (ms)", "0")
        dense_lats.append(float(dl))
        sparse_lats.append(float(sl))

    x = np.arange(len(labels_short))
    if len(x) > 0:
        width = min(0.35, 0.8 / max(len(x), 1))
        ax2.bar(x - width / 2, dense_lats, width, color=COLORS["dense"],
                label="Dense", alpha=0.9)
        ax2.bar(x + width / 2, sparse_lats, width, color=COLORS["sparse"],
                label="Sparse", alpha=0.9)
    ax2.set_title("Latency Comparison", fontsize=13, fontweight="bold")
    ax2.set_ylabel("ms")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels_short, fontsize=8)
    ax2.legend(fontsize=8, framealpha=0.3)
    ax2.grid(axis="y", alpha=0.3)

    ax3 = fig.add_subplot(gs[0, 2])
    hbm_reductions = []
    for c in comparisons:
        val = c.get("HBM Reduction", "0%").replace("%", "")
        hbm_reductions.append(float(val))

    if hbm_reductions:
        colors_bar = [COLORS["accent"] if v >= 35 else COLORS["warning"] for v in hbm_reductions]
        ax3.bar(x, hbm_reductions, color=colors_bar, alpha=0.9, width=0.5)
        ax3.axhline(y=40, color=COLORS["dense"], linestyle="--", alpha=0.5, label="40% target")
    ax3.set_title("HBM Bandwidth Reduction", fontsize=13, fontweight="bold")
    ax3.set_ylabel("Reduction (%)")
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels_short, fontsize=8)
    ax3.legend(fontsize=8, framealpha=0.3)
    ax3.grid(axis="y", alpha=0.3)

    ax4 = fig.add_subplot(gs[1, 0])
    speedups = []
    for c in comparisons:
        val = c.get("Speedup", "1x").replace("x", "")
        speedups.append(float(val))

    if speedups:
        ax4.bar(x, speedups, color=COLORS["purple"], alpha=0.9, width=0.5)
        ax4.axhline(y=1.0, color="#8b949e", linestyle="--", alpha=0.5, label="1× baseline")
    ax4.set_title("Speedup (Dense / Sparse)", fontsize=13, fontweight="bold")
    ax4.set_ylabel("Speedup (×)")
    ax4.set_xticks(x)
    ax4.set_xticklabels(labels_short, fontsize=8)
    ax4.legend(fontsize=8, framealpha=0.3)
    ax4.grid(axis="y", alpha=0.3)

    ax5 = fig.add_subplot(gs[1, 1])
    flops_reds = []
    for c in comparisons:
        val = c.get("FLOPs Reduction", "0%").replace("%", "")
        flops_reds.append(float(val))

    if flops_reds:
        ax5.bar(x, flops_reds, color=COLORS["cyan"], alpha=0.9, width=0.5)
    ax5.set_title("FLOPs Reduction", fontsize=13, fontweight="bold")
    ax5.set_ylabel("Reduction (%)")
    ax5.set_xticks(x)
    ax5.set_xticklabels(labels_short, fontsize=8)
    ax5.grid(axis="y", alpha=0.3)

    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis("off")
    cfg = summary.get("config", {})
    info_text = (
        f"Configuration\n"
        f"─────────────────────\n"
        f"Device:     {summary.get('device', 'N/A')}\n"
        f"JAX:        {summary.get('jax_version', 'N/A')}\n"
        f"Pattern:    {cfg.get('sparsity_type', 'N/A')}\n"
        f"Block size: {cfg.get('block_size', 'N/A')}\n"
        f"Heads:      {cfg.get('n_heads', 'N/A')}×{cfg.get('d_head', 'N/A')}d\n"
        f"Warmup:     {cfg.get('n_warmup', 'N/A')}\n"
        f"Iterations: {cfg.get('n_iterations', 'N/A')}\n"
        f"─────────────────────\n"
        f"Timestamp:  {summary.get('timestamp', 'N/A')}"
    )
    ax6.text(0.1, 0.9, info_text, transform=ax6.transAxes, fontsize=10,
             verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#21262d", edgecolor="#30363d"))

    fig.suptitle("Sparse Attention Benchmark Dashboard",
                 fontsize=18, fontweight="bold", y=0.98, color=COLORS["accent"])

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    _save_fig(fig, output_path, dpi=120)



def generate_all_plots(
    results_json_path: str = "benchmark_results/benchmark_results.json",
    output_dir: str = "plots",
):
    print(f"\n{'='*60}")
    print("GENERATING VISUALIZATION PLOTS")
    print(f"{'='*60}")
    print(f"Input:  {results_json_path}")
    print(f"Output: {output_dir}/")

    with open(results_json_path) as f:
        summary = json.load(f)

    os.makedirs(output_dir, exist_ok=True)
    comparisons = summary.get("comparisons", [])

    plot_dashboard(summary, os.path.join(output_dir, "dashboard.png"))

    if comparisons:
        latency_data = []
        for c in comparisons:
            latency_data.append({
                "batch_size": c.get("B", 1),
                "seq_len": c.get("N", 1024),
                "dense_latency_ms": float(c.get("Dense Latency (ms)", 0)),
                "sparse_latency_ms": float(c.get("Sparse Latency (ms)", 0)),
            })
        plot_latency_comparison(latency_data, os.path.join(output_dir, "latency_comparison.png"))

        configs = [f"B={c.get('B')}\nN={c.get('N')}" for c in comparisons]
        dense_hbm = [float(c.get("Dense HBM (MB)", "0")) for c in comparisons]
        sparse_hbm = [float(c.get("Sparse HBM (MB)", "0")) for c in comparisons]
        plot_memory_usage(configs, dense_hbm, sparse_hbm,
                         os.path.join(output_dir, "memory_usage.png"))

    print(f"\n✅ All plots saved to {output_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    import sys
    json_path = sys.argv[1] if len(sys.argv) > 1 else "benchmark_results/benchmark_results.json"
    output = sys.argv[2] if len(sys.argv) > 2 else "plots"
    generate_all_plots(json_path, output)

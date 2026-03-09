from __future__ import annotations

import os
from typing import Dict, List

import numpy as np

try:
    from IPython.display import clear_output, display
    from IPython import get_ipython
except ImportError:
    clear_output = None
    display = None
    get_ipython = None

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    go = None
    make_subplots = None


def in_notebook() -> bool:
    if clear_output is None or display is None or get_ipython is None or go is None:
        return False
    try:
        shell = get_ipython()
    except Exception:
        return False
    return shell is not None and shell.__class__.__name__ == "ZMQInteractiveShell"


class LiveNotebookDisplay:
    def __init__(self, title: str, output_dir: str, filename: str, min_interval_steps: int = 10) -> None:
        self.title = title
        self.output_dir = output_dir
        self.filename = filename
        self.min_interval_steps = min_interval_steps
        self.enabled = in_notebook()
        self.last_step = -10**9

    def should_update(self, step: int) -> bool:
        return self.enabled and (step - self.last_step) >= self.min_interval_steps

    def show(self, fig, step: int) -> None:
        if not self.enabled:
            return
        os.makedirs(self.output_dir, exist_ok=True)
        fig.write_html(os.path.join(self.output_dir, self.filename))
        clear_output(wait=True)
        display(fig)
        self.last_step = step


def render_training_live(
    steps: List[int],
    losses: List[float],
    grad_norms: List[float],
    lrs: List[float],
    tokens_per_sec: List[float],
    eval_points: Dict[str, float] | None,
    output_dir: str,
    step: int,
    live: LiveNotebookDisplay,
) -> None:
    if not live.should_update(step):
        return
    fig = make_subplots(rows=2, cols=2, subplot_titles=("Loss", "Grad norm", "Learning rate", "Tokens/sec"))
    fig.add_trace(go.Scatter(x=steps, y=losses, mode="lines", name="loss", line=dict(color="#56ccf2")), row=1, col=1)
    fig.add_trace(go.Scatter(x=steps, y=grad_norms, mode="lines", name="grad", line=dict(color="#ff9f43")), row=1, col=2)
    fig.add_trace(go.Scatter(x=steps, y=lrs, mode="lines", name="lr", line=dict(color="#a29bfe")), row=2, col=1)
    fig.add_trace(go.Scatter(x=steps, y=tokens_per_sec, mode="lines", name="tok/s", line=dict(color="#2ed573")), row=2, col=2)
    title = "Training live dashboard"
    if eval_points:
        title += (
            f" | val_loss={eval_points['validation_loss']:.3f}"
            f" ppl={eval_points['perplexity']:.1f}"
            f" acc={eval_points['token_accuracy']:.3f}"
            f" rep4={eval_points['repetition_4gram']:.3f}"
        )
    fig.update_layout(title=title, template="plotly_dark", height=700)
    live.show(fig, step)


def render_roofline_live(
    results: List[Dict],
    roofline_points: List[Dict],
    ridge: float,
    peak_tflops: float,
    peak_bw_gb_s: float,
    output_dir: str,
    step: int,
    live: LiveNotebookDisplay,
) -> None:
    if not live.should_update(step):
        return
    ai_range = np.logspace(-2, 4, 300)
    roofline = np.minimum(peak_tflops, ai_range * peak_bw_gb_s / 1000)
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Roofline", "Latency by sequence"))
    fig.add_trace(go.Scatter(x=ai_range, y=roofline, mode="lines", name="roofline", line=dict(color="#2ed573")), row=1, col=1)
    fig.add_vline(x=ridge, line_dash="dash", line_color="#999999", row=1, col=1)
    for point in roofline_points:
        ai = point.get("arithmetic_intensity", 0)
        tf = point.get("achieved_tflops", 0)
        if ai > 0 and tf > 0:
            color = "#56ccf2" if point.get("type") == "sparse" else "#ff7675"
            fig.add_trace(go.Scatter(x=[ai], y=[tf], mode="markers+text", text=[point.get("name", "")], textposition="top center", marker=dict(color=color, size=10), showlegend=False), row=1, col=1)
    dense = [r for r in results if r["type"] == "dense"]
    sparse = [r for r in results if r["type"] == "sparse"]
    if dense and sparse:
        xs = [r["seq_len"] for r in dense]
        fig.add_trace(go.Scatter(x=xs, y=[r["latency_mean_ms"] for r in dense], mode="lines+markers", name="dense", line=dict(color="#ff7675")), row=1, col=2)
        fig.add_trace(go.Scatter(x=xs, y=[r["latency_mean_ms"] for r in sparse], mode="lines+markers", name="sparse", line=dict(color="#56ccf2")), row=1, col=2)
    fig.update_xaxes(type="log", row=1, col=1)
    fig.update_yaxes(type="log", row=1, col=1)
    fig.update_layout(title="Roofline live dashboard", template="plotly_dark", height=520)
    live.show(fig, step)


def render_mega_stress_live(
    metrics: List[Dict],
    latest_summary: Dict[str, Dict] | None,
    output_dir: str,
    step: int,
    live: LiveNotebookDisplay,
) -> None:
    if not live.should_update(step) or not metrics:
        return
    xs = list(range(len(metrics)))
    mem = [row.get("host_memory_used_gb", 0.0) for row in metrics]
    lat = [row.get("latency_ms", row.get("attention_under_pressure_ms", 0.0)) for row in metrics]
    tok = [row.get("tok_s", row.get("tokens_per_sec", 0.0)) for row in metrics]
    cpu = [row.get("cpu_percent", 0.0) for row in metrics]
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        subplot_titles=("Memory and latency", "Throughput and CPU", "Latest stage summary"),
        specs=[[{}], [{}], [{"type": "table"}]],
    )
    fig.add_trace(go.Scatter(x=xs, y=mem, mode="lines+markers", name="host ram gb", line=dict(color="#56ccf2")), row=1, col=1)
    fig.add_trace(go.Scatter(x=xs, y=lat, mode="lines", name="latency ms", line=dict(color="#ff9f43")), row=1, col=1)
    fig.add_trace(go.Scatter(x=xs, y=tok, mode="lines", name="tok/s", line=dict(color="#2ed573")), row=2, col=1)
    fig.add_trace(go.Scatter(x=xs, y=cpu, mode="lines", name="cpu %", line=dict(color="#a29bfe")), row=2, col=1)
    headers = ["stage", "status", "key metrics"]
    rows = []
    for stage, payload in (latest_summary or {}).items():
        if not isinstance(payload, dict):
            continue
        bits = []
        for key in ("achieved_gb", "attention_under_pressure_ms", "avg_ms", "p99_ms", "jitter_ms", "tok_s"):
            if key in payload:
                bits.append(f"{key}={payload[key]}")
        rows.append([stage, payload.get("status", ""), ", ".join(bits)])
    if not rows:
        rows = [["live", "running", "collecting metrics"]]
    fig.add_trace(
        go.Table(
            header=dict(values=headers, fill_color="#1f2a44", font=dict(color="white")),
            cells=dict(values=list(zip(*rows)), fill_color="#0d1526", font=dict(color="white")),
        ),
        row=3,
        col=1,
    )
    fig.update_layout(title="Mega stress live dashboard", template="plotly_dark", height=960)
    live.show(fig, step)

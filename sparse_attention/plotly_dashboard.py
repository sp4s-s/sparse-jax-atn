from __future__ import annotations

import json
import os
from typing import Dict, List, Any

try:
    import plotly.graph_objects as go
    import plotly.express as px
    import plotly.io as pio
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


def ensure_dir(path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)


def save_plotly_figure(fig, output_path: str, save_html: bool = True):
    if not PLOTLY_AVAILABLE:
        return
        
    ensure_dir(output_path)
    
    try:
        fig.write_image(output_path, scale=2)
        print(f"  📊 Saved Plotly static: {output_path}")
    except Exception as e:
        print(f"  [WARN] Kaleido failed to save static PNG ({output_path}): {e}")
        
    if save_html:
        html_path = output_path.replace(".png", ".html")
        fig.write_html(html_path)
        print(f"  🌐 Saved interactive HTML: {html_path}")


def generate_plotly_dashboard(
    results_json_path: str = "benchmark_results/benchmark_results.json",
    stress_json_path: str = "benchmark_results/stress_test_all.json",
    output_dir: str = "benchmark_results/plotly_plots",
):
    if not PLOTLY_AVAILABLE:
        print("[WARN] Plotly not installed. Skipping interactive dashboards.")
        return
        
    print(f"\n{'='*70}")
    print("GENERATING PLOTLY DASHBOARDS & STATIC EXPORTS")
    print(f"{'='*70}")
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        with open(results_json_path) as f:
            summary = json.load(f)
        comparisons = summary.get("comparisons", [])
    except Exception as e:
        print(f"Could not load benchmark results: {e}")
        comparisons = []
        
    if comparisons:
        labels = [f"B={c.get('B')}<br>N={c.get('N')}" for c in comparisons]
        dense_lats = [float(c.get("Dense Latency (ms)", 0)) for c in comparisons]
        sparse_lats = [float(c.get("Sparse Latency (ms)", 0)) for c in comparisons]
        hbm_reductions = [float(c.get("HBM Reduction", "0").replace("%", "")) for c in comparisons]
        speedups = [float(c.get("Speedup", "1").replace("x", "")) for c in comparisons]
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(name="Dense Latency (ms)", x=labels, y=dense_lats, marker_color="#f85149"), secondary_y=False)
        fig.add_trace(go.Bar(name="Sparse Latency (ms)", x=labels, y=sparse_lats, marker_color="#58a6ff"), secondary_y=False)
        fig.add_trace(go.Scatter(name="Speedup (x)", x=labels, y=speedups, mode='lines+markers', 
                                 marker_color="#3fb950", line=dict(width=3)), secondary_y=True)
        
        fig.update_layout(title="Latency & Speedup Analysis", template="plotly_dark", barmode='group')
        fig.update_yaxes(title_text="Latency (ms)", secondary_y=False)
        fig.update_yaxes(title_text="Speedup Multiplier", secondary_y=True)
        save_plotly_figure(fig, os.path.join(output_dir, "latency_speedup_plotly.png"))
        
        fig2 = go.Figure(data=[
            go.Bar(name="HBM Reduction %", x=labels, y=hbm_reductions, marker_color="#bc8cff")
        ])
        fig2.add_hline(y=40, line_dash="dash", line_color="#f85149", annotation_text="Target (40%)")
        fig2.update_layout(title="HBM Bandwidth Reduction %", template="plotly_dark")
        fig2.update_yaxes(title_text="Reduction (%)")
        save_plotly_figure(fig2, os.path.join(output_dir, "hbm_reduction_plotly.png"))
        
    try:
        with open(stress_json_path) as f:
            stress = json.load(f)
    except Exception:
        stress = None
        
    if stress and "sustained" in stress:
        timeseries = stress["sustained"].get("time_series", {})
        ts_lat = timeseries.get("latencies_ms", [])
        ts_tok = timeseries.get("throughputs", [])
        
        if ts_lat and ts_tok:
            fig3 = make_subplots(specs=[[{"secondary_y": True}]])
            x_steps = list(range(len(ts_lat)))
            
            fig3.add_trace(go.Scatter(x=x_steps, y=ts_tok, mode="lines", name="Tokens/sec", line=dict(color="#39d2c0")), secondary_y=False)
            fig3.add_trace(go.Scatter(x=x_steps, y=ts_lat, mode="lines", name="Latency (ms)", line=dict(color="#f0883e", width=1), opacity=0.7), secondary_y=True)
            
            fig3.update_layout(title="Sustained Throughput Stability & Jitter", template="plotly_dark")
            fig3.update_yaxes(title_text="Tokens / Sec", secondary_y=False)
            fig3.update_yaxes(title_text="Latency (ms)", secondary_y=True)
            save_plotly_figure(fig3, os.path.join(output_dir, "sustained_throughput_plotly.png"))

    if stress and "oom_sparse" in stress:
        oom_s = stress["oom_sparse"]
        z_status = oom_s.get("status_matrix", [])
        y_seq = oom_s.get("seq_lengths", [])
        x_bat = oom_s.get("batch_sizes", [])
        
        if z_status:
            colorscale = [[0, '#d29922'], [1, '#3fb950']]
            
            fig4 = go.Figure(data=go.Heatmap(
                z=z_status,
                x=[f"B={b}" for b in x_bat],
                y=[f"N={n}" for n in y_seq],
                colorscale=colorscale,
                showscale=False,
                text=[[("✅ Pass" if v==1 else "❌ OOM") for v in row] for row in z_status],
                texttemplate="%{text}",
            ))
            fig4.update_layout(title="OOM Boundary Pass/Fail Matrix", template="plotly_dark")
            save_plotly_figure(fig4, os.path.join(output_dir, "oom_heatmap_plotly.png"))

    print(f"✅ Plotly outputs saved to {output_dir}")

if __name__ == "__main__":
    generate_plotly_dashboard()

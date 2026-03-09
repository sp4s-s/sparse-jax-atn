from __future__ import annotations

import json
import math
import os
import statistics
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

try:
    import psutil
except ImportError:
    psutil = None

try:
    from IPython.display import HTML, display, clear_output
    from IPython import get_ipython
except ImportError:
    HTML = None
    display = None
    clear_output = None
    get_ipython = None

from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.proto.summary_pb2 import Summary
from tensorboard.summary.writer.event_file_writer import EventFileWriter
from sparse_attention.live_viz import LiveNotebookDisplay, render_mega_stress_live


def _now() -> float:
    return time.time()


def _safe_mean(values: List[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _safe_p(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    values_sorted = sorted(values)
    idx = min(len(values_sorted) - 1, max(0, int(round((len(values_sorted) - 1) * p))))
    return float(values_sorted[idx])


def format_bytes(num_bytes: float) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    for unit in units:
        if abs(value) < 1024.0 or unit == units[-1]:
            return f"{value:.1f}{unit}"
        value /= 1024.0
    return f"{value:.1f}TB"


def format_seconds(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(seconds, 60)
    return f"{int(minutes)}m {secs:.0f}s"


def runtime_snapshot() -> Dict[str, float]:
    snapshot = {}
    if psutil is None:
        return snapshot
    vm = psutil.virtual_memory()
    snapshot["host_memory_used_gb"] = vm.used / (1024 ** 3)
    snapshot["host_memory_available_gb"] = vm.available / (1024 ** 3)
    snapshot["host_memory_percent"] = vm.percent
    cpu_pct = psutil.cpu_percent(interval=None)
    snapshot["cpu_percent"] = cpu_pct
    try:
        load1, _, _ = os.getloadavg()
        snapshot["load_1m"] = load1
    except (AttributeError, OSError):
        pass
    return snapshot


def _is_colab_runtime() -> bool:
    return os.environ.get("COLAB_RELEASE_TAG") is not None


@dataclass
class StageState:
    name: str
    total: Optional[float] = None
    unit: str = ""
    started_at: float = field(default_factory=_now)
    progress: float = 0.0
    status: str = "running"
    message: str = ""
    last_metric: Dict[str, float] = field(default_factory=dict)


class BenchmarkTelemetry:
    def __init__(
        self,
        run_name: str,
        output_dir: str,
        config: Optional[Dict] = None,
        tensorboard_subdir: str = "tensorboard",
    ) -> None:
        self.run_name = run_name
        self.output_dir = output_dir
        self.config = config or {}
        self.started_at = _now()
        self.metrics: List[Dict] = []
        self.stage_order: List[str] = []
        self.stages: Dict[str, StageState] = {}
        self.latest_summary: Dict[str, Dict] = {}
        self.artifacts_dir = os.path.join(output_dir, "live")
        self.tensorboard_dir = os.path.join(output_dir, tensorboard_subdir, run_name)
        os.makedirs(self.artifacts_dir, exist_ok=True)
        os.makedirs(self.tensorboard_dir, exist_ok=True)
        self.metrics_path = os.path.join(self.artifacts_dir, f"{run_name}_metrics.jsonl")
        self.summary_path = os.path.join(self.artifacts_dir, f"{run_name}_summary.json")
        self.links_path = os.path.join(self.artifacts_dir, f"{run_name}_links.md")
        self._event_writer = EventFileWriter(self.tensorboard_dir)
        self._display_enabled = self._detect_notebook_display()
        self._last_flush_at = 0.0
        self._live_plot = LiveNotebookDisplay(run_name, self.artifacts_dir, f"{run_name}_live_plot.html", min_interval_steps=8)
        self._write_links()

    def start_stage(self, name: str, total: Optional[float] = None, unit: str = "", message: str = "") -> None:
        if name not in self.stages:
            self.stage_order.append(name)
        self.stages[name] = StageState(name=name, total=total, unit=unit, message=message)
        self._emit_console(name, message or "started")
        self.flush()

    def update_stage(
        self,
        name: str,
        progress: Optional[float] = None,
        message: Optional[str] = None,
        metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        stage = self.stages[name]
        if progress is not None:
            stage.progress = progress
        if message is not None:
            stage.message = message
        metric_payload = metrics or {}
        stage.last_metric.update(metric_payload)
        step = len(self.metrics)
        payload = {
            "timestamp": _now(),
            "stage": name,
            "progress": stage.progress,
            "message": stage.message,
        }
        payload.update(runtime_snapshot())
        payload.update(metric_payload)
        self.metrics.append(payload)
        with open(self.metrics_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")
        self._write_tb(step, payload)
        render_mega_stress_live(self.metrics, self.latest_summary, self.artifacts_dir, step, self._live_plot)
        self._emit_console(name, stage.message, stage.progress, stage.total, stage.unit, metric_payload)
        self.flush()

    def end_stage(self, name: str, status: str = "done", message: str = "") -> None:
        stage = self.stages[name]
        stage.status = status
        if message:
            stage.message = message
        self._emit_console(name, stage.message or status, stage.progress, stage.total, stage.unit)
        self.flush()

    def record_summary(self, key: str, value: Dict) -> None:
        self.latest_summary[key] = value
        self._last_flush_at = 0.0
        self.flush()

    def close(self) -> None:
        self._last_flush_at = 0.0
        self.flush()
        self._event_writer.close()

    def flush(self) -> None:
        now = _now()
        if now - self._last_flush_at < 0.75:
            return
        self._last_flush_at = now
        summary = {
            "run_name": self.run_name,
            "started_at": self.started_at,
            "elapsed_s": _now() - self.started_at,
            "config": self.config,
            "stages": {
                name: {
                    "status": stage.status,
                    "total": stage.total,
                    "progress": stage.progress,
                    "unit": stage.unit,
                    "message": stage.message,
                    "elapsed_s": _now() - stage.started_at,
                    "last_metric": stage.last_metric,
                }
                for name, stage in self.stages.items()
            },
            "latest_summary": self.latest_summary,
            "metrics_path": self.metrics_path,
            "tensorboard_dir": self.tensorboard_dir,
        }
        with open(self.summary_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        self._write_links()
        self._render_notebook(summary)

    def _write_tb(self, step: int, payload: Dict[str, float]) -> None:
        values = []
        for key, value in payload.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
                values.append(Summary.Value(tag=key, simple_value=float(value)))
        if not values:
            return
        event = Event(wall_time=_now(), step=step, summary=Summary(value=values))
        self._event_writer.add_event(event)
        self._event_writer.flush()

    def _emit_console(
        self,
        stage: str,
        message: str,
        progress: Optional[float] = None,
        total: Optional[float] = None,
        unit: str = "",
        metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        prefix = f"[{stage}]"
        progress_txt = ""
        if progress is not None and total:
            pct = min(100.0, max(0.0, (progress / total) * 100.0))
            progress_txt = f" {progress:.1f}/{total:.1f}{unit} ({pct:.0f}%)"
        metrics_txt = ""
        if metrics:
            parts = []
            for key, value in metrics.items():
                if not isinstance(value, (int, float)):
                    continue
                if "bytes" in key:
                    parts.append(f"{key}={format_bytes(value)}")
                elif key.endswith("_gb"):
                    parts.append(f"{key}={value:.2f}GB")
                elif key.endswith("_ms"):
                    parts.append(f"{key}={value:.1f}ms")
                elif "percent" in key or key.endswith("_pct"):
                    parts.append(f"{key}={value:.1f}%")
                else:
                    parts.append(f"{key}={value:.2f}")
            if parts:
                metrics_txt = " | " + ", ".join(parts[:5])
        print(f"{prefix} {message}{progress_txt}{metrics_txt}", flush=True)

    def _render_notebook(self, summary: Dict) -> None:
        if not self._display_enabled:
            return
        html = self._build_notebook_html(summary)
        clear_output(wait=True)
        display(HTML(html))

    def _detect_notebook_display(self) -> bool:
        if HTML is None or display is None or clear_output is None or get_ipython is None:
            return False
        try:
            shell = get_ipython()
        except Exception:
            return False
        if shell is None:
            return False
        shell_name = shell.__class__.__name__
        if shell_name not in {"ZMQInteractiveShell", "Shell"}:
            return False
        if os.environ.get("COLAB_RELEASE_TAG") is None and os.environ.get("JPY_PARENT_PID") is None:
            return False
        return os.environ.get("SPARSE_ATTENTION_ENABLE_INLINE", "0") == "1"

    def _write_links(self) -> None:
        tensorboard_url = "http://localhost:6006/"
        lines = [
            f"# Live Run Links: {self.run_name}",
            "",
            f"- TensorBoard logdir: `{self.tensorboard_dir}`",
            f"- Metrics stream: `{self.metrics_path}`",
            f"- Run summary: `{self.summary_path}`",
            "",
            "## TensorBoard",
            "",
            "Start TensorBoard in a notebook cell:",
            "```python",
            "%load_ext tensorboard",
            f"%tensorboard --logdir {self.tensorboard_dir}",
            "```",
            "",
            "Or from a shell:",
            "```bash",
            f"tensorboard --logdir '{self.tensorboard_dir}' --port 6006",
            "```",
            "",
            f"Default local URL after launch: {tensorboard_url}",
        ]
        if _is_colab_runtime():
            lines.extend(
                [
                    "",
                    "## Colab note",
                    "",
                    "If this benchmark was started with `!python ...`, use a new notebook cell for TensorBoard.",
                    "Colab does not reliably surface a clickable live view from the subprocess itself.",
                ]
            )
        with open(self.links_path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")

    def _build_notebook_html(self, summary: Dict) -> str:
        cards = []
        for name in self.stage_order:
            stage = summary["stages"][name]
            total = stage["total"] or 0
            progress = stage["progress"] or 0
            pct = 0 if total == 0 else min(100.0, max(0.0, progress / total * 100.0))
            metrics = " · ".join(
                f"{k}: {v:.2f}" if isinstance(v, float) else f"{k}: {v}"
                for k, v in list(stage["last_metric"].items())[:6]
            )
            cards.append(
                f"""
                <div class="card">
                  <div class="card-head"><span>{name}</span><span>{stage['status']}</span></div>
                  <div class="bar"><span style="width:{pct:.1f}%"></span></div>
                  <div class="meta">{stage['message']}</div>
                  <div class="meta">{progress:.1f}/{total:.1f}{stage['unit']} · {format_seconds(stage['elapsed_s'])}</div>
                  <div class="metrics">{metrics}</div>
                </div>
                """
            )
        summary_rows = []
        for key, value in self.latest_summary.items():
            summary_rows.append(f"<tr><td>{key}</td><td><pre>{json.dumps(value, indent=2)}</pre></td></tr>")
        return f"""
        <html>
        <head>
          <style>
            body {{ font-family: 'Google Sans', 'Segoe UI', sans-serif; background: linear-gradient(135deg, #07111f, #132238 60%, #1b3b33); color: #f7fbff; padding: 20px; }}
            .hero {{ display:flex; justify-content:space-between; align-items:end; gap:20px; margin-bottom:18px; }}
            .title {{ font-size:28px; font-weight:700; }}
            .sub {{ color:#8fd3ff; }}
            .grid {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap:14px; }}
            .card {{ background: rgba(255,255,255,0.08); border:1px solid rgba(255,255,255,0.14); border-radius:18px; padding:16px; backdrop-filter: blur(10px); }}
            .card-head {{ display:flex; justify-content:space-between; font-weight:600; margin-bottom:10px; }}
            .bar {{ height:10px; background: rgba(255,255,255,0.12); border-radius:999px; overflow:hidden; margin-bottom:10px; }}
            .bar span {{ display:block; height:100%; background: linear-gradient(90deg, #56ccf2, #6fcf97, #f2c94c); border-radius:999px; }}
            .meta {{ color:#d7efff; margin-bottom:6px; }}
            .metrics {{ color:#b4d7f3; font-size:12px; line-height:1.5; }}
            table {{ width:100%; margin-top:20px; border-collapse:collapse; }}
            td {{ border-top:1px solid rgba(255,255,255,0.14); padding:10px 8px; vertical-align:top; }}
            pre {{ margin:0; white-space:pre-wrap; color:#dff7ff; }}
          </style>
        </head>
        <body>
          <div class="hero">
            <div>
              <div class="title">{self.run_name.replace('_', ' ').title()}</div>
              <div class="sub">elapsed {format_seconds(summary['elapsed_s'])} · tensorboard {self.tensorboard_dir}</div>
            </div>
            <div class="sub">{json.dumps(self.config)}</div>
          </div>
          <div class="grid">{''.join(cards)}</div>
          <table>{''.join(summary_rows)}</table>
        </body>
        </html>
        """

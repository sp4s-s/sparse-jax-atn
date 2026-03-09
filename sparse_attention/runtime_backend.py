from __future__ import annotations

import re
import os
from typing import Tuple

import jax


TPU_BUSY_PATTERN = re.compile(r"TPU is already in use by process with pid (\d+)")


def detect_tpu_status() -> Tuple[bool, str]:
    try:
        devices = jax.devices("tpu")
        if devices:
            return True, ""
        return False, "No TPU devices were reported by JAX."
    except Exception as exc:
        message = str(exc)
        match = TPU_BUSY_PATTERN.search(message)
        if match:
            return False, f"TPU is already in use by process pid {match.group(1)}."
        return False, message


def resolve_training_backend(prefer_tpu: bool = True) -> str:
    if not prefer_tpu:
        jax.config.update("jax_platform_name", "cpu")
        return "cpu"
    ok, _ = detect_tpu_status()
    if ok:
        return "tpu"
    raise RuntimeError(build_tpu_busy_message("Training"))


def require_tpu(task_name: str) -> None:
    ok, message = detect_tpu_status()
    if ok:
        return
    raise RuntimeError(build_tpu_busy_message(task_name, message))


def build_tpu_busy_message(task_name: str, detail: str | None = None) -> str:
    base_detail = detail
    if base_detail is None:
        _, base_detail = detect_tpu_status()
    lines = [
        f"{task_name} requires exclusive TPU access.",
        f"TPU init failed: {base_detail}",
        "This project is TPU-only, so the command will not fall back to CPU.",
        "Most common cause in Colab: the notebook kernel or another `!python` process already initialized JAX TPU runtime.",
        "Fix options:",
        "1. Stop the other TPU process if you started one.",
        "2. Restart the Colab runtime to release libtpu ownership.",
        "3. Avoid launching multiple TPU jobs at the same time.",
        "4. Prefer running one TPU command at a time from a fresh runtime.",
    ]
    if os.environ.get("COLAB_RELEASE_TAG") is not None:
        lines.append("Colab note: if the notebook process already touched `jax.devices()` or ran another TPU cell, a later `!python ...` command may fail because it is a separate process.")
    return "\n".join(lines)

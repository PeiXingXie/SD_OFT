"""
Factories for building model clients from config.
"""

from __future__ import annotations

from typing import Any

from .client import MLLMClient


def build_mllm_client(api_cfg: dict[str, Any]) -> MLLMClient:
    api_cfg = dict(api_cfg or {})
    use_reasoning = api_cfg.get("useReasoning", None)
    temperature = api_cfg.get("temperature", None)
    top_p = api_cfg.get("topP", None)
    return MLLMClient(
        base_url=str(api_cfg.get("base_url", "")),
        app_id=str(api_cfg.get("app_id", "")),
        secret_key=str(api_cfg.get("secret_key", "")),
        model=str(api_cfg.get("model", "")),
        timeout_sec=int(api_cfg.get("timeout_sec", 120)),
        retry_on_code=int(api_cfg.get("retry_on_code", -20001)),
        retry_sleep_sec=int(api_cfg.get("retry_sleep_sec", 60)),
        use_reasoning=(None if use_reasoning is None else bool(use_reasoning)),
        temperature=(None if temperature is None else float(temperature)),
        top_p=(None if top_p is None else float(top_p)),
    )


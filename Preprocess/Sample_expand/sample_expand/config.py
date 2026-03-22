"""
Config loader for Sample_expand external image generation API.

- Supports ${ENV:VAR} placeholders in YAML recursively.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

_ENV_PATTERN = re.compile(r"^\$\{ENV:([A-Za-z_][A-Za-z0-9_]*)\}$")


def resolve_env_placeholders(x: Any) -> Any:
    if isinstance(x, dict):
        return {k: resolve_env_placeholders(v) for k, v in x.items()}
    if isinstance(x, list):
        return [resolve_env_placeholders(v) for v in x]
    if isinstance(x, str):
        m = _ENV_PATTERN.match(x.strip())
        if m:
            return os.environ.get(m.group(1), "")
        return x
    return x


@dataclass(frozen=True)
class AndesAuthConfig:
    app_id: str
    secret_key: str


@dataclass(frozen=True)
class RetryConfig:
    timeout_sec: int = 300
    max_retries: int = 3
    retry_delay_sec: float = 3.0
    rate_limit_delay_sec: float = 10.0


@dataclass(frozen=True)
class RequestDefaults:
    model: str
    size: str = "1024x1024"
    num: int = 1
    quality: str | None = None  # gpt-image-1 requires "high"


@dataclass(frozen=True)
class ImageGenConfig:
    provider: Literal["andes_gateway"] = "andes_gateway"
    # Prefer YAML `base_url` or env `ANDES_BASE_URL`. Keep empty by default to avoid leaking
    # internal endpoints when this repo is migrated/shared.
    base_url: str = ""
    api_url: str | None = None  # if set, takes precedence over base_url
    endpoint_path: str = "/image/v1/generations"
    auth: AndesAuthConfig | None = None
    request_defaults: RequestDefaults | None = None
    retry: RetryConfig = RetryConfig()

    def resolved_api_url(self) -> str:
        if isinstance(self.api_url, str) and self.api_url.strip():
            url = self.api_url.strip()
        else:
            if not str(self.base_url or "").strip():
                raise ValueError(
                    "Missing base_url/api_url. Set `base_url` in YAML or export ANDES_BASE_URL."
                )
            url = self.base_url.rstrip("/") + "/" + self.endpoint_path.lstrip("/")
        # enforce required suffix
        if not url.endswith(self.endpoint_path):
            # if user supplied base_url or wrong path, normalize to required suffix
            base = url.split("/image/v1/")[0].rstrip("/")
            url = base + self.endpoint_path
        return url


def load_image_gen_config(config_path: str) -> ImageGenConfig:
    p = Path(config_path).expanduser().resolve()
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    raw = resolve_env_placeholders(raw)

    provider = raw.get("provider", "andes_gateway")
    base_url = raw.get("base_url") or os.environ.get("ANDES_BASE_URL") or ""
    api_url = raw.get("api_url")
    endpoint_path = raw.get("endpoint_path", "/image/v1/generations")

    auth_raw = raw.get("auth") or {}
    auth = None
    if auth_raw:
        auth = AndesAuthConfig(
            app_id=str(auth_raw.get("app_id") or "").strip(),
            secret_key=str(auth_raw.get("secret_key") or "").strip(),
        )

    retry_raw = raw.get("retry") or {}
    retry = RetryConfig(
        timeout_sec=int(retry_raw.get("timeout_sec", 300)),
        max_retries=int(retry_raw.get("max_retries", 3)),
        retry_delay_sec=float(retry_raw.get("retry_delay_sec", 3.0)),
        rate_limit_delay_sec=float(retry_raw.get("rate_limit_delay_sec", 10.0)),
    )

    req_raw = raw.get("request_defaults") or raw.get("request") or {}
    request_defaults = None
    if req_raw:
        request_defaults = RequestDefaults(
            model=str(req_raw.get("model") or "").strip(),
            size=str(req_raw.get("size") or "1024x1024").strip(),
            num=int(req_raw.get("num", 1)),
            quality=(str(req_raw.get("quality")).strip() if req_raw.get("quality") is not None else None),
        )

    return ImageGenConfig(
        provider=provider,
        base_url=str(base_url),
        api_url=(str(api_url).strip() if isinstance(api_url, str) else None),
        endpoint_path=str(endpoint_path),
        auth=auth,
        request_defaults=request_defaults,
        retry=retry,
    )


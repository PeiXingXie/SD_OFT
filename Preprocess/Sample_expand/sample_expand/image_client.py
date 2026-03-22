from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable

import requests

from .andes_auth import sign
from .config import ImageGenConfig
from .utils_images import download_image_as_data_url


def _resp_to_raw_http(resp: requests.Response, *, max_text_chars: int = 200000) -> dict[str, Any]:
    try:
        text = resp.text
    except Exception:
        text = ""
    if text and max_text_chars and len(text) > max_text_chars:
        text = text[:max_text_chars] + "\n...[truncated]..."
    return {
        "status_code": resp.status_code,
        "url": getattr(resp, "url", None),
        "headers": dict(resp.headers or {}),
        "text": text,
    }


@dataclass(frozen=True)
class ImageGenerationResult:
    ok: bool
    image_url: str | None = None
    image_base64: str | None = None  # data:image/png;base64,...
    error: str | None = None
    response: dict[str, Any] | None = None  # parsed JSON
    raw_http: dict[str, Any] | None = None  # raw http summary
    attempts: int = 1
    elapsed_sec: float = 0.0  # total wall time of this call (including retries/sleeps/download)


class AndesGatewayImageClient:
    """
    External text-to-image (and optionally image-to-image) client through Andes gateway.

    Required constraints:
    - endpoint suffix must be /image/v1/generations
    - size must be 1024x1024
    - gpt-image-1 must set quality=high
    """

    REQUIRED_ENDPOINT_PATH = "/image/v1/generations"
    REQUIRED_SIZE = "1024x1024"

    def __init__(self, cfg: ImageGenConfig):
        self.cfg = cfg
        if not cfg.auth or not cfg.auth.app_id or not cfg.auth.secret_key:
            raise RuntimeError("Missing auth.app_id or auth.secret_key in config (or env placeholders resolved to empty)")

    def generate(
        self,
        prompt: str,
        *,
        model: str | None = None,
        images: list[str] | None = None,
        status_cb: Callable[[str], None] | None = None,
    ) -> ImageGenerationResult:
        t0 = time.perf_counter()

        def done(**kwargs: Any) -> ImageGenerationResult:
            elapsed = time.perf_counter() - t0
            if "elapsed_sec" not in kwargs:
                kwargs["elapsed_sec"] = float(elapsed)
            return ImageGenerationResult(**kwargs)  # type: ignore[arg-type]

        def emit(msg: str) -> None:
            if status_cb is None:
                return
            try:
                status_cb(str(msg))
            except Exception:
                # never let logging break the main flow
                return

        model = (model or (self.cfg.request_defaults.model if self.cfg.request_defaults else "")).strip()
        if not model:
            return done(ok=False, error="missing model", attempts=1)

        api_url = self.cfg.resolved_api_url()
        if not api_url.endswith(self.REQUIRED_ENDPOINT_PATH):
            return done(
                ok=False,
                error=f"api_url must end with {self.REQUIRED_ENDPOINT_PATH}: {api_url}",
                attempts=1,
            )

        # enforce size
        size = self.REQUIRED_SIZE

        # enforce quality for gpt-image-1
        quality = None
        if model == "gpt-image-1":
            quality = "high"
        elif self.cfg.request_defaults and self.cfg.request_defaults.quality:
            quality = str(self.cfg.request_defaults.quality).strip()

        max_retries = int(self.cfg.retry.max_retries)
        timeout = int(self.cfg.retry.timeout_sec)
        retry_delay = float(self.cfg.retry.retry_delay_sec)
        rate_limit_delay = float(self.cfg.retry.rate_limit_delay_sec)

        last_error: str | None = None
        for attempt in range(max_retries + 1):
            if attempt > 0:
                delay = retry_delay * attempt
                if last_error and ("Rate limit" in last_error or "rate limit" in last_error or "code=-20001" in last_error):
                    delay = rate_limit_delay * attempt
                emit(f"[api] retry {attempt}/{max_retries}: sleep {delay:.1f}s, last_error={str(last_error)[:200]}")
                time.sleep(delay)

            req_id = str(uuid.uuid4())
            try:
                attempt_t0 = time.perf_counter()
                emit(
                    f"[api] attempt {attempt + 1}/{max_retries + 1}: model={model}, size={size}"
                    + (f", quality={quality}" if quality else "")
                )
                body: dict[str, Any] = {
                    "model": model,
                    "caption": str(prompt or ""),
                    "num": 1,
                    "size": size,
                }
                if quality:
                    body["quality"] = quality
                if images:
                    body["images"] = images

                data_str = json.dumps(body, ensure_ascii=False)
                headers = {
                    "recordId": req_id,
                    "Authorization": sign(None, data_str, self.cfg.auth.app_id, self.cfg.auth.secret_key),
                    "Content-Type": "application/json; charset=utf-8",
                }
                resp = requests.request(
                    "POST",
                    url=api_url,
                    headers=headers,
                    data=data_str.encode("utf-8"),
                    timeout=timeout,
                )
                raw_http = _resp_to_raw_http(resp)

                if resp.status_code != 200:
                    last_error = f"HTTP {resp.status_code}: {raw_http.get('text','')[:200]}"
                    emit(f"[api] http_error: {last_error}")
                    if attempt < max_retries:
                        continue
                    return done(ok=False, error=last_error, raw_http=raw_http, attempts=attempt + 1)

                try:
                    result = resp.json()
                except Exception as e:
                    last_error = f"bad_json: {e}: {raw_http.get('text','')[:200]}"
                    emit(f"[api] bad_json: {str(last_error)[:200]}")
                    if attempt < max_retries:
                        continue
                    return done(ok=False, error=last_error, raw_http=raw_http, attempts=attempt + 1)

                if not isinstance(result, dict):
                    last_error = f"bad_response_type: {type(result).__name__}"
                    emit(f"[api] bad_response_type: {last_error}")
                    if attempt < max_retries:
                        continue
                    return done(ok=False, error=last_error, raw_http=raw_http, attempts=attempt + 1)

                code = result.get("code")
                if code not in (0, None):
                    msg = result.get("msg") or result.get("message") or ""
                    last_error = f"api_code={code}: {msg}"
                    emit(f"[api] api_error: {str(last_error)[:200]}")
                    if int(code) in (-20001, -30002, -30001) and attempt < max_retries:
                        continue
                    return done(ok=False, error=last_error, response=result, raw_http=raw_http, attempts=attempt + 1)

                data_obj = result.get("data", {})
                results = data_obj.get("results", []) if isinstance(data_obj, dict) else []
                if not results or not isinstance(results[0], dict):
                    last_error = "missing image results in response"
                    emit(f"[api] schema_error: {last_error}")
                    if attempt < max_retries:
                        continue
                    return done(ok=False, error=last_error, response=result, raw_http=raw_http, attempts=attempt + 1)

                image_url = results[0].get("url") or results[0].get("contentUrl")
                if not image_url:
                    last_error = "missing image url in results[0]"
                    emit(f"[api] schema_error: {last_error}")
                    if attempt < max_retries:
                        continue
                    return done(ok=False, error=last_error, response=result, raw_http=raw_http, attempts=attempt + 1)

                try:
                    image_base64 = download_image_as_data_url(image_url, timeout_sec=timeout)
                except Exception as e:
                    last_error = f"download_failed: {e}"
                    emit(f"[api] download_failed: {str(last_error)[:200]}")
                    if attempt < max_retries:
                        continue
                    return done(
                        ok=False,
                        error=last_error,
                        image_url=image_url,
                        response=result,
                        raw_http=raw_http,
                        attempts=attempt + 1,
                    )

                attempt_elapsed = time.perf_counter() - attempt_t0
                emit(f"[api] success: attempt_elapsed={attempt_elapsed:.3f}s")
                return done(
                    ok=True,
                    image_url=image_url,
                    image_base64=image_base64,
                    response=result,
                    raw_http=raw_http,
                    attempts=attempt + 1,
                )

            except requests.Timeout:
                last_error = f"timeout({timeout}s)"
                emit(f"[api] timeout: {last_error}")
                if attempt < max_retries:
                    continue
                return done(ok=False, error=last_error, attempts=attempt + 1)
            except requests.RequestException as e:
                last_error = f"request_exception: {e}"
                emit(f"[api] request_exception: {str(last_error)[:200]}")
                if attempt < max_retries:
                    continue
                return done(ok=False, error=last_error, attempts=attempt + 1)
            except Exception as e:
                last_error = f"unexpected: {e}"
                emit(f"[api] unexpected: {str(last_error)[:200]}")
                if attempt < max_retries:
                    continue
                return done(ok=False, error=last_error, attempts=attempt + 1)

        return done(ok=False, error=f"exhausted retries: {last_error}", attempts=max_retries + 1)


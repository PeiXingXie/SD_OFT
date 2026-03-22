"""
Andes Gateway style MLLM client (HMAC-signed chat completions).

Copied/adapted from Project/ReasonEdit/Judger/V5 for MLLM_Extract usage.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import requests


def _sign(params: dict | None, body: str, app_id: str, secret_key: str) -> str:
    """
    Andes signature:
    - auth_string_prefix: bot-auth-v1/{appId}/{timestamp_ms}/
    - signature: hmac_sha256(secret_key, auth_string_prefix + sorted_query + body)
    - return auth_string_prefix + signature
    """

    auth_string_prefix = f"bot-auth-v1/{app_id}/{int(time.time() * 1000)}/"
    sb = [auth_string_prefix]
    if params:
        ordered_params = OrderedDict(sorted(params.items()))
        sb.extend([f"{k}={v}&" for k, v in ordered_params.items()])
    sign_str = "".join(sb) + body
    signature = hmac.new(
        secret_key.encode("utf-8"),
        sign_str.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return auth_string_prefix + signature


@dataclass
class AndesAPIError(RuntimeError):
    error_type: str
    message: str
    record_id: str | None = None
    http_status: int | None = None
    api_code: int | None = None
    api_msg: str | None = None
    response_text: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "error_type": self.error_type,
            "message": self.message,
            "record_id": self.record_id,
            "http_status": self.http_status,
            "api_code": self.api_code,
            "api_msg": self.api_msg,
            "response_text": self.response_text,
        }

    def __str__(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


class MLLMClient:
    def __init__(
        self,
        *,
        base_url: str,
        app_id: str,
        secret_key: str,
        model: str,
        timeout_sec: int = 120,
        retry_on_code: int = -20001,
        retry_sleep_sec: int = 60,
        use_reasoning: bool | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
    ):
        self.base_url = str(base_url)
        self.model = str(model)
        self.timeout = int(timeout_sec)
        self.app_id = str(app_id)
        if not isinstance(secret_key, str) or not secret_key.strip():
            raise RuntimeError("Missing secret key (api.secret_key is empty)")
        self.secret_key = secret_key.strip()
        self.retry_on_code = int(retry_on_code)
        self.retry_sleep_sec = int(retry_sleep_sec)
        self.use_reasoning = use_reasoning
        self.temperature = temperature
        self.top_p = top_p

    def chat_completions(self, messages: list[dict]) -> str:
        """
        Returns assistant message content as string.
        Expected response:
          {"code": 0, "data": {"choices":[{"message":{"content":"..."}}]}}
        """

        req_id = str(uuid.uuid1())
        payload: dict[str, Any] = {"model": self.model, "messages": messages}
        if self.use_reasoning is not None:
            payload["useReasoning"] = bool(self.use_reasoning)
        if self.temperature is not None:
            payload["temperature"] = float(self.temperature)
        if self.top_p is not None:
            payload["topP"] = float(self.top_p)

        data_str = json.dumps(payload, ensure_ascii=False)
        data_bytes = data_str.encode("utf-8")
        headers = {
            "recordId": req_id,
            "Authorization": _sign(None, data_str, self.app_id, self.secret_key),
            "Content-Type": "application/json; charset=utf-8",
        }

        def _do_request(timeout: int) -> dict:
            try:
                resp = requests.request(
                    "POST",
                    url=self.base_url,
                    headers=headers,
                    data=data_bytes,
                    timeout=timeout,
                )
            except requests.Timeout as e:
                raise AndesAPIError(error_type="timeout", message=str(e), record_id=req_id) from e
            except requests.RequestException as e:
                raise AndesAPIError(error_type="request_exception", message=str(e), record_id=req_id) from e

            if resp.status_code < 200 or resp.status_code >= 300:
                txt = None
                try:
                    txt = resp.text
                except Exception:
                    txt = None
                raise AndesAPIError(
                    error_type="http_status",
                    message=f"HTTP {resp.status_code}",
                    record_id=req_id,
                    http_status=int(resp.status_code),
                    response_text=(txt[:2000] if isinstance(txt, str) else None),
                )
            try:
                data = resp.json()
            except Exception as e:
                txt = None
                try:
                    txt = resp.text
                except Exception:
                    txt = None
                raise AndesAPIError(
                    error_type="bad_json",
                    message=f"response json decode error: {e}",
                    record_id=req_id,
                    response_text=(txt[:2000] if isinstance(txt, str) else None),
                ) from e
            if not isinstance(data, dict):
                raise AndesAPIError(
                    error_type="bad_response_type",
                    message=f"Invalid response type: {type(data)}",
                    record_id=req_id,
                    response_text=str(data)[:2000],
                )
            return data

        data = _do_request(self.timeout)
        if isinstance(data, dict) and data.get("code") == self.retry_on_code:
            time.sleep(self.retry_sleep_sec)
            data = _do_request(min(30, self.timeout))

        if data.get("code") not in [0, None]:
            raise AndesAPIError(
                error_type="api_code",
                message="Andes gateway returned non-zero code",
                record_id=req_id,
                api_code=int(data.get("code")) if isinstance(data.get("code"), int) else None,
                api_msg=str(data.get("msg") or data.get("message") or data)[:2000],
                response_text=str(data)[:2000],
            )

        if "data" not in data:
            raise AndesAPIError(
                error_type="missing_data_field",
                message="Missing 'data' in response",
                record_id=req_id,
                response_text=str(data)[:2000],
            )
        try:
            return data["data"]["choices"][0]["message"]["content"]
        except Exception as e:
            raise AndesAPIError(
                error_type="bad_response_schema",
                message=f"Missing expected fields in response: {e}",
                record_id=req_id,
                response_text=str(data)[:2000],
            ) from e


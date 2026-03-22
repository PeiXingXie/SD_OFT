from __future__ import annotations

import hashlib
import hmac
import time
from collections import OrderedDict
from typing import Any


def sign(params: dict[str, Any] | None, body: str, app_id: str, secret_key: str) -> str:
    """
    Andes gateway style signature:
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


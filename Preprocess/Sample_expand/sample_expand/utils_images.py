from __future__ import annotations

import base64
from io import BytesIO

import requests
from PIL import Image


def bytes_to_png_data_url(img_bytes: bytes) -> str:
    img = Image.open(BytesIO(img_bytes))
    img.verify()
    img = Image.open(BytesIO(img_bytes))
    if img.mode != "RGB":
        img = img.convert("RGB")
    buf = BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def download_image_as_data_url(url: str, *, timeout_sec: int = 300) -> str:
    resp = requests.get(url, timeout=timeout_sec)
    resp.raise_for_status()
    if not resp.content:
        raise RuntimeError("downloaded image is empty")
    return bytes_to_png_data_url(resp.content)


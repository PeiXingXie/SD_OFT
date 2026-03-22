"""
Image loading/encoding helpers.

Reads local image files and returns `data:<mime>;base64,...` URLs for passing to
multimodal chat gateways.
"""

from __future__ import annotations

import base64
from pathlib import Path


def load_image_as_data_url(image_path: str) -> str:
    """
    Read local image file and return data URL: data:image/png;base64,...
    If suffix unknown, default to octet-stream.
    """

    p = Path(image_path)
    data = p.read_bytes()
    b64 = base64.b64encode(data).decode("utf-8")

    mime = _guess_mime(p.suffix.lower())
    return f"data:{mime};base64,{b64}"


def _guess_mime(ext: str) -> str:
    if ext in [".jpg", ".jpeg"]:
        return "image/jpeg"
    if ext == ".png":
        return "image/png"
    if ext == ".webp":
        return "image/webp"
    if ext == ".gif":
        return "image/gif"
    return "application/octet-stream"


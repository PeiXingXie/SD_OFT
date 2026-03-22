"""
Message builder for Andes Gateway multimodal chat.

Andes gateway expects images to be passed via a top-level `images` field on the user
message:
  {"role":"user","content":"...","images":[{"url":"data:...","detail":"auto"}]}
"""

from __future__ import annotations


def build_messages(
    system_prompt: str,
    user_prompt: str,
    *,
    image_data_url: str | None = None,
    image_detail: str = "auto",
) -> list[dict]:
    system_prompt = system_prompt or ""
    user_prompt = user_prompt or ""

    base_msgs = [{"role": "system", "content": system_prompt}]

    u = image_data_url.strip() if isinstance(image_data_url, str) and image_data_url.strip() else None
    if not u:
        base_msgs.append({"role": "user", "content": user_prompt})
        return base_msgs

    detail = str(image_detail or "auto").strip().lower()
    if detail not in ["low", "high", "auto"]:
        detail = "auto"

    base_msgs.append(
        {
            "role": "user",
            "content": user_prompt + "\n\n[image_mapping]\n- image is images[0]",
            "images": [{"url": u, "detail": detail}],
        }
    )
    return base_msgs


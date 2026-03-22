from __future__ import annotations

from .config import ImageGenConfig
from .image_client import AndesGatewayImageClient


def build_client(cfg: ImageGenConfig) -> AndesGatewayImageClient:
    if cfg.provider != "andes_gateway":
        raise ValueError(f"Unsupported provider: {cfg.provider}")
    return AndesGatewayImageClient(cfg)


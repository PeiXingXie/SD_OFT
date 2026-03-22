from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional


@dataclass(frozen=True)
class OFTArgs:
    oft_block_size: int = 32
    use_cayley_neumann: bool = True
    module_dropout: float = 0.0
    bias: str = "none"  # "none" | "all" | "oft_only"
    target_modules: List[str] = None  # filled by default factory below


def default_target_modules() -> List[str]:
    # SD1.5 UNet attention projections in diffusers
    return ["to_q", "to_k", "to_v", "to_out.0"]


def apply_oft_to_unet(
    unet,
    *,
    oft_block_size: int = 32,
    use_cayley_neumann: bool = True,
    module_dropout: float = 0.0,
    bias: str = "none",
    target_modules: Optional[Iterable[str]] = None,
):
    """
    Wrap a diffusers UNet with PEFT OFT adapters.

    Returns:
      peft.peft_model.PeftModel
    """
    from peft import OFTConfig, get_peft_model

    if target_modules is None:
        target_modules = default_target_modules()
    target_modules = list(target_modules)

    cfg = OFTConfig(
        oft_block_size=int(oft_block_size),
        use_cayley_neumann=bool(use_cayley_neumann),
        module_dropout=float(module_dropout),
        bias=str(bias),
        target_modules=target_modules,
    )
    return get_peft_model(unet, cfg)


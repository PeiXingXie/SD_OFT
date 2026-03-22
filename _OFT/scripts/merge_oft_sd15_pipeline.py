from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline
from peft import PeftModel


_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _default_model_dir(subdir: str, env_var: str) -> str:
    v = os.environ.get(env_var, "").strip()
    if v:
        return str(Path(v).expanduser())
    base = os.environ.get("OFT_MODELS_DIR", "").strip()
    if base:
        return str((Path(base).expanduser() / subdir).resolve())
    return str((_PROJECT_ROOT / "models" / subdir).resolve())


DEFAULT_SD15_DIR = _default_model_dir("stable-diffusion-v1-5", "SD15_MODEL_DIR")

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", type=str, default=DEFAULT_SD15_DIR)
    p.add_argument("--adapter_dir", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument(
        "--torch_dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
    )
    return p


def _dtype_from_str(s: str) -> torch.dtype:
    if s == "float16":
        return torch.float16
    if s == "bfloat16":
        return torch.bfloat16
    if s == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {s}")


def main() -> None:
    args = build_argparser().parse_args()
    dtype = _dtype_from_str(args.torch_dtype)

    pipe = StableDiffusionPipeline.from_pretrained(
        args.model_dir,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )

    peft_unet = PeftModel.from_pretrained(pipe.unet, args.adapter_dir)
    pipe.unet = peft_unet.merge_and_unload()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pipe.save_pretrained(str(out_dir), safe_serialization=True)
    print(f"Saved merged pipeline: {out_dir}")


if __name__ == "__main__":
    main()


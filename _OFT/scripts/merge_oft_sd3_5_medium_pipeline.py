from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from diffusers import StableDiffusion3Pipeline
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


DEFAULT_SD35_MEDIUM_DIR = _default_model_dir("stable-diffusion-3.5-medium", "SD35_MODEL_DIR")

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", type=str, default=DEFAULT_SD35_MEDIUM_DIR)
    p.add_argument("--adapter_dir", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument(
        "--torch_dtype",
        type=str,
        default="auto",
        choices=["auto", "bfloat16", "float16", "float32"],
    )
    p.add_argument("--use_safetensors", action="store_true", default=True)
    p.add_argument("--no_use_safetensors", action="store_false", dest="use_safetensors")
    return p


def _dtype_from_str(s: str) -> torch.dtype:
    if s == "bfloat16":
        return torch.bfloat16
    if s == "float16":
        return torch.float16
    if s == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {s}")


def main() -> None:
    args = build_argparser().parse_args()

    if args.torch_dtype == "auto":
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        if not torch.cuda.is_available():
            dtype = torch.float32
    else:
        dtype = _dtype_from_str(args.torch_dtype)

    pipe = StableDiffusion3Pipeline.from_pretrained(
        args.model_dir,
        torch_dtype=dtype,
        use_safetensors=bool(args.use_safetensors),
    )

    peft_transformer = PeftModel.from_pretrained(pipe.transformer, args.adapter_dir)
    pipe.transformer = peft_transformer.merge_and_unload()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pipe.save_pretrained(str(out_dir), safe_serialization=True)
    print(f"Saved merged pipeline: {out_dir}")


if __name__ == "__main__":
    main()


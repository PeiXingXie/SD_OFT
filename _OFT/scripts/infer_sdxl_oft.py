from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import torch
from diffusers import DiffusionPipeline
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


DEFAULT_SDXL_BASE_DIR = _default_model_dir("stable-diffusion-xl-base-1.0", "SDXL_MODEL_DIR")

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", type=str, default=DEFAULT_SDXL_BASE_DIR)
    p.add_argument("--adapter_dir", type=str, required=True)
    p.add_argument("--device", type=str, default=None)
    p.add_argument(
        "--torch_dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
    )
    p.add_argument("--use_safetensors", action="store_true", default=True)
    p.add_argument("--no_use_safetensors", action="store_false", dest="use_safetensors")
    p.add_argument(
        "--variant",
        type=str,
        default="fp16",
        help='Model variant, e.g. "fp16". Only used when dtype=float16.',
    )
    p.add_argument("--prompt", type=str, required=True)
    p.add_argument("--negative_prompt", type=str, default=None)
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--guidance", type=float, default=7.5)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--height", type=int, default=None)
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--out", type=str, required=True)
    p.add_argument(
        "--merge_and_unload",
        action="store_true",
        default=False,
        help="Merge OFT into UNet weights for inference (slightly faster, no adapter dependency).",
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

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    dtype = _dtype_from_str(args.torch_dtype)
    variant = args.variant if (dtype == torch.float16 and args.variant) else None

    pipe = DiffusionPipeline.from_pretrained(
        args.model_dir,
        torch_dtype=dtype,
        use_safetensors=bool(args.use_safetensors),
        variant=variant,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)

    peft_unet = PeftModel.from_pretrained(pipe.unet, args.adapter_dir)
    if args.merge_and_unload:
        pipe.unet = peft_unet.merge_and_unload()
    else:
        pipe.unet = peft_unet

    generator: Optional[torch.Generator] = None
    if args.seed is not None:
        generator = torch.Generator(device=device).manual_seed(int(args.seed))

    kwargs = {}
    if args.height is not None:
        kwargs["height"] = int(args.height)
    if args.width is not None:
        kwargs["width"] = int(args.width)

    with torch.inference_mode():
        out = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            num_inference_steps=int(args.steps),
            guidance_scale=float(args.guidance),
            generator=generator,
            **kwargs,
        )

    img = out.images[0]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(out_path))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()


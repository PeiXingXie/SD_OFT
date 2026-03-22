"""
Runnable demo for SDXL Base 1.0 (local weights, base only).

Example:
  python BaseModel/SDXL-Base-1.0/demo_sdxl_base.py \
    --prompt "An astronaut riding a green horse" \
    --out outputs/sdxl.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

from sdxl_base import DEFAULT_SDXL_BASE_DIR, SDXLBase10


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", type=str, default=DEFAULT_SDXL_BASE_DIR)
    p.add_argument("--device", type=str, default=None, help='e.g. "cuda", "cuda:0", "cpu"')
    p.add_argument(
        "--torch_dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
    )
    p.add_argument("--variant", type=str, default="fp16", help='Usually "fp16" for SDXL base.')

    p.add_argument("--prompt", type=str, required=True)
    p.add_argument("--negative_prompt", type=str, default=None)
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--guidance", type=float, default=7.5)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--height", type=int, default=None)
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--num_images_per_prompt", type=int, default=None)
    p.add_argument("--out", type=str, required=True)
    return p


def main() -> None:
    args = build_argparser().parse_args()

    torch_dtype = None if args.torch_dtype == "auto" else args.torch_dtype
    variant = args.variant if args.variant else None

    model = SDXLBase10(
        model_dir=args.model_dir,
        device=args.device,
        torch_dtype=torch_dtype,
        variant=variant,
    )
    out_path = model.generate_to_file(
        args.prompt,
        Path(args.out),
        negative_prompt=args.negative_prompt,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        seed=args.seed,
        height=args.height,
        width=args.width,
        num_images_per_prompt=args.num_images_per_prompt,
    )
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()


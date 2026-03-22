"""
Runnable demo for Stable Diffusion v1.5 (local weights).

Example:
  python BaseModel/SD-1-5/demo_sd1_5.py \
    --prompt "a photo of an astronaut riding a horse on mars" \
    --out outputs/astronaut_rides_horse.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

from sd1_5 import DEFAULT_SD15_DIR, SD15


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", type=str, default=DEFAULT_SD15_DIR)
    p.add_argument("--device", type=str, default=None)
    p.add_argument(
        "--torch_dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
    )
    p.add_argument("--prompt", type=str, required=True)
    p.add_argument("--negative_prompt", type=str, default=None)
    p.add_argument("--steps", type=int, default=30)
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
    model = SD15(
        model_dir=args.model_dir,
        device=args.device,
        torch_dtype=torch_dtype,
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


"""
Runnable demo for Stable Diffusion 3.5 Medium (local weights).

Example:
  python BaseModel/SD-3-5-Medium/demo_sd3_5_medium.py \
    --prompt "A capybara holding a sign that reads Hello World" \
    --out outputs/capybara.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

from sd3_5_medium import DEFAULT_SD35_MEDIUM_DIR, SD35Medium


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model_dir",
        type=str,
        default=DEFAULT_SD35_MEDIUM_DIR,
        help="Local model directory (diffusers format).",
    )
    p.add_argument("--device", type=str, default=None, help='e.g. "cuda", "cuda:0", "cpu"')
    p.add_argument(
        "--torch_dtype",
        type=str,
        default="auto",
        choices=["auto", "bfloat16", "float16", "float32"],
        help="Torch dtype override (default: auto).",
    )
    p.add_argument("--prompt", type=str, required=True)
    p.add_argument("--negative_prompt", type=str, default=None)
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--guidance", type=float, default=4.5)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--out", type=str, required=True, help="Output image path, e.g. /tmp/out.png")
    return p


def main():
    args = build_argparser().parse_args()

    torch_dtype = None if args.torch_dtype == "auto" else args.torch_dtype
    model = SD35Medium(
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
    )
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()


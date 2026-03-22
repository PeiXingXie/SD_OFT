"""
Batch run Stable Diffusion 3.5 Medium with a single model load.

Example:
  python BaseModel/SD-3-5-Medium/batch_run_sd3_5_medium.py \
    --csv_file prompts.csv \
    --prompt_col prompt \
    --out_dir outputs/sd35_out
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from sd3_5_medium import DEFAULT_SD35_MEDIUM_DIR, get_sd35_medium


def read_prompts_from_csv(
    csv_path: Path,
    prompt_col: str,
    *,
    encoding: str = "utf-8",
    delimiter: str = ",",
) -> list[str]:
    if not csv_path.exists():
        raise FileNotFoundError(f"csv_file not found: {csv_path}")

    with csv_path.open("r", encoding=encoding, newline="") as f:
        reader = csv.reader(f, delimiter=delimiter)
        try:
            header = next(reader)
        except StopIteration as e:
            raise ValueError(f"CSV is empty: {csv_path}") from e

        header = [h.lstrip("\ufeff").strip() for h in header]
        if not header or all(h == "" for h in header):
            raise ValueError(f"CSV header row is empty: {csv_path}")

        if prompt_col.isdigit():
            col_idx = int(prompt_col)
        else:
            try:
                col_idx = header.index(prompt_col)
            except ValueError as e:
                raise ValueError(
                    f'prompt_col "{prompt_col}" not found in CSV header. '
                    f"Available columns: {header}"
                ) from e

        prompts: list[str] = []
        for row in reader:
            if col_idx >= len(row):
                continue
            prompt = row[col_idx].strip()
            if prompt:
                prompts.append(prompt)

    if not prompts:
        raise ValueError(f"No prompts found in CSV: {csv_path} (prompt_col={prompt_col})")
    return prompts


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
    p.add_argument("--negative_prompt", type=str, default=None)
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--guidance", type=float, default=4.5)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument(
        "--csv_file",
        "--prompts_file",
        dest="csv_file",
        type=str,
        required=True,
        help="CSV file containing prompts (use --prompt_col to choose the column).",
    )
    p.add_argument(
        "--prompt_col",
        type=str,
        default="prompt",
        help='Prompt column in CSV header (default: "prompt"). '
        "You may also pass a zero-based column index, e.g. 0.",
    )
    p.add_argument("--csv_encoding", type=str, default="utf-8")
    p.add_argument("--csv_delimiter", type=str, default=",")
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--ext", type=str, default="png", choices=["png", "jpg", "webp"])
    return p


def main():
    args = build_argparser().parse_args()

    prompts_path = Path(args.csv_file)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prompts = read_prompts_from_csv(
        prompts_path,
        args.prompt_col,
        encoding=args.csv_encoding,
        delimiter=args.csv_delimiter,
    )

    torch_dtype = None if args.torch_dtype == "auto" else args.torch_dtype
    model = get_sd35_medium(
        args.model_dir,
        device=args.device,
        torch_dtype=torch_dtype,
    )

    for i, prompt in enumerate(prompts):
        out_path = out_dir / f"{i:05d}.{args.ext}"
        model.generate_to_file(
            prompt,
            out_path,
            negative_prompt=args.negative_prompt,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            seed=args.seed,
        )
        print(f"[{i+1}/{len(prompts)}] saved: {out_path}")


if __name__ == "__main__":
    main()


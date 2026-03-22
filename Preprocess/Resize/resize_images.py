from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path


DEFAULT_IMAGE_EXTS = [".png", ".jpg", ".jpeg", ".webp", ".gif"]


@dataclass
class Stats:
    total_found: int = 0
    processed: int = 0
    skipped_exists: int = 0
    skipped_unsupported: int = 0
    failed: int = 0
    warn_in_size: int = 0
    warn_out_size: int = 0
    warn_animated_gif: int = 0


def _eprint(msg: str) -> None:
    print(msg, file=sys.stderr)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Batch resize images from input_dir to output_dir (default 1024x1024 -> 512x512).",
    )
    p.add_argument("--input_dir", type=str, required=True, help="Input directory containing images.")
    p.add_argument("--output_dir", type=str, required=True, help="Output directory to write resized images.")

    p.add_argument(
        "--in_size",
        type=int,
        nargs=2,
        metavar=("W", "H"),
        default=[1024, 1024],
        help="Expected input resolution; samples not matching will be warned (default: 1024 1024).",
    )
    p.add_argument(
        "--out_size",
        type=int,
        nargs=2,
        metavar=("W", "H"),
        default=[512, 512],
        help="Target output resolution (default: 512 512).",
    )
    p.add_argument(
        "--recursive",
        action="store_true",
        default=False,
        help="Enable recursive scanning.",
    )
    p.add_argument(
        "--no_recursive",
        action="store_true",
        help="Disable recursive scanning (default is recursive).",
    )
    p.add_argument(
        "--exts",
        type=str,
        default=",".join(DEFAULT_IMAGE_EXTS),
        help='Comma-separated extensions to include (default: ".png,.jpg,.jpeg,.webp,.gif").',
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files (default: skip if exists).",
    )
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Only print planned operations; do not write outputs.",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-file logs.",
    )
    return p.parse_args()


def _iter_images(input_dir: Path, *, recursive: bool, exts: list[str]) -> list[Path]:
    pat = "**/*" if recursive else "*"
    out: list[Path] = []
    exts_norm = {e.lower() for e in exts}
    for fp in input_dir.glob(pat):
        if not fp.is_file():
            continue
        if fp.suffix.lower() in exts_norm:
            out.append(fp)
    out.sort()
    return out


def _maybe_tqdm(iterable, total: int):
    try:
        from tqdm import tqdm  # type: ignore

        return tqdm(iterable, total=total)
    except Exception:
        return iterable


def _resize_one(
    src_path: Path,
    *,
    input_root: Path,
    output_root: Path,
    expected_in_size: tuple[int, int],
    out_size: tuple[int, int],
    overwrite: bool,
    dry_run: bool,
    verbose: bool,
    stats: Stats,
) -> None:
    # Lazy import so this file can be imported without Pillow installed, but requires Pillow to run.
    try:
        from PIL import Image, ImageOps  # type: ignore
    except Exception as e:
        raise RuntimeError("Pillow is required. Please install: pip install pillow") from e

    rel = src_path.relative_to(input_root)
    dst_path = output_root / rel

    if dst_path.exists() and not overwrite:
        stats.skipped_exists += 1
        if verbose:
            _eprint(f"[skip-exists] {dst_path}")
        return

    dst_path.parent.mkdir(parents=True, exist_ok=True)

    if dry_run:
        if verbose:
            _eprint(f"[dry-run] {src_path} -> {dst_path}")
        stats.processed += 1
        return

    try:
        with Image.open(src_path) as img0:
            # Fix EXIF orientation if present.
            img = ImageOps.exif_transpose(img0)

            # GIF handling: PIL may expose is_animated.
            if src_path.suffix.lower() == ".gif" and getattr(img0, "is_animated", False):
                stats.warn_animated_gif += 1
                _eprint(f"[warn] animated GIF detected; only first frame will be processed: {src_path}")

            in_w, in_h = img.size
            if (in_w, in_h) != expected_in_size:
                stats.warn_in_size += 1
                _eprint(
                    f"[warn] input size mismatch: got {in_w}x{in_h}, expected {expected_in_size[0]}x{expected_in_size[1]} :: {src_path}"
                )

            # Ensure a sane mode for saving to common formats.
            if img.mode not in ("RGB", "RGBA"):
                img = img.convert("RGB")

            resized = img.resize(out_size, resample=Image.Resampling.LANCZOS)

            # Validate output size in-memory.
            out_w, out_h = resized.size
            if (out_w, out_h) != out_size:
                stats.warn_out_size += 1
                _eprint(
                    f"[warn] output size mismatch (in-memory): got {out_w}x{out_h}, expected {out_size[0]}x{out_size[1]} :: {dst_path}"
                )

            save_kwargs = {}
            # Better JPEG defaults; keep original extension.
            if dst_path.suffix.lower() in (".jpg", ".jpeg"):
                save_kwargs.update({"quality": 95, "subsampling": 0, "optimize": True})
            if dst_path.suffix.lower() == ".png":
                save_kwargs.update({"optimize": True})

            resized.save(dst_path, **save_kwargs)

        # Re-open to validate actual written image size (cheap, but catches weird saves).
        with Image.open(dst_path) as out_img:
            out_img = ImageOps.exif_transpose(out_img)
            ow, oh = out_img.size
            if (ow, oh) != out_size:
                stats.warn_out_size += 1
                _eprint(
                    f"[warn] output size mismatch (on-disk): got {ow}x{oh}, expected {out_size[0]}x{out_size[1]} :: {dst_path}"
                )

        stats.processed += 1
        if verbose:
            _eprint(f"[ok] {src_path} -> {dst_path}")
    except Exception as e:
        stats.failed += 1
        _eprint(f"[fail] {src_path}: {type(e).__name__}: {e}")


def main() -> int:
    args = _parse_args()
    input_root = Path(args.input_dir).expanduser().resolve()
    output_root = Path(args.output_dir).expanduser().resolve()

    if not input_root.exists():
        _eprint(f"input_dir not found: {input_root}")
        return 2
    if not input_root.is_dir():
        _eprint(f"input_dir is not a directory: {input_root}")
        return 2

    # Default behavior: recursive=True unless explicitly disabled.
    if bool(args.no_recursive):
        recursive = False
    elif bool(args.recursive):
        recursive = True
    else:
        recursive = True
    exts = [e.strip().lower() for e in str(args.exts).split(",") if e.strip()]
    if not exts:
        _eprint("No extensions configured via --exts")
        return 2

    expected_in_size = (int(args.in_size[0]), int(args.in_size[1]))
    out_size = (int(args.out_size[0]), int(args.out_size[1]))
    if expected_in_size[0] <= 0 or expected_in_size[1] <= 0 or out_size[0] <= 0 or out_size[1] <= 0:
        _eprint("--in_size/--out_size must be positive integers")
        return 2

    images = _iter_images(input_root, recursive=recursive, exts=exts)
    stats = Stats(total_found=len(images))

    if stats.total_found == 0:
        _eprint(f"No images found in: {input_root} (exts={exts}, recursive={recursive})")
        return 1

    _eprint(
        f"Found {stats.total_found} images. in_size={expected_in_size[0]}x{expected_in_size[1]} out_size={out_size[0]}x{out_size[1]} overwrite={args.overwrite} dry_run={args.dry_run}"
    )

    for fp in _maybe_tqdm(images, total=stats.total_found):
        _resize_one(
            fp,
            input_root=input_root,
            output_root=output_root,
            expected_in_size=expected_in_size,
            out_size=out_size,
            overwrite=bool(args.overwrite),
            dry_run=bool(args.dry_run),
            verbose=bool(args.verbose),
            stats=stats,
        )

    _eprint("=== Summary ===")
    _eprint(f"total_found: {stats.total_found}")
    _eprint(f"processed: {stats.processed}")
    _eprint(f"skipped_exists: {stats.skipped_exists}")
    _eprint(f"failed: {stats.failed}")
    _eprint(f"warn_in_size: {stats.warn_in_size}")
    _eprint(f"warn_out_size: {stats.warn_out_size}")
    _eprint(f"warn_animated_gif: {stats.warn_animated_gif}")

    # Non-zero exit if there were failures.
    return 0 if stats.failed == 0 else 3


if __name__ == "__main__":
    raise SystemExit(main())


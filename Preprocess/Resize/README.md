# Resize (batch image resolution conversion)

## What
Batch resize images in a directory (optionally recursive), mirroring the input directory structure into an output directory.

Default behavior is **1024x1024 → 512x512**, and it will print **warnings** when:
- input sample resolution is not the expected `--in_size`
- output sample resolution is not the expected `--out_size`

## Usage

### 1) Dry-run (recommended first)
```bash
python Preprocess/Resize/resize_images.py \
  --input_dir  /path/to/input_images \
  --output_dir /path/to/output_images \
  --dry_run --verbose
```

### 2) Run (default 1024→512)
```bash
python Preprocess/Resize/resize_images.py \
  --input_dir  /path/to/input_images \
  --output_dir /path/to/output_images
```

### 3) Customize sizes
```bash
python Preprocess/Resize/resize_images.py \
  --input_dir  /path/to/input_images \
  --output_dir /path/to/output_images \
  --in_size 1024 1024 \
  --out_size 768 768
```

### 4) Overwrite outputs
```bash
python Preprocess/Resize/resize_images.py \
  --input_dir  /path/to/input_images \
  --output_dir /path/to/output_images \
  --overwrite
```

## Notes
- Supported input extensions default to: `.png,.jpg,.jpeg,.webp,.gif` (configurable via `--exts`).
- For **animated GIF**, the script will warn and resize **only the first frame**.


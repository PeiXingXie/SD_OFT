# BaseModel (base model wrappers / demo entrypoints)

This folder provides lightweight wrappers and runnable demos for **Stable Diffusion family base models** (Diffusers), used for OFT experiments and data generation.

- You typically start by running `demo_*.py` in each subfolder to validate inference works
- If you need repeated calls without re-loading weights, use **batch/daemon** scripts provided in each subfolder

## Layout

- `SD-1-5/`: Stable Diffusion v1.5 (Diffusers) wrapper + demo
- `SDXL-Base-1.0/`: SDXL base 1.0 (base-only, no refiner) wrapper + demo
- `SD-3-5-Medium/`: Stable Diffusion 3.5 Medium (Diffusers) wrapper + demo
- `output/`: runtime/validation outputs (e.g. `sd15_valid/`, `sdxl_valid/`, `sd35_valid/`)

> Note: subfolder names may contain `-` (e.g. `SD-3-5-Medium`), which is not a valid Python module identifier; avoid imports like `import BaseModel.SD-3-5-Medium`.
> Recommendation: **run the scripts directly** from the subfolder, or follow each subfolder README to adjust `sys.path` for imports.

## Where to put model weights (shared convention)

All subfolders follow the same convention to locate local models (Diffusers format; the directory should contain `model_index.json`):

- Default root: `models/`
- Override the root via `OFT_MODELS_DIR` (scripts resolve `${OFT_MODELS_DIR}/<model-subdir>`)
- Or specify per-model dir via:
  - `SD15_MODEL_DIR`: SD1.5 model dir
  - `SDXL_MODEL_DIR`: SDXL base model dir
  - `SD35_MODEL_DIR`: SD3.5 medium model dir

Example (place all models under one custom root):

```bash
export OFT_MODELS_DIR=/mnt/workspace/open_source_model
```

## Quickstart (run from the project root)

### SD1.5

```bash
python BaseModel/SD-1-5/demo_sd1_5.py \
  --prompt "a photo of an astronaut riding a horse on mars" \
  --out outputs/sd15.png
```

### SDXL base 1.0（base-only）

```bash
python BaseModel/SDXL-Base-1.0/demo_sdxl_base.py \
  --prompt "An astronaut riding a green horse" \
  --out outputs/sdxl.png
```

### SD3.5 Medium

```bash
python BaseModel/SD-3-5-Medium/demo_sd3_5_medium.py \
  --prompt "A capybara holding a sign that reads Hello World" \
  --steps 40 \
  --guidance 4.5 \
  --out outputs/sd35.png
```

## Dependencies (overview)

Scripts in subfolders typically depend on:

- PyTorch（建议 CUDA 版）
- diffusers（以及 `transformers` / `accelerate` / `safetensors` 等）

For detailed parameters / daemon / batch usage, see the subfolder READMEs:

- `BaseModel/SD-1-5/README.md`
- `BaseModel/SDXL-Base-1.0/README.md`
- `BaseModel/SD-3-5-Medium/README.md`


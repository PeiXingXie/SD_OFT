# OFT_for_SD (OFT fine-tuning for Stable Diffusion + tools)

This project is an experiment/workbench around **OFT (Orthogonal Finetuning, via PEFT)**, covering:

- **Training / inference (OFT adapters)**: SD1.5 / SDXL Base 1.0 / SD3.5 Medium (Diffusers)
- **Preprocessing**: CSV editing, batch resize, external T2I augmentation, MLLM extraction/evaluation
- **Visualization**: validation generations comparison (baseline + steps), CSV gallery viewer & filtering

If you just want the shortest path to a working run, jump to **Quickstart (end-to-end)** below. For full details, read the per-module READMEs.

---

## Repo layout (where to start)

- **`_OFT/`**: OFT training / inference / merge scripts (core)
  - See: `_OFT/README.md`
- **`BaseModel/`**: base model wrappers + demos/daemons (useful to smoke-test local weights first)
  - See: `BaseModel/README.md`
- **`Preprocess/`**: preprocessing toolbox (CSV / resize / augmentation / MLLM)
  - See: `Preprocess/README.md`
- **`Visualization/`**: local HTML viewers + lightweight file API (validation compare, CSV gallery)
  - See: `Visualization/README.md`
- **`Dataset/`**: dataset CSVs and example experiment folders (e.g. `Dataset/Exps/PointillismExpand/*.csv`)
  - See: `Dataset/README.md`
- **`models/`**: default location for local diffusers weights (usually you provide these yourself)
  - See: `models/README.md`

---

## Environment setup

### Training / inference deps (`_OFT`)

```bash
python3 -m pip install -r _OFT/requirements.txt
```

> Submodules under `Preprocess/` may require extra dependencies (e.g. Pillow / requests / yaml). Install them per `Preprocess/*/README.md`.

### Local model weights (Diffusers format)

By default, scripts expect these folders to exist and contain `model_index.json`:

- `models/stable-diffusion-v1-5/`
- `models/stable-diffusion-xl-base-1.0/`
- `models/stable-diffusion-3.5-medium/`

You can override via environment variables:

- **Common root**: `OFT_MODELS_DIR=/abs/path/to/models_root`
- **Per-model**: `SD15_MODEL_DIR` / `SDXL_MODEL_DIR` / `SD35_MODEL_DIR`

See: `models/README.md` and `BaseModel/README.md`.

---

## Data input (recommended: CSV)

`_OFT/scripts/train_*.py` training scripts accept CSV files for train/val, and column names are configurable via CLI:

- `--image_col`: image path column (absolute paths supported; relative paths are resolved against the CSV parent dir, or `--image_root`)
- `--text_col`: prompt text column
- (optional) `--negative_col`: negative prompt column

This repo includes example CSVs (column names: `path` + `caption`):

- `Dataset/Exps/PointillismExpand/train_captions.csv`
- `Dataset/Exps/PointillismExpand/valid_captions.csv`

### Images (important: extract into `Dataset/Images` first)

The train/val CSVs reference image paths under `Dataset/Images/`. Before running anything, download `TrainImagesforSDOFT.tar.gz` from GitHub Releases and extract it to:

- `/mnt/workspace/xiepeixing/Project/_TestProject/OFT_for_SD/Dataset/Images`

Example (run from the project root):

```bash
mkdir -p Dataset/Images

# Recommended: inspect archive paths first
tar -tzf TrainImagesforSDOFT.tar.gz | head

# If the archive directly contains *.jpg (or contains an images/ folder with images), extract into Dataset/Images:
tar -xzf TrainImagesforSDOFT.tar.gz -C Dataset/Images
```

---

## Quickstart (end-to-end: train once → infer → visualize)

The commands below assume you run from `Project/_TestProject/OFT_for_SD`.

### 1) Configure local weights paths (choose one)

```bash
export OFT_MODELS_DIR="/abs/path/to/your/models_root"
```

Or set per-model variables:

```bash
export SD15_MODEL_DIR="/abs/path/to/stable-diffusion-v1-5"
export SDXL_MODEL_DIR="/abs/path/to/stable-diffusion-xl-base-1.0"
export SD35_MODEL_DIR="/abs/path/to/stable-diffusion-3.5-medium"
```

### 2) Smoke-test weights via a BaseModel demo

```bash
python BaseModel/SD-1-5/demo_sd1_5.py \
  --prompt "a photo of an astronaut riding a horse on mars" \
  --out outputs/sd15_smoke.png
```

More demos (SDXL/SD3.5): see `BaseModel/README.md`.

### 3) Train (SD1.5 example)

```bash
accelerate launch _OFT/scripts/train_sd15_oft_t2i.py \
  --model_dir models/stable-diffusion-v1-5 \
  --train_csv Dataset/Exps/PointillismExpand/train_captions.csv \
  --val_csv Dataset/Exps/PointillismExpand/valid_captions.csv \
  --image_col path --text_col caption \
  --output_dir _OFT/outputs/sd15_run_demo \
  --resolution 512 \
  --train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --max_train_steps 200 \
  --save_steps 50
```

Typical output layout:

- `_OFT/outputs/<run>/adapter/step_000050/`: OFT adapter (one subdir per saved step)
- `_OFT/outputs/<run>/logs/metrics.csv`: training metrics (loss / grad_norm / timing)
- `_OFT/outputs/<run>/validation/baseline/`: baseline (no adapter) validation generations (written once)
- `_OFT/outputs/<run>/validation/step_000050/`: validation generations for a specific step (for comparison)

SDXL / SD3.5 Medium commands and differences: see `_OFT/README.md`.

### 4) Inference (load a specific step adapter)

```bash
python _OFT/scripts/infer_sd15_oft.py \
  --model_dir models/stable-diffusion-v1-5 \
  --adapter_dir _OFT/outputs/sd15_run_demo/adapter/step_000200 \
  --prompt "a pointillism painting of a coastal town" \
  --out outputs/oft_sd15.png
```

### 5) Visualization: compare validation (baseline + steps) in browser

Start the visualization server (**always** set `--allow-prefix` allowlist):

```bash
python3 Visualization/server.py \
  --bind 127.0.0.1 --port 8008 \
  --static-dir Visualization \
  --allow-prefix _OFT/outputs \
  --allow-prefix Dataset
```

Then open the compare page:

- `http://127.0.0.1:8008/_View_ALL_check_compare.html`

In the page, set the validation root directory:

- `_OFT/outputs/<run>/validation`

More pages/APIs/security notes: see `Visualization/README.md`.

---

## Common workflows (recommended)

- **Prepare data / fix CSVs**: `Preprocess/easy_act_csv.py` (add/merge/drop columns, path rewrite, pipelines)
  - Entry doc: `Preprocess/README.md`
- **Batch resize**: `Preprocess/Resize/resize_images.py`
- **Augmentation (external T2I API)**: `Preprocess/Sample_expand/`
- **MLLM extraction / evaluation**: `Preprocess/MLLM_Extract/`
- **Training curve analysis**: `_OFT/bestcheckpoint/plot_metrics.ipynb` (reads `logs/metrics.csv`)

---

## Security notes (visualization server)

`Visualization/server.py` can read files by absolute path (guarded by `--allow-prefix` allowlists). Recommendations:

- **Always set `--allow-prefix` and keep it minimal**
- **Do not expose it to the public internet** (do not port-forward it publicly)

---

## Further reading

- Training / inference / OFT parameters: `_OFT/README.md`
- Base model demos/daemons: `BaseModel/README.md`
- Preprocessing toolbox: `Preprocess/README.md`
- Visualization pages & APIs: `Visualization/README.md`
- Local weights convention: `models/README.md`


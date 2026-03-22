# Dataset (OFT_for_SD)

This folder stores datasets and configuration files (CSV) used for **OFT-for-SD** training/validation.

## Layout

```text
Dataset/
  Images/                         # training images (jpg/png/...)
  Exps/                           # experiment presets (CSV configs)
    PointillismExpand/
      train_captions.csv
      valid_captions.csv
    PointillismHalf/
      train_captions.csv
      valid_captions.csv
```

## How CSV files are read (very important)

Training scripts (e.g. `_OFT/scripts/train_sd15_oft_t2i.py`) read CSV files via `_OFT/utils/csv_data.py`:

- **Train CSV**: requires at least two columns
  - `path` (selected by `--image_col`): image path (absolute or relative)
  - `caption` (selected by `--text_col`): prompt text
- **Val CSV**: the script only reads `caption` (via `--text_col`) as validation prompts (it does **not** read images).

### Relative path resolution rules

For the `path` field in the train CSV:

- If `path` is **absolute**: use it directly.
- If `path` is **relative**:
  - If `--image_root` is provided: final path = `image_root / path`
  - Otherwise: final path = (parent directory of `train_captions.csv`) / `path`

Therefore:

- In this repo, `train_captions.csv` typically uses `Dataset/Images/xxx.jpg` (relative to the project root).
- To make it resolve correctly, it’s recommended to explicitly set:
  - `--image_root .` (make sure your current working directory is `Project/_TestProject/OFT_for_SD/`)
  - or `--image_root /mnt/workspace/.../Project/_TestProject/OFT_for_SD` (absolute path)

## `train_captions.csv` columns (PointillismExpand example)

文件头示例：

```text
id,path,caption,__raw_response,__error,style,Gen_by,category_category,task
```

- **Columns actually used for training**
  - `path`: image path (see the resolution rules above)
  - `caption`: prompt (rows with empty caption are skipped)
- **Other columns (optional / metadata)**
  - `id`: sample id
  - `style`, `Gen_by`, `category_category`, `task`, `__raw_response`, `__error`: labeling/provenance/preprocess logs; ignored by training

> Note: if `path`/`caption` are non-empty but the image file does not exist, training raises `FileNotFoundError` immediately.

## `valid_captions.csv` columns

`valid_captions.csv` is essentially a “validation prompt list”. The script only cares about `caption` (selected by `--text_col`).

常见字段：

```text
valid_index,id,path,caption,...,valid_suffix,valid_id
```

- `caption`: prompt for validation generation
- `valid_id` / `valid_suffix`: often used to save generations with stable filenames (e.g. `0000.png`); usage depends on your evaluation/visualization pipeline
- `path` being `nan` is acceptable (validation does not use images)

## How to use experiment presets (`Exps/`)

Each experiment folder (e.g. `Exps/PointillismExpand/`) contains a pair of train/val CSV files.

Example with SD1.5 (run from `Project/_TestProject/OFT_for_SD/`):

```bash
python _OFT/scripts/train_sd15_oft_t2i.py \
  --output_dir _OFT/outputs/sd15_std-s_exp \
  --train_csv Dataset/Exps/PointillismExpand/train_captions.csv \
  --val_csv   Dataset/Exps/PointillismExpand/valid_captions.csv \
  --image_col path \
  --text_col  caption \
  --image_root .
```

## Known notes (PointillismHalf)

Some `path` values in `Exps/PointillismHalf/train_captions.csv` may be legacy paths (e.g. pointing to `Dataset/_Style_Set`, `Dataset/Intermediate`, etc.).

- If your `Dataset/` only contains `Images/` and `Exps/`, those legacy paths will cause missing files during training.
- Two ways to fix:
  - **Edit the CSV**: rewrite `path` to match real existing paths (e.g. `Dataset/Images/...`)
  - **Provide the referenced folders**: place external datasets at the expected locations


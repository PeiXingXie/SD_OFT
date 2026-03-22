# OFT 微调 Stable Diffusion 1.5（T2I）

本目录提供一套**用 PEFT 的 OFT（Orthogonal Finetuning）**对 **Stable Diffusion v1.5** 进行 **Text-to-Image** 微调的最小工程化代码：

- 训练：注入 OFT 到 SD1.5 的 `UNet`（注意力投影 `to_q/to_k/to_v/to_out.0`）
- 保存：只保存 OFT adapter（参数量很小）
- 推理：加载 adapter 进行生成；可选 **merge** 到 UNet 并导出完整 pipeline

参考：
- OFT 参数说明（PEFT 官方文档）：`https://huggingface.co/docs/peft/main/en/conceptual_guides/oft#oft-specific-parameters`
- SD1.5 模型卡：`https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5`

## TL;DR（Quickstart）

从 `Project/_TestProject/OFT_for_SD` 目录执行（推荐）：

1) 安装依赖：

```bash
python3 -m pip install -r _OFT/requirements.txt
```

2) 配置本地 diffusers 权重目录（二选一）：

- **方式 A（推荐）**：设置统一根目录：

```bash
export OFT_MODELS_DIR="/abs/path/to/your/models_root"
```

目录下期望有：

- `${OFT_MODELS_DIR}/stable-diffusion-v1-5/`
- `${OFT_MODELS_DIR}/stable-diffusion-xl-base-1.0/`
- `${OFT_MODELS_DIR}/stable-diffusion-3.5-medium/`

- **方式 B**：分别指定：
  - `SD15_MODEL_DIR` / `SDXL_MODEL_DIR` / `SD35_MODEL_DIR`

> 说明：本 repo 的 `models/` 目录默认只有 `README.md`（占位说明），你需要**自行准备权重**并通过上述环境变量指向它们；细节见 `models/README.md` 与 `BaseModel/README.md`。

3) 跑一个最小训练（以 repo 内示例 CSV 为例：列名是 `path` + `caption`）：

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

训练输出默认包含：

- `output_dir/adapter/step_000050/`：每次保存的 adapter
- `output_dir/logs/metrics.csv`：训练曲线（loss/grad_norm/耗时）
- `output_dir/validation/baseline/` + `output_dir/validation/step_000050/`：验证出图（便于对比）

## 约定：本地 SD1.5 权重

默认使用你仓库已有的本地权重目录：

- `models/stable-diffusion-v1-5`

需要包含 `model_index.json`（你的 `BaseModel/SD-1-5/README.md` 已说明）。

## 数据格式（两种任选其一）

假设 `--train_data_dir /path/to/data`

### 方式 A：图片 + 同名 txt caption（推荐最简单）

```
data/
  0001.png
  0001.txt
  0002.jpg
  0002.txt
```

### 方式 B：metadata.jsonl

```
data/
  images/
    a.png
    b.png
  metadata.jsonl
```

`metadata.jsonl` 每行：

```json
{"file_name":"images/a.png","text":"a photo of ..."}
```

## 新版输入：CSV（训练集 + 验证集）

训练脚本 `scripts/train_sd15_oft_t2i.py` 现在使用 **CSV** 作为输入，并由 CLI 指定列名。

### CSV 最小示例

假设你有：

- `train.csv`
- `val.csv`

列名默认是：

- `image`：图片路径（可以是绝对路径；相对路径默认相对 CSV 文件所在目录，也可用 `--image_root` 指定根目录）
- `text`：prompt 文本
- （可选）`negative`：negative prompt（用 `--negative_col negative` 指定）

> 本 repo 的示例数据（`Dataset/Exps/PointillismExpand/*.csv`）列名是：`path`（图片路径）与 `caption`（文本）。

`train.csv` 示例：

```csv
image,text,negative
images/0001.png,"a pointillism painting of a woman",""
images/0002.png,"a photo of an astronaut riding a horse on mars","low quality, blurry"
```

`val.csv` 示例：

```csv
image,text,negative
images/0003.png,"a watercolor painting of a city at night",""
```

## 安装依赖

（你的环境大概率已装好，`requirements.txt` 只是固定版本便于复现）

```bash
python3 -m pip install -r _OFT/requirements.txt
```

## 训练（SD1.5 + OFT）

最小示例（单卡）：

```bash
accelerate launch _OFT/scripts/train_sd15_oft_t2i.py \
  --model_dir models/stable-diffusion-v1-5 \
  --train_csv Dataset/your_dataset/train.csv \
  --val_csv Dataset/your_dataset/val.csv \
  --image_col image --text_col text \
  --output_dir _OFT/outputs/sd15_oft_run1 \
  --resolution 512 \
  --train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1e-4 \
  --max_train_steps 1000 \
  --save_steps 200
```

多 GPU（数据并行）示例（4 卡）：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --multi_gpu --num_processes 4 \
  _OFT/scripts/train_sd15_oft_t2i.py \
  --model_dir models/stable-diffusion-v1-5 \
  --train_csv Dataset/your_dataset/train.csv \
  --val_csv Dataset/your_dataset/val.csv \
  --image_col image --text_col text \
  --output_dir _OFT/outputs/sd15_oft_run1_mp \
  --resolution 512 --train_batch_size 1 --gradient_accumulation_steps 4 \
  --max_train_steps 1000 --save_steps 200
```

常用 OFT 参数：

- `--oft_block_size`：默认 32（需可整除线性层 in_features，SD1.5 通常没问题）
- `--use_cayley_neumann`：默认开启（更快，但近似）
- `--module_dropout`：默认 0（可尝试 0.1）
- `--target_modules`：默认注入到注意力投影 `to_q,to_k,to_v,to_out.0`

## CLI 参数说明（scripts）

### `scripts/train_sd15_oft_t2i.py`

启动方式：建议用 `accelerate launch`（脚本内部也有 `--mixed_precision` 参数，见下）。

- **模型/数据**
  - **`--model_dir`**: SD1.5 diffusers 权重目录；默认 `models/stable-diffusion-v1-5`（也可用 `SD15_MODEL_DIR` / `OFT_MODELS_DIR` 覆盖）
  - **`--output_dir`**: 输出目录；必填（会写入 `run_args.json`、`global_step.txt`、`adapter/`）
  - **`--train_csv`**: 训练 CSV；必填
  - **`--val_csv`**: 验证 CSV；必填（用于“每次保存 adapter 生成验证图”）
  - **`--csv_delimiter`**: CSV 分隔符；默认 `,`
  - **`--image_root`**: 可选；当 CSV 里图片路径是相对路径时，用它作为根目录（不传则相对 CSV 所在目录）
  - **`--image_col`**: 图片路径列名；默认 `"image"`
  - **`--text_col`**: prompt 列名；默认 `"text"`
  - **`--negative_col`**: negative prompt 列名；默认 `None`（不使用）
- **数据增强/预处理**
  - **`--resolution`**: 训练分辨率；默认 `512`
  - **`--center_crop` / `--no_center_crop`**: 是否中心裁剪到正方形；默认开启（`--center_crop`）
  - **`--random_flip` / `--no_random_flip`**: 是否随机水平翻转；默认开启（`--random_flip`）
- **训练超参**
  - **`--train_batch_size`**: batch size；默认 `1`
  - **`--num_workers`**: DataLoader workers；默认 `4`
  - **`--gradient_accumulation_steps`**: 梯度累计步数；默认 `1`
  - **`--learning_rate`**: 学习率；默认 `1e-4`
  - **`--lr_warmup_steps`**: 学习率 warm-up 步数（按 optimizer step 计数）；默认 `0`（关闭）
  - **`--lr_warmup_ratio`**: warm-up 比例（0~1），当 `lr_warmup_steps=0` 且该参数设置时生效；例如 `0.03` 表示 `warmup_steps=int(max_train_steps*0.03)`
  - **`--adam_beta1`**: AdamW beta1；默认 `0.9`
  - **`--adam_beta2`**: AdamW beta2；默认 `0.999`
  - **`--adam_weight_decay`**: AdamW weight_decay；默认 `1e-2`
  - **`--adam_epsilon`**: AdamW eps；默认 `1e-8`
  - **`--max_grad_norm`**: 梯度裁剪阈值；默认 `1.0`
  - **`--max_train_steps`**: 最大训练步数；默认 `1000`
  - **`--save_steps`**: 每 N step 保存一次 adapter；默认 `200`
  - **`--logging_steps`**: 每 N step 更新一次进度条 loss；默认 `20`
  - **`--seed`**: 随机种子；默认 `42`
  - **`--mixed_precision`**: 脚本内部 `Accelerator(..., mixed_precision=...)` 的混精；默认 `"fp16"`；可选：`"no" | "fp16" | "bf16"`
  - **`--ddp_timeout_sec`**: 多卡 DDP/NCCL process group 超时（秒）；默认 `1800`。当“模型加载/验证出图很慢导致初始化超时”时可调大。
- **OFT（PEFT）相关**
  - **`--oft_block_size`**: OFT block size；默认 `32`
  - **`--use_cayley_neumann` / `--no_use_cayley_neumann`**: 是否使用 Cayley-Neumann 近似（更快）；默认开启
  - **`--module_dropout`**: 乘性 dropout 概率；默认 `0.0`
  - **`--bias`**: 是否训练 bias；默认 `"none"`；可选：`"none" | "all" | "oft_only"`
  - **`--target_modules`**: 要注入 OFT 的模块后缀（逗号分隔字符串）；默认 `"to_q,to_k,to_v,to_out.0"`
- **保存时验证集出图**
  - **`--val_num_samples`**: 每次保存 adapter 时，从验证 CSV 取多少条 prompt 出图；默认 `4`；`<=0` 表示全量
  - **`--val_every_steps`**: 每隔多少个 optimizer step 跑一次验证出图；默认等于 `save_steps`；设为 `0` 可关闭验证出图（只保存 adapter）
  - **`--val_distributed`**: 验证出图是否多卡并行；默认 `"none"`；设为 `"shard"` 时按 `i % world_size == rank` 分片，每条样本 seed 仍为 `val_seed + i`（与单卡一致）
  - **`--val_steps`**: 验证出图采样步数；默认 `30`
  - **`--val_guidance`**: 验证出图 CFG；默认 `7.5`
  - **`--val_seed`**: 验证出图 base seed；默认 `1234`（第 i 条使用 `val_seed + i`）

验证输出目录结构（关键，便于对比）：

- `output_dir/validation/baseline/`：**baseline（未加载 adapter）** 的验证出图（只会生成一次，目录内会写 `_DONE` 标记）
- `output_dir/validation/step_000200/`：某次保存 step 对应的验证出图
- 每个目录内都有 `prompts.jsonl`（记录 prompt/seed 等元信息；多卡 shard 时会先写 `prompts_rankXX.jsonl` 再合并）

补充说明：

- **`accelerate launch` 自身也有 `--mixed_precision`**（那是 accelerate 的 CLI 参数），本脚本用的是“脚本内部参数” `--mixed_precision` 来配置 `Accelerator`。如果你想统一管理，可以只用脚本参数即可。
  - 本脚本会把训练指标流式写到 `output_dir/logs/metrics.csv`：包含 `loss`（原始 loss 数值）、`grad_norm_pre_clip`（裁剪前梯度范数）、时间戳、每步耗时、累计耗时。
  - 每次保存 adapter（`--save_steps`）会额外写入：`output_dir/validation/step_xxxxxx/` 下的生成图与 `prompts.jsonl`。

### `scripts/infer_sd15_oft.py`

- **`--model_dir`**: base SD1.5 目录；默认 `models/stable-diffusion-v1-5`
- **`--adapter_dir`**: 训练输出的 adapter 目录；必填（现在每个 save step 会保存到 `output_dir/adapter/step_000200/` 这种子目录；推理时请指向某个 step 子目录，目录内应有 `adapter_model.safetensors` 等）
- **`--device`**: `"cuda"|"cuda:0"|"cpu"` 等；默认自动（有 CUDA 则 `cuda` 否则 `cpu`）
- **`--torch_dtype`**: 推理 dtype；默认 `"float16"`；可选：`"float16" | "bfloat16" | "float32"`
- **`--prompt`**: 正向 prompt；必填
- **`--negative_prompt`**: 反向 prompt；默认 `None`
- **`--steps`**: 采样步数；默认 `30`
- **`--guidance`**: CFG guidance_scale；默认 `7.5`
- **`--seed`**: 随机种子；默认 `None`（不固定）
- **`--height` / `--width`**: 输出尺寸；默认 `None`（使用模型默认，通常 512）
- **`--out`**: 输出图片路径；必填
- **`--merge_and_unload`**: 是否 merge adapter 到 UNet 权重后再推理；默认关闭

### `scripts/merge_oft_sd15_pipeline.py`

作用：将 OFT adapter **merge** 到 UNet，并保存为一个不依赖 adapter 的完整 diffusers pipeline 目录。

- **`--model_dir`**: base SD1.5 目录；默认 `models/stable-diffusion-v1-5`
- **`--adapter_dir`**: adapter 目录；必填
- **`--out_dir`**: 导出的完整 pipeline 目录；必填
- **`--torch_dtype`**: 加载/保存时使用的 dtype；默认 `"float16"`；可选：`"float16" | "bfloat16" | "float32"`

## 推理（加载 adapter）

```bash
python _OFT/scripts/infer_sd15_oft.py \
  --model_dir models/stable-diffusion-v1-5 \
  --adapter_dir _OFT/outputs/sd15_oft_run1/adapter/step_001000 \
  --prompt "a photo of an astronaut riding a horse on mars" \
  --out outputs/oft_out.png
```

## 导出（可选）：merge OFT 到 UNet 并保存完整 pipeline

```bash
python _OFT/scripts/merge_oft_sd15_pipeline.py \
  --model_dir models/stable-diffusion-v1-5 \
  --adapter_dir _OFT/outputs/sd15_oft_run1/adapter/step_001000 \
  --out_dir _OFT/outputs/sd15_oft_run1/merged_pipeline
```

---

## 训练输出的可视化与分析（可选，但强烈推荐）

### 1) 网页查看 validation 对比图（baseline + steps）

训练会在 `output_dir/validation/` 下生成：

- `baseline/`
- `step_000050/`、`step_000100/` ...

可以直接用 `Visualization/` 模块的本地服务打开对比页：

```bash
python3 Visualization/server.py \
  --bind 127.0.0.1 --port 8008 \
  --static-dir Visualization \
  --allow-prefix _OFT/outputs \
  --allow-prefix Dataset
```

然后打开：

- `http://127.0.0.1:8008/_View_ALL_check_compare.html`

并在页面里填写你的 validation 根目录（建议绝对路径；相对路径在同一机器上也可用）：

- `_OFT/outputs/<run>/validation`

### 2) 绘制训练曲线（metrics.csv）

训练会写入：`_OFT/outputs/<run>/logs/metrics.csv`

`_OFT/bestcheckpoint/` 下的 notebook（`plot_metrics.ipynb` / `plot_each_metrics.ipynb`）可以用来读取并绘制 loss / grad_norm / step_seconds 等曲线。

# OFT 微调 Stable Diffusion XL Base 1.0（T2I）

本目录也提供对 **SDXL Base 1.0** 的 OFT 微调脚本（依旧：只训练 UNet 上的 OFT adapter，text encoders / VAE 全部冻结）。

模型卡参考：

- `https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0`

## 约定：本地 SDXL Base 1.0 权重

默认使用本地目录：

- `models/stable-diffusion-xl-base-1.0`

> 需要包含 `model_index.json` 且建议有 `fp16` 变体权重（脚本默认 `--variant fp16` + `--mixed_precision fp16`）。

## 训练（SDXL + OFT）

最小示例（单卡）：

```bash
accelerate launch _OFT/scripts/train_sdxl_oft_t2i.py \
  --model_dir models/stable-diffusion-xl-base-1.0 \
  --train_csv Dataset/Exps/PointillismExpand/train_captions.csv \
  --val_csv Dataset/Exps/PointillismExpand/valid_captions.csv \
  --image_col path --text_col caption \
  --output_dir _OFT/outputs/sdxl_oft_run1 \
  --resolution 1024 \
  --train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1e-4 \
  --max_train_steps 1000 \
  --save_steps 200
```

多 GPU（数据并行）示例（4 卡）：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --multi_gpu --num_processes 4 \
  _OFT/scripts/train_sdxl_oft_t2i.py \
  --model_dir models/stable-diffusion-xl-base-1.0 \
  --train_csv Dataset/Exps/PointillismExpand/train_captions.csv \
  --val_csv Dataset/Exps/PointillismExpand/valid_captions.csv \
  --image_col path --text_col caption \
  --output_dir _OFT/outputs/sdxl_oft_run1_mp \
  --resolution 1024 --train_batch_size 1 --gradient_accumulation_steps 4 \
  --max_train_steps 1000 --save_steps 200
```

说明：

- SDXL 训练会为 UNet 额外传入 `added_cond_kwargs`（`text_embeds` + `time_ids`）。
- 默认使用“固定 time_ids”：`original_size=(res,res)`, `crop=(0,0)`, `target_size=(res,res)`；如需自定义可传：
  - `--original_size H W`
  - `--crop_coords_top_left Y X`
  - `--target_size H W`

补充（常用但容易忽略的参数）：

- `--ddp_timeout_sec`：多卡 DDP 超时（秒），初始化/验证很慢时可调大
- `--val_height` / `--val_width`：仅影响“验证出图”的分辨率（训练分辨率仍由 `--resolution` 决定）

## 推理（加载 adapter）

```bash
python _OFT/scripts/infer_sdxl_oft.py \
  --model_dir models/stable-diffusion-xl-base-1.0 \
  --adapter_dir _OFT/outputs/sdxl_oft_run1/adapter/step_001000 \
  --prompt "An astronaut riding a green horse" \
  --out outputs/oft_sdxl_out.png
```

## 导出（可选）：merge OFT 到 UNet 并保存完整 pipeline

```bash
python _OFT/scripts/merge_oft_sdxl_pipeline.py \
  --model_dir models/stable-diffusion-xl-base-1.0 \
  --adapter_dir _OFT/outputs/sdxl_oft_run1/adapter/step_001000 \
  --out_dir _OFT/outputs/sdxl_oft_run1/merged_pipeline
```

---

# OFT 微调 Stable Diffusion 3.5 Medium（T2I）

本目录也提供对 **Stable Diffusion 3.5 Medium** 的 OFT 微调脚本。

模型卡参考：

- https://huggingface.co/stabilityai/stable-diffusion-3.5-medium

## 重要差异（相对 SD1.5/SDXL）

- SD3.5 Medium 在 diffusers 里核心网络是 **`transformer`**（不是 UNet）。
- 使用 **Flow Matching** 的 scheduler（`FlowMatchEulerDiscreteScheduler`），训练目标不再是“预测 epsilon”，而是学习 **velocity**（脚本里用 `target = noise - latents`，并构造 $x_\\sigma=(1-\\sigma)x_0+\\sigma\\epsilon$）。
- `encode_prompt` 需要传入 `prompt/prompt_2/prompt_3`（三套文本编码器）；脚本里默认三者都用同一个 prompt。

## 约定：本地 SD3.5 Medium 权重

默认使用本地目录：

- `models/stable-diffusion-3.5-medium`

## 训练（SD3.5 Medium + OFT）

最小示例（单卡）：

```bash
accelerate launch _OFT/scripts/train_sd3_5_medium_oft_t2i.py \
  --model_dir models/stable-diffusion-3.5-medium \
  --train_csv Dataset/Exps/PointillismExpand/train_captions.csv \
  --val_csv Dataset/Exps/PointillismExpand/valid_captions.csv \
  --image_col path --text_col caption \
  --output_dir _OFT/outputs/sd35m_oft_run1 \
  --resolution 1024 \
  --train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1e-4 \
  --max_train_steps 1000 \
  --save_steps 200 \
  --mixed_precision bf16 \
  --max_sequence_length 256
```

多 GPU（数据并行）示例（4 卡）：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --multi_gpu --num_processes 4 \
  _OFT/scripts/train_sd3_5_medium_oft_t2i.py \
  --model_dir models/stable-diffusion-3.5-medium \
  --train_csv Dataset/Exps/PointillismExpand/train_captions.csv \
  --val_csv Dataset/Exps/PointillismExpand/valid_captions.csv \
  --image_col path --text_col caption \
  --output_dir _OFT/outputs/sd35m_oft_run1_mp \
  --resolution 1024 --train_batch_size 1 --gradient_accumulation_steps 4 \
  --max_train_steps 1000 --save_steps 200 --mixed_precision bf16 --max_sequence_length 256
```

补充（SD3.5 Medium 常用参数）：

- `--ddp_timeout_sec`：多卡 DDP 超时（秒）
- `--sigma_min` / `--sigma_max`：Flow Matching 训练采样区间（默认 `0.0~1.0`）
- `--val_max_sequence_length`：验证出图时传给 `encode_prompt` 的 `max_sequence_length`（不传则沿用训练的 `--max_sequence_length`）
- `--val_height` / `--val_width`：仅影响验证出图分辨率

## 推理（加载 adapter）

```bash
python _OFT/scripts/infer_sd3_5_medium_oft.py \
  --model_dir models/stable-diffusion-3.5-medium \
  --adapter_dir _OFT/outputs/sd35m_oft_run1/adapter/step_001000 \
  --prompt "A capybara holding a sign that reads Hello World" \
  --steps 40 --guidance 4.5 \
  --max_sequence_length 256 \
  --out outputs/oft_sd35m_out.png
```

推理 dtype 说明（SD3.5）：

- `infer_sd3_5_medium_oft.py` 的 `--torch_dtype` 默认是 `auto`：
  - CUDA 且支持 bf16 → 使用 `bfloat16`
  - CUDA 但不支持 bf16 → 使用 `float16`
  - CPU → 使用 `float32`

## 导出（可选）：merge OFT 到 transformer 并保存完整 pipeline

```bash
python _OFT/scripts/merge_oft_sd3_5_medium_pipeline.py \
  --model_dir models/stable-diffusion-3.5-medium \
  --adapter_dir _OFT/outputs/sd35m_oft_run1/adapter/step_001000 \
  --out_dir _OFT/outputs/sd35m_oft_run1/merged_pipeline
```

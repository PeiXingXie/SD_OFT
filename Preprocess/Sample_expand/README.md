# Sample_expand - 外部文生图 API 调用框架

目标：在 `Preprocess/Sample_expand` 提供一个**可扩展的外部 API 调用框架**，用于按文本调用文生图模型，并拿到图片结果（`url` + `data:image/png;base64,...`）。

本实现参考 `@Project/ImageModelServer` 的外部 API 调用方式（HMAC 签名 + `POST /image/v1/generations`），**不使用任何本地模型**。

## 约束（已在代码中强制）

- **endpoint**：强制使用 URL 后缀 **`/image/v1/generations`**
- **size**：强制 **`1024x1024`**
- **gpt-image-1**：强制 **`quality=high`**

## 目录结构

- `sample_expand/`: 核心库
  - `config.py`: YAML 配置加载（支持 `${ENV:VAR}`）
  - `andes_auth.py`: Andes Gateway HMAC 签名
  - `image_client.py`: 调用 `/image/v1/generations` 并解析结果、下载图片转 base64
  - `factory.py`: client 工厂
- `configs/`: 示例配置（`gpt-image-1` / `gemini-3-pro-image-preview`）
- `scripts/generate_image.py`: 最小可运行 CLI（输入 prompt -> 输出 png）

## 安装依赖

在 `Preprocess/Sample_expand` 目录下：

```bash
pip install -r requirements.txt
```

## 配置环境变量（推荐）

配置文件默认从环境变量读取：

- `ANDES_APP_ID`
- `ANDES_SECRET_KEY`
- `ANDES_BASE_URL`（可选；用于配置网关 base_url）

例如：

```bash
export ANDES_APP_ID="your_app_id"
export ANDES_SECRET_KEY="your_secret_key"
```

## 直接使用 CLI 参数传入鉴权（可选）

你也可以不设置环境变量，直接在命令行传入（会覆盖 YAML/ENV）：

- `--api-app-id`
- `--api-secret-key`

## 使用方法

### 1) 使用 gpt-image-1（quality=high）

在 `Preprocess/Sample_expand` 目录下执行：

```bash
python scripts/generate_image.py \
  --config configs/gpt-image-1.yaml \
  --api-app-id "your_app_id" \
  --api-secret-key "your_secret_key" \
  --api-verbose \
  --prompt "a cute capybara wearing a red scarf" \
  --out /tmp/out_gpt.png \
  --dump-json /tmp/out_gpt.json
```

### 2) 使用 gemini-3-pro-image-preview

```bash
python scripts/generate_image.py \
  --config configs/gemini-3-pro-image-preview.yaml \
  --prompt "a small wooden cabin in a snowy forest at sunset" \
  --out /tmp/out_gemini.png
```

### 3) 批量处理 CSV（id + caption）

从 CSV 读取 `id`、`caption` 两列进行文生图，并自动跳过：

- `caption` 为空
- `caption` 为 `"error"`（大小写不敏感）
- 存在 `__error` 列且该列非空（表示上游/上一步已标记错误）

示例（参考 `captions_pointillism.csv` 的字段风格）：

```bash
python scripts/batch_generate_from_csv.py \
  --config configs/gpt-image-1.yaml \
  --api-app-id "your_app_id" \
  --api-secret-key "your_secret_key" \
  --log-every 1 \
  --api-verbose \
  --csv Dataset/your_dataset/captions.csv \
  --out-dir /tmp/gen_images \
  --ext .png \
  --report /tmp/gen_report.csv
```

常用参数：

- `--start/--end`：按 **CSV 行序（0-based）** 截取处理范围（`start <= idx < end`，`--end -1` 表示到结尾）
- `--range-on`：`raw|valid`，指定 `--start/--end` 的计数口径；`raw`=按 CSV 原始行号，`valid`=按通过 skip 规则后的有效样本序号
- `--log-every`：每 N 条（在范围内的样本）输出一条状态，默认 `1`（每条都输出）
- `--quiet`：仅输出最后汇总（不输出每条样本状态）
- `--api-verbose`：输出每次 API 请求的 attempt/retry/sleep 等细节（便于排查）
- `--id-col`：默认 `id`
- `--caption-col`：默认 `caption`
- `--error-col`：默认 `__error`（若列存在且非空则跳过）
- `--overwrite`：覆盖已存在图片
- `--sleep-sec`：每次请求后 sleep（便于控频）
- `--report`：输出 report CSV（**流式写入**，每条样本处理后立即落盘；程序中断也会保留已写入部分）

## 返回结果说明

核心调用返回 `ImageGenerationResult`（见 `sample_expand/image_client.py`）：

- `ok`: 是否成功
- `image_url`: 上游返回的图片 URL（通常在 `data.results[0].url`）
- `image_base64`: 下载后转换的 `data:image/png;base64,...`
- `response`: 上游业务 JSON（已解析）
- `raw_http`: 原始 HTTP 摘要（状态码/headers/text 截断），便于排查


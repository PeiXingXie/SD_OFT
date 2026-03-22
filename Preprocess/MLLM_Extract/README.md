## MLLM_Extract：图片 MLLM 提取模型调用框架

参考 `Project/ReasonEdit/Judger/V5` 的组织方式与调用约定（`PYTHONPATH=src python3 -m ...`），本目录提供一个用于**批量图片 -> MLLM** 的最小框架：

- `src/mllm_extract/`: 核心 Python 包（API client / 多模态 messages / IO / CLI）
- `configs/`: YAML 配置（API、prompt、输入输出等）

### 安装

```bash
cd Preprocess/MLLM_Extract
python3 -m pip install -r requirements.txt
```

### 运行（目录模式：扫描图片）

```bash
cd Preprocess/MLLM_Extract
export PYTHONPATH=src

python3 -m mllm_extract.cli.run_images \
  --config configs/config_mllm.yaml \
  --threads 16 \
  --prompt-name pointillism \
  --api-model gpt-5.2 \
  --output-csv outputs/captions.csv \
  --output-jsonl outputs/captions.jsonl
```

输出默认写入 `configs/config_mllm.yaml` 中的 `output.output_csv`（可断点续跑）。

### 运行（CSV 模式：从 CSV 读取图片路径列）

把 config 里的 `input.mode` 改为 `csv` 并设置 `input.input_csv` + `input.image_path_column`。

补充（CSV 列覆盖与图片读取）：

- **`--id-col <col>`**：指定 CSV 的 id 列（默认来自 `input.sample_id_column`；否则回退为 `path` 的文件名 stem）
- **`--path-col <col>`**：指定 CSV 的图片路径列（默认来自 `input.image_path_column`；可不提供）
- **`--caption-col <col>`**：指定 CSV 的 caption 列（仅评估任务需要；默认来自 `input.caption_column` 或 `output.caption_column`）

图片读取支持两种方式（二选一或混用）：

- **方式 A：CSV 提供 path**：通过 `--path-col`（或配置 `input.image_path_column`）读取图片
- **方式 B：CSV 不提供 path，用 input-dir 按 id 匹配**：当 CSV 的 `path` 为空时，会扫描 `--input-dir`（或 `input.image_dir`），用 **文件名 stem == id** 自动补全 `path` 后读取图片

### Prompt 选择

- `--prompt-name common|pointillism|used`: 使用仓库内置 prompt（来自 `prompt/ForCommon.py` / `prompt/ForPointillism.py` / `prompt_used.py`）
- `--prompt-name semantic_adherence`: 评估任务 - 语义一致性（来自 `prompt/SemanticAdherence.py`，要求输出 JSON：`{"score":1..5,"short_reason":"..."}`）
- `--prompt-name structural_plausibility`: 评估任务 - 结构合理性（来自 `prompt/StructuralPlausibility.py`，要求输出 JSON：`{"score":1..5,"short_reason":"..."}`）
- `--prompt-file /ABS/PATH/to/prompt.py`: 使用自定义 prompt 文件（需定义变量 `system_prompt`、`user_prompt`）

### Task 选择（新增，不影响原 caption 逻辑）

- **默认任务**：`caption`（保持原逻辑：批量图片 -> caption）
- **评估任务**：
  - `semantic_adherence`：根据 `{caption}` 对图片打分
  - `structural_plausibility`：只看结构质量打分（也会携带 `{caption}` 作为上下文）

通过 CLI 指定：

- `--task caption|semantic_adherence|structural_plausibility`

说明：

- 当你不显式传 `--task` 时，默认仍是 **caption**；但若你把 `--prompt-name` 设为 `semantic_adherence` / `structural_plausibility`，会自动推断成对应评估任务。

### API 覆盖（常用）

- `--api-base-url` / `--api-app-id` / `--api-model`
- `--api-secret-key` 或 `--api-secret-key-env ANDES_SK`（推荐用 env，不要把 SK 写进 yaml）

### CLI 参数说明（含可接受值）

#### 基础与并发

- **`--config`**: YAML 配置路径（默认：`configs/config_mllm.yaml`）
- **`--threads`**: 并发线程数，`int >= 1`（默认 `1`）
- **`--log-interval-s`**: 并发状态打印间隔秒数，`int >= 1`（默认 `10`）
- **`--retries`**: 每张图失败重试次数，`int >= 0`（默认 `3`）
- **`--no-resume`**: 禁用断点续跑
  - **默认策略（启用续算）**：仅当 **caption 非空且 error 为空** 才跳过；否则（caption 为空/NaN，或 error 非空）会自动重试并覆盖落盘结果
  - **例外**：遇到网关 **内容安全拦截**（例如 `api_code=-30002`）会被标记为 `non_retriable:...`，后续续算会自动跳过，避免无限重试
  - **评估任务的续算策略**：当 `score` 非空且 error 为空才跳过（否则会重试）

#### 输入（覆盖 YAML 的 `input.*`）

- **`--input-dir`**: 输入图片目录（使用目录模式）
- **`--input-csv`**: 输入 CSV（会自动切到 CSV 模式）
- **`--id-col`** / **`--path-col`** / **`--caption-col`**: 仅 CSV 模式下生效（用于覆盖列名，详见上文）

#### 输出（覆盖 YAML 的 `output.*`）

- **`--output-csv`**: 输出 CSV 路径（例如 `outputs/captions.csv`）
- **`--output-jsonl`**: 输出 JSONL 路径
  - **可接受值**：任意路径；传空字符串 `""` 可禁用 JSONL
- **`--output-dir`**: 输出目录前缀
  - **规则**：当你**不显式**传 `--output-csv/--output-jsonl` 时，会把默认文件名（或 YAML 里的文件名的 basename）放到该目录下

评估任务输出列（可在 YAML 的 `output.*` 里覆盖列名）：

- `output.score_column`（默认 `score`）
- `output.short_reason_column`（默认 `short_reason`）

#### Prompt（覆盖 YAML 的 `prompt.*`）

- **`--prompt-name`**: 选择内置 prompt
  - **可接受值**：`common` | `pointillism` | `used`
- **`--prompt-file`**: 指定自定义 prompt `.py` 文件
  - **要求**：文件内必须定义字符串变量 `system_prompt`、`user_prompt`

#### API（覆盖 YAML 的 `api.*`）

- **`--api-base-url`**: 网关 URL（例如 Andes gateway）
- **`--api-app-id`**: app_id
- **`--api-model`**: 模型名（字符串，例如 `gpt-5` / `gpt-5.2` 等，取决于你的网关支持）
- **`--api-secret-key`**: secret_key（不推荐写死）
- **`--api-secret-key-env`**: 从环境变量读取 secret_key 的变量名（默认 `ANDES_SK`）
  - **规则**：当 `api.secret_key` 为空且未传 `--api-secret-key` 时，会读取该 env
- **`--api-timeout-sec`**: 超时秒数，`int`
- **`--api-retry-on-code`**: 触发网关重试的 code，`int`（默认 `-20001`）
- **`--api-retry-sleep-sec`**: 网关重试前 sleep 秒数，`int`（默认 `60`）
- **`--api-use-reasoning`**: 推理开关
  - **可接受值**：`true/false`（也接受 `1/0`, `yes/no`, `on/off`，大小写不敏感）
- **`--api-temperature`**: `float`
- **`--api-top-p`**: `float`

#### 运行切片与图片细节（覆盖 YAML 的 `run.*`）

- **`--start`**: 起始 index（`int`，默认来自 YAML）
- **`--end`**: 结束 index（`int`；不传表示直到末尾）
- **`--image-detail`**: 图片 detail
  - **可接受值**：`low` | `high` | `auto`
- **`--non-retriable-api-codes`**: 不可重试的 `api_code` 列表（逗号分隔）
  - **可接受值**：例如 `-30002,-30003`
  - **默认值**：来自 `config.run.non_retriable_api_codes`（默认包含 `-30002`）

### 示例：评估任务（Semantic Adherence）

```bash
cd Preprocess/MLLM_Extract
export PYTHONPATH=src

python3 -m mllm_extract.cli.run_images \
  --task semantic_adherence \
  --prompt-name semantic_adherence \
  --input-csv /ABS/PATH/input.csv \
  --id-col id \
  --path-col path \
  --caption-col caption \
  --threads 16 \
  --output-csv /ABS/PATH/semantic_scores.csv \
  --output-jsonl ""
```

### 示例：评估任务（Structural Plausibility，CSV 无 path，按 id 从 input-dir 匹配图片）

```bash
python3 -m mllm_extract.cli.run_images \
  --task structural_plausibility \
  --prompt-name structural_plausibility \
  --input-csv /ABS/PATH/input.csv \
  --input-dir /ABS/PATH/images_dir \
  --id-col id \
  --caption-col caption \
  --threads 16 \
  --output-csv /ABS/PATH/struct_scores.csv \
  --output-jsonl ""
```

### 覆盖优先级（建议）

- **CLI 参数** > **YAML 配置** >（API secret）**环境变量** > 默认值

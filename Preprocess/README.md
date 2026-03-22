# Preprocess（数据/图片预处理工具集）

本目录收纳了训练/评估前常用的预处理脚本，按用途大致分为四类：

- **CSV 处理**：`easy_act_csv.py`（基于 pandas，支持 `chunksize` 分块，适合大 CSV）
- **图片批量缩放**：`Resize/`
- **外部文生图批量生成**：`Sample_expand/`
- **图片 + MLLM 批量提取/评估**：`MLLM_Extract/`

如果你只是想快速上手：先看下面的“常见任务”与“快速开始”，需要更细的参数说明再进入各子模块 README。

---

## 目录结构

- `easy_act_csv.py`：CSV 处理小工具（列增删改、列合并、路径替换、pipeline 串行多步）
- `Resize/`：批量缩放图片（可递归、镜像目录结构输出）
  - 详见：`Resize/README.md`
- `Sample_expand/`：外部文生图 API 调用框架（从 prompt 或 CSV 批量生成图片）
  - 详见：`Sample_expand/README.md`
- `MLLM_Extract/`：批量图片 -> MLLM（caption / 评估打分 / 分类等）
  - 详见：`MLLM_Extract/README.md`

---

## 常见任务 → 用哪个脚本

### 1) 改 CSV 列（增列/合并/抽取/删除/路径前缀替换）

用：`easy_act_csv.py`

典型场景：

- **修复挂载前缀**：把 `image_path` 的旧前缀替换为新前缀
- **生成 instruction/prompt**：常量列 + mergecols 合并成最终 `instruction`
- **抽取训练所需最小列**：只保留 `image_path` / `instruction` 等
- **多步流水线**：一次命令串多个操作（`pipeline`）

> `easy_act_csv.py` 的设计约束：**只对“新增列的字符串内容”强制包裹双引号**，其它列按 CSV 的“最小必要引号”写出，避免对全表过度加引号导致下游误解析。

### 2) 批量缩放图片（例如 1024→512）

用：`Resize/resize_images.py`（需要 Pillow）

详见：`Resize/README.md`

### 3) 按 CSV 批量文生图，生成图片文件

用：`Sample_expand/scripts/batch_generate_from_csv.py`

详见：`Sample_expand/README.md`

### 4) 批量图片 -> MLLM 提取/评估（caption / semantic_adherence / structural_plausibility / style_match / category_classify）

用：`MLLM_Extract/src/mllm_extract/cli/run_images.py`（按 YAML 配置运行）

详见：`MLLM_Extract/README.md`

---

## 快速开始

以下命令示例建议在**仓库根目录**执行（路径按你的 repo 实际位置调整）。

### A) `easy_act_csv.py`（CSV 处理）

#### 安装依赖

`easy_act_csv.py` 依赖 `pandas`：

```bash
python3 -m pip install pandas
```

#### 常用命令

- **抽取列**：

```bash
python3 Preprocess/easy_act_csv.py -i in.csv -o out.csv extractcols --cols image_path instruction
```

- **路径前缀替换**（修复挂载前缀）：

```bash
python3 Preprocess/easy_act_csv.py -i in.csv -o out.csv replacepath --col image_path --old "OLD_PREFIX/" --new "NEW_PREFIX/"
```

- **pipeline 串多步**（对同一 CSV 连续加工）：

```bash
python3 Preprocess/easy_act_csv.py \
  -i data.csv \
  --inplace \
  --if-exists replace \
  pipeline \
    addcol --new-col preinst --value "Edit the image to show the impact of this factor:" \
    mergecols --new-col instruction --cols preinst driving_factor_text --sep " " --drop \
    replacepath --col image_path --old "OLD_PREFIX/" --new "NEW_PREFIX/"
```

> 更多子命令与示例直接看 `easy_act_csv.py` 文件头部 docstring（非常全）。

### B) `Resize/`（批量缩放）

推荐先 dry-run：

```bash
python Preprocess/Resize/resize_images.py --input_dir /path/in --output_dir /path/out --dry_run --verbose
```

更多用法：见 `Resize/README.md`。

### C) `Sample_expand/`（外部文生图 API 批量生成）

从 CSV 批量生成图片（`id + caption`）：

```bash
python Preprocess/Sample_expand/scripts/batch_generate_from_csv.py \
  --config Preprocess/Sample_expand/configs/gpt-image-1.yaml \
  --api-app-id "YOUR_APP_ID" \
  --api-secret-key "YOUR_SECRET_KEY" \
  --csv Dataset/your_dataset/captions.csv \
  --out-dir /tmp/gen_images \
  --report /tmp/gen_report.csv
```

配置/环境变量/skip 规则：见 `Sample_expand/README.md`。

### D) `MLLM_Extract/`（批量图片 -> MLLM）

目录模式（扫描图片）：

```bash
cd Preprocess/MLLM_Extract
python3 -m pip install -r requirements.txt
export PYTHONPATH=src

python3 -m mllm_extract.cli.run_images \
  --config configs/config_mllm.yaml \
  --threads 16 \
  --prompt-name pointillism \
  --api-model gpt-5.2 \
  --output-csv outputs/captions.csv \
  --output-jsonl outputs/captions.jsonl
```

CSV 模式、评估任务（semantic/structural/style_match/category）等：见 `MLLM_Extract/README.md`。

---

## 注意事项

- **路径/挂载**：这些脚本通常处理的是“训练环境可见的绝对路径”。如果你在容器/服务器环境跑，注意 CSV 里的路径是否与实际挂载一致（必要时用 `easy_act_csv.py replacepath` 修复）。
- **大 CSV**：优先用 `easy_act_csv.py`（支持分块处理，避免一次性把大文件读入内存）。
- **依赖隔离**：`Resize/`、`Sample_expand/`、`MLLM_Extract/` 各自可能有额外依赖（Pillow、requests/yaml 等），建议按各自 README 安装对应 `requirements.txt`。


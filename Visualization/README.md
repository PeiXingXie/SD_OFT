# Visualization（本地 HTML 可视化 + 轻量文件 API）

本目录提供一组**本地可视化 HTML 页面**与一个**轻量 HTTP 服务**（`server.py`），用于在浏览器中查看：

- 图片目录（按目录/子目录分组、筛选、搜索）
- Validation 输出（`baseline` vs `step_XXXXXX` 多 step 对比）
- CSV 结果表（图片 + caption/label，并可将选中的图片写入同目录 `_selected.csv`）

后端服务通过同源 API 提供**按绝对路径读取文件/列目录**的能力，并用 `--allow-prefix` 做路径白名单限制。

---

## 快速开始

### 1) 启动本地服务

在本目录启动（仅依赖 Python 标准库）：

```bash
python3 /mnt/workspace/xiepeixing/Project/_TestProject/OFT_for_SD/Visualization/server.py \
  --bind 127.0.0.1 \
  --port 8008 \
  --static-dir /mnt/workspace/xiepeixing/Project/_TestProject/OFT_for_SD/Visualization \
  --allow-prefix /mnt/workspace/xiepeixing/Project/_TestProject/OFT_for_SD
```

说明：

- `--static-dir`：用于托管本目录的 HTML 静态文件（默认就是本目录）。
- `--allow-prefix`：**必须提供**；不提供时服务会拒绝所有文件访问（安全默认）。
  - 可多次传入，例如同时允许读取多个根目录。

> 如需局域网访问可用 `--bind 0.0.0.0`，但务必把 `--allow-prefix` 限制到最小范围，且不要暴露到公网。

### 2) 打开页面

启动后在浏览器打开：

- 目录图库：`http://127.0.0.1:8008/_View_ALL_check_dir.html`
- Validation 对比：`http://127.0.0.1:8008/_View_ALL_check_compare.html`
- CSV 图库（带 `_selected.csv` 追加写入）：`http://127.0.0.1:8008/_View_ALL_check_result.html`

---

## 页面说明

### `_View_ALL_check_dir.html`（按目录看图）

- **输入**：`图片目录（路径）`（建议填**绝对路径**）
- **读取方式**：调用 `/api/ls?path=...` 列出图片文件，再用 `/api/bin?path=...` 加载图片
- **常用选项**：
  - `递归子目录`：是否递归列出所有子目录图片
  - `最多`：最多读取多少张（对应 `limit`，上限 200000）
  - `label/caption`：根据目录结构/文件名展示标签与说明

### `_View_ALL_check_compare.html`（Validation：baseline + steps 对比）

- **输入**：validation 根目录（建议绝对路径），例如 `_OFT/outputs/.../validation`
- **目录结构约定**（关键）：
  - 根目录下包含若干 step 目录：`baseline/` 与 `step_000200/`、`step_000400/` ...（`step_` + 数字）
  - 每个 step 目录下应有**同名样本相对路径**的图片（用于对齐对比）
- **实现逻辑**：后端递归列出全部图片后，按 `baseline`/`step_XXXXXX` 作为 step key，把其余相对路径作为 sample key 进行对齐展示。

### `_View_ALL_check_result.html`（按 CSV 结果表看图 + 选择写入）

- **输入**：
  - `CSV 路径`（建议绝对路径；通过 `/api/csv?path=...` 读取）
  - `图片根目录（绝对路径）`：为空时可自动取 `CSV 同目录/images`（可勾选“自动”）
  - 选择 `图片列 / caption列 / label列`
- **写入能力**：
  - 页面里可点击“加入 `_selected.csv`”，会将**图片文件名**追加写入到“该图片所在目录”的 `_selected.csv`
  - 后端只允许写入 **以 `_selected.csv` 结尾的文件**，且必须在 `--allow-prefix` 范围内

---

## 后端 API（`server.py`）

所有 API 都要求 `path` 为**绝对路径**，且 realpath 必须落在 `--allow-prefix` 白名单内，否则返回 403。

- **GET `/api/text?path=ABS_PATH`**：读取 UTF-8 文本
- **GET `/api/csv?path=ABS_PATH`**：读取 CSV 并返回 JSON（避免前端解析大字段/引号字段问题）
- **GET `/api/ls?path=ABS_DIR&recursive=0|1&limit=N`**：列出目录下图片文件路径（JSON）
- **GET `/api/bin?path=ABS_PATH`**：读取二进制（图片等），自动设置 `Content-Type`
- **POST `/api/append?path=ABS_PATH`**：追加一行到文件
  - 仅允许写入 `*_selected.csv`
  - body：`application/json`，形如 `{"line":"xxx"}`

---

## 安全提示（重要）

该服务的设计目标是“本机/受控环境下快速查看本地文件”。因为它支持按绝对路径读取文件：

- **务必使用 `--allow-prefix` 限制可读路径**（默认不允许任何路径）。
- **不要暴露到公网**（不要把 `--bind 0.0.0.0` + 端口映射到公网）。
- 写入虽然被限制为 `*_selected.csv`，仍建议把白名单范围尽量缩小。

---

## 常见问题

- **页面提示 403 forbidden**
  - 说明目标文件不在 `--allow-prefix` 白名单下；请把需要访问的根目录加入 `--allow-prefix`（可多次传入）。
- **页面提示 404 not found**
  - 检查你填的路径是否真实存在（注意是否是容器内路径/宿主机路径不一致）。
- **CSV 内容复杂（带引号/大字段）解析异常**
  - 页面优先使用 `/api/csv`（后端解析），比纯前端 CSV 解析更稳；确保你是通过服务打开页面（同源），而不是直接 `file://` 打开。


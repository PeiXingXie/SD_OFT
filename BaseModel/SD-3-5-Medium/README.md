# SD-3-5-Medium (Diffusers) 调用说明

本目录提供对本地模型 `models/stable-diffusion-3.5-medium` 的最小封装与可运行 demo。

## 模型路径（默认）

- 默认模型目录：`models/stable-diffusion-3.5-medium`
- 也可通过环境变量覆盖：
  - `SD35_MODEL_DIR`: 直接指定模型目录
  - `OFT_MODELS_DIR`: 指定 models 根目录（脚本会拼上 `stable-diffusion-3.5-medium`）
- 该目录需要包含 `model_index.json`（你当前的本地目录已包含）。

## 依赖

- **PyTorch**（建议 CUDA 版）
- **diffusers**（以及其常见依赖：`transformers`、`accelerate`、`safetensors` 等）

说明：本封装只在真正推理时才 import `torch/diffusers`，因此 `--help` 等不会触发下载/加载权重。

## 方式 1：直接运行 demo（推荐）

```bash
python BaseModel/SD-3-5-Medium/demo_sd3_5_medium.py \
  --prompt "A capybara holding a sign that reads Hello World, " \
  --steps 40 \
  --guidance 4.5 \
  --out BaseModel/SD-3-5-Medium/test/capybara.png
```

可选参数：

- `--model_dir`: 指定本地模型目录
- `--device`: 例如 `cuda` / `cuda:0` / `cpu`（默认自动：有 CUDA 用 CUDA）
- `--torch_dtype`: `auto|bfloat16|float16|float32`（默认 `auto`）
- `--negative_prompt`、`--seed`

## 方式 2：以“你给的参考代码”风格调用

在任意 Python 脚本中，确保能 import 到本目录下的 `sd3_5_medium.py`（例如把脚本放在同目录，或手动加 `sys.path`），然后：

```python
from sd3_5_medium import SD35Medium

model = SD35Medium(
    model_dir="models/stable-diffusion-3.5-medium",
    device="cuda",
    torch_dtype="bfloat16",
)

image = model.generate(
    "A capybara holding a sign that reads Hello World",
    num_inference_steps=40,
    guidance_scale=4.5,
)
image.save("capybara.png")
```

## 避免“每次调用都重新加载权重”（长进程复用）

如果你每次都是 `python demo_sd3_5_medium.py ...` 启动新进程，那么权重一定会重复加载。
想要复用加载好的权重，需要在**同一个 Python 进程**里复用同一个实例。

本目录提供了一个带缓存的单例工厂 `get_sd35_medium()`：

```python
from sd3_5_medium import get_sd35_medium

# 第一次会加载权重；同进程内后续再次调用会直接复用，不会重复加载
model = get_sd35_medium(device="cuda", torch_dtype="bfloat16")

prompts = [
    "A capybara holding a sign that reads Hello World",
    "A cinematic photo of a capybara astronaut, ultra detailed",
]

for i, p in enumerate(prompts):
    img = model.generate(p, num_inference_steps=40, guidance_scale=4.5)
    img.save(f"out_{i}.png")
```

## 一次加载，批量生成（从文件读取多条 prompt）

如果你想“命令行方式多次调用但只加载一次”，可以用批量脚本（同一进程内循环）：

```bash
python BaseModel/SD-3-5-Medium/batch_run_sd3_5_medium.py \
  --prompts_file prompts.txt \
  --out_dir outputs/sd35_out
```

其中 `prompts.txt` 为“每行一个 prompt”（空行会忽略）。

## 保持模型实例常驻，并支持“多次命令调用”（Daemon 模式）

如果你希望像下面这样反复在 shell 里调用，但**不想每次都重新加载权重**：

- `python xxx.py --prompt ... --out ...`
- `python xxx.py --prompt ... --out ...`

那必须启动一个**常驻进程**来持有模型实例。这里提供了一个本地 TCP daemon（JSON Lines 协议）：

### 1) 启动服务端（只启动一次，会加载权重）

```bash
python BaseModel/SD-3-5-Medium/sd3_5_medium_daemon_server.py \
  --host 127.0.0.1 --port 6319 \
  --device cuda --torch_dtype bfloat16
```

把它放在一个长期存在的终端（或 `tmux`/后台进程）里运行即可。

### 2) 客户端多次命令调用（不会重复加载权重）

```bash
python BaseModel/SD-3-5-Medium/sd3_5_medium_daemon_client.py \
  --host 127.0.0.1 --port 6319 \
  --prompt "A capybara holding a sign that reads Hello World, in Impressionism Style." \
  --out BaseModel/SD-3-5-Medium/test/capybara-Impressionism.png

python BaseModel/SD-3-5-Medium/sd3_5_medium_daemon_client.py \
  --host 127.0.0.1 --port 6319 \
  --prompt "A capybara holding a sign that reads Hello World, in Baroque Style." \
  --out BaseModel/SD-3-5-Medium/test/capybara-Baroque.png

python BaseModel/SD-3-5-Medium/sd3_5_medium_daemon_client.py \
  --host 127.0.0.1 --port 6319 \
  --prompt "A capybara holding a sign that reads Hello World, in Rococo Style." \
  --out BaseModel/SD-3-5-Medium/test/capybara-Rococo.png

python BaseModel/SD-3-5-Medium/sd3_5_medium_daemon_client.py \
  --host 127.0.0.1 --port 6319 \
  --prompt "A capybara holding a sign that reads Hello World, in Ukiyo-e Style." \
  --out BaseModel/SD-3-5-Medium/test/capybara-Ukiyo-e.png
```

健康检查：

```bash
python BaseModel/SD-3-5-Medium/sd3_5_medium_daemon_client.py --ping
```

## 反复调用常驻实例以占用 GPU（Keep-alive / Burn）

如果你的目标是“让 GPU 持续有活干”，可以用循环脚本反复向 daemon 发请求（同一个输出文件会被覆盖，避免磁盘膨胀）：

```bash
python BaseModel/SD-3-5-Medium/gpu_burn_sd3_5_medium.py \
  --host 127.0.0.1 --port 6319 \
  --repeat 1000000 \
  --steps 30 \
  --out BaseModel/SD-3-5-Medium/test/gpu_burn.png
```

## 文件说明

- `sd3_5_medium.py`: `SD35Medium` 封装（`load()` / `generate()` / `generate_to_file()`）
- `demo_sd3_5_medium.py`: 命令行 demo
- `batch_run_sd3_5_medium.py`: 批量生成（一次加载，多次生成）
- `sd3_5_medium_daemon_server.py`: 常驻服务端（一次加载，多次命令调用）
- `sd3_5_medium_daemon_client.py`: 客户端命令（向常驻服务端发请求）
- `gpu_burn_sd3_5_medium.py`: 循环调用 daemon（持续占用 GPU）


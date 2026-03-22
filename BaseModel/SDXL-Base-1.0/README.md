# SDXL Base 1.0（Diffusers）调用说明

本目录提供对本地模型 `models/stable-diffusion-xl-base-1.0` 的调用封装与 demo。

> 仅使用 **base** 模型，不使用 refiner（与你的要求一致）。

## 模型路径（默认）

- 默认模型目录：`models/stable-diffusion-xl-base-1.0`
- 也可通过环境变量覆盖：
  - `SDXL_MODEL_DIR`: 直接指定模型目录
  - `OFT_MODELS_DIR`: 指定 models 根目录（脚本会拼上 `stable-diffusion-xl-base-1.0`）
- 该目录需要包含 `model_index.json`（你当前的本地目录已包含）。

## 依赖

- **PyTorch**（建议 CUDA 版）
- **diffusers**（及 `transformers` / `accelerate` / `safetensors` 等）

## 方式 1：直接运行 demo

```bash
python BaseModel/SDXL-Base-1.0/demo_sdxl_base.py \
  --prompt "An astronaut riding a green horse" \
  --out outputs/sdxl.png
```

常用参数：

- `--device`: `cuda` / `cuda:0` / `cpu`（默认自动）
- `--torch_dtype`: `auto|float16|bfloat16|float32`（默认 `auto`）
- `--variant`: 默认 `fp16`（当 dtype=fp16 时会生效）
- `--steps` / `--guidance` / `--seed`
- `--height` / `--width`

## 方式 2：以你的参考代码风格调用（base-only）

等价于：

```python
from diffusers import DiffusionPipeline
import torch

pipe = DiffusionPipeline.from_pretrained(
    "models/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
).to("cuda")

image = pipe(prompt="An astronaut riding a green horse").images[0]
image.save("sdxl.png")
```

本目录封装版（更方便复用/缓存）：

```python
from sdxl_base import SDXLBase10

model = SDXLBase10(device="cuda", torch_dtype="float16", variant="fp16")
img = model.generate("An astronaut riding a green horse", num_inference_steps=40, guidance_scale=7.5)
img.save("sdxl.png")
```

## 避免重复加载（同进程复用）

```python
from sdxl_base import get_sdxl_base

model = get_sdxl_base(device="cuda", torch_dtype="float16", variant="fp16")  # 首次加载
img1 = model.generate("An astronaut riding a green horse")
img2 = model.generate("A cinematic photo of a green horse in space")
```

## Daemon 模式：保持模型实例常驻 + 多次命令调用

### 1) 启动服务端（一次加载）

```bash
python BaseModel/SDXL-Base-1.0/sdxl_base_daemon_server.py \
  --host 127.0.0.1 --port 6320 \
  --device cuda --torch_dtype float16 --variant fp16
```

### 2) 客户端多次调用（不重复加载）

```bash
python BaseModel/SDXL-Base-1.0/sdxl_base_daemon_client.py \
  --host 127.0.0.1 --port 6320 \
  --prompt "An astronaut riding a green horse" \
  --out outputs/sdxl.png
```

健康检查：

```bash
python BaseModel/SDXL-Base-1.0/sdxl_base_daemon_client.py --ping
```

## 反复调用 daemon 以占用 GPU（Burn）

```bash
python BaseModel/SDXL-Base-1.0/gpu_burn_sdxl_base.py \
  --host 127.0.0.1 --port 6320 \
  --repeat 1000000 \
  --steps 30 \
  --out BaseModel/SDXL-Base-1.0/test/gpu_burn.png
```

## 文件说明

- `sdxl_base.py`: `SDXLBase10` 封装 + `get_sdxl_base()` 缓存工厂
- `demo_sdxl_base.py`: 单次生成 demo
- `batch_run_sdxl_base.py`: 读文件批量生成（一次加载，多次生成）
- `sdxl_base_daemon_server.py`: 常驻服务端
- `sdxl_base_daemon_client.py`: 命令客户端
- `gpu_burn_sdxl_base.py`: 循环调用 daemon（持续占用 GPU）


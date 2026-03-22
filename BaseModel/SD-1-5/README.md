# SD-1-5（Stable Diffusion v1.5 / Diffusers）调用说明

本目录提供对本地模型 `models/stable-diffusion-v1-5` 的调用封装与 demo。

## 模型路径（默认）

- 默认模型目录：`models/stable-diffusion-v1-5`
- 也可通过环境变量覆盖：
  - `SD15_MODEL_DIR`: 直接指定模型目录
  - `OFT_MODELS_DIR`: 指定 models 根目录（脚本会拼上 `stable-diffusion-v1-5`）
- 需要包含 `model_index.json`（你当前目录已包含）。

## 依赖

- **PyTorch**（建议 CUDA 版）
- **diffusers**（及 `transformers` / `accelerate` / `safetensors` 等）

## 方式 1：直接运行 demo（对应你给的示例）

```bash
python BaseModel/SD-1-5/demo_sd1_5.py \
  --prompt "a photo of an astronaut riding a horse on mars" \
  --out outputs/astronaut_rides_horse.png
```

常用参数：

- `--device`: `cuda` / `cuda:0` / `cpu`（默认自动）
- `--torch_dtype`: `auto|float16|bfloat16|float32`（默认 `auto`）
- `--steps` / `--guidance` / `--seed`
- `--height` / `--width`

## 方式 2：Python 代码调用（封装版）

```python
from sd1_5 import SD15

model = SD15(
    model_dir="models/stable-diffusion-v1-5",
    device="cuda",
    torch_dtype="float16",
)
image = model.generate(
    "a photo of an astronaut riding a horse on mars",
    num_inference_steps=30,
    guidance_scale=7.5,
)
image.save("astronaut_rides_horse.png")
```

## 避免重复加载（同进程复用）

```python
from sd1_5 import get_sd15

model = get_sd15(device="cuda", torch_dtype="float16")  # 首次加载
img1 = model.generate("a photo of an astronaut riding a horse on mars")
img2 = model.generate("a watercolor painting of a city at night")
```

## Daemon 模式：保持模型实例常驻 + 多次命令调用

### 1) 启动服务端（一次加载）

```bash
python BaseModel/SD-1-5/sd1_5_daemon_server.py \
  --host 127.0.0.1 --port 6315 \
  --device cuda --torch_dtype float16
```

### 2) 客户端多次调用（不重复加载）

```bash
python BaseModel/SD-1-5/sd1_5_daemon_client.py \
  --host 127.0.0.1 --port 6315 \
  --prompt "a photo of an astronaut riding a horse on mars" \
  --out outputs/astronaut_rides_horse.png
```

健康检查：

```bash
python BaseModel/SD-1-5/sd1_5_daemon_client.py --ping
```

## 反复调用 daemon 以占用 GPU（Burn）

```bash
python BaseModel/SD-1-5/gpu_burn_sd1_5.py \
  --host 127.0.0.1 --port 6315 \
  --repeat 1000000 \
  --steps 30 \
  --out BaseModel/SD-1-5/test/gpu_burn.png
```

## 文件说明

- `sd1_5.py`: `SD15` 封装 + `get_sd15()` 缓存工厂
- `demo_sd1_5.py`: 单次生成 demo
- `batch_run_sd1_5.py`: 读文件批量生成（一次加载，多次生成）
- `sd1_5_daemon_server.py`: 常驻服务端
- `sd1_5_daemon_client.py`: 命令客户端
- `gpu_burn_sd1_5.py`: 循环调用 daemon（持续占用 GPU）


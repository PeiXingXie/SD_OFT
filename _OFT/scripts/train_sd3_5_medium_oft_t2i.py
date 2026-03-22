from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, InitProcessGroupKwargs, set_seed
from diffusers import StableDiffusion3Pipeline
from tqdm.auto import tqdm

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.csv_data import read_csv_examples, read_csv_prompts
from utils.data import CsvTextImageDataset, csv_collate_fn
from utils.peft_oft import apply_oft_to_unet, default_target_modules


_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _default_model_dir(subdir: str, env_var: str) -> str:
    v = os.environ.get(env_var, "").strip()
    if v:
        return str(Path(v).expanduser())
    base = os.environ.get("OFT_MODELS_DIR", "").strip()
    if base:
        return str((Path(base).expanduser() / subdir).resolve())
    return str((_PROJECT_ROOT / "models" / subdir).resolve())


DEFAULT_SD35_MEDIUM_DIR = _default_model_dir("stable-diffusion-3.5-medium", "SD35_MODEL_DIR")


def _parse_list(s: str) -> List[str]:
    parts = [p.strip() for p in s.split(",")]
    return [p for p in parts if p]


def _resolve_dtype(mixed_precision: str, *, device_type: str) -> torch.dtype:
    # accelerate mixed_precision: "no" | "fp16" | "bf16"
    if device_type != "cuda":
        return torch.float32
    if mixed_precision == "bf16":
        return torch.bfloat16
    if mixed_precision == "fp16":
        return torch.float16
    return torch.float32


def _apply_sigma_shift(sigma: torch.Tensor, *, shift: float) -> torch.Tensor:
    """
    Match FlowMatchEulerDiscreteScheduler's default static sigma shift:
      sigma <- shift*sigma / (1 + (shift-1)*sigma)
    """
    if shift is None:
        return sigma
    s = float(shift)
    if s == 1.0:
        return sigma
    return s * sigma / (1.0 + (s - 1.0) * sigma)


def _encode_prompt_sd3(
    *,
    pipe: StableDiffusion3Pipeline,
    prompts: List[str],
    device: torch.device,
    dtype: torch.dtype,
    max_sequence_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    SD3 encode_prompt returns 4 tensors when CFG is enabled:
      (prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds)
    We set do_classifier_free_guidance=False for training and still handle both cases defensively.
    """
    # For SD3, encode_prompt signature requires prompt_2 and prompt_3 explicitly.
    out = pipe.encode_prompt(
        prompt=prompts,
        prompt_2=prompts,
        prompt_3=prompts,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=False,
        max_sequence_length=int(max_sequence_length),
    )

    if not isinstance(out, tuple):
        raise RuntimeError(f"Unexpected encode_prompt return type: {type(out)}")

    if len(out) == 4:
        prompt_embeds, _neg, pooled_prompt_embeds, _neg_pooled = out
    elif len(out) == 2:
        prompt_embeds, pooled_prompt_embeds = out
    else:
        raise RuntimeError(f"Unexpected encode_prompt return length: {len(out)}")

    prompt_embeds = prompt_embeds.to(device=device, dtype=dtype)
    pooled_prompt_embeds = pooled_prompt_embeds.to(device=device, dtype=dtype)
    return prompt_embeds, pooled_prompt_embeds


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    # model / data
    p.add_argument("--model_dir", type=str, default=DEFAULT_SD35_MEDIUM_DIR)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--use_safetensors", action="store_true", default=True)
    p.add_argument("--no_use_safetensors", action="store_false", dest="use_safetensors")

    # CSV input (train + val)
    p.add_argument("--train_csv", type=str, required=True, help="Training CSV file path.")
    p.add_argument("--val_csv", type=str, required=True, help="Validation CSV file path.")
    p.add_argument("--csv_delimiter", type=str, default=",", help="CSV delimiter, default ','")
    p.add_argument(
        "--image_root",
        type=str,
        default=None,
        help="Optional root dir for relative image paths in CSV. Defaults to CSV parent dir.",
    )
    p.add_argument("--image_col", type=str, default="image", help="CSV column name for image path.")
    p.add_argument("--text_col", type=str, default="text", help="CSV column name for prompt text.")
    p.add_argument(
        "--negative_col",
        type=str,
        default=None,
        help="Optional CSV column name for negative prompt text (only used in validation gen).",
    )

    # training
    p.add_argument("--resolution", type=int, default=1024)
    p.add_argument("--center_crop", action="store_true", default=True)
    p.add_argument("--no_center_crop", action="store_false", dest="center_crop")
    p.add_argument("--random_flip", action="store_true", default=True)
    p.add_argument("--no_random_flip", action="store_false", dest="random_flip")

    p.add_argument("--train_batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    # LR warmup (applied on optimizer steps, i.e. after gradient accumulation)
    p.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=0,
        help="Linear warmup steps for learning rate (in optimizer steps). 0 disables warmup.",
    )
    p.add_argument(
        "--lr_warmup_ratio",
        type=float,
        default=None,
        help="Optional warmup ratio in [0,1]. If set and lr_warmup_steps=0, warmup_steps = int(max_train_steps*ratio).",
    )
    p.add_argument("--adam_beta1", type=float, default=0.9)
    p.add_argument("--adam_beta2", type=float, default=0.999)
    p.add_argument("--adam_weight_decay", type=float, default=1e-2)
    p.add_argument("--adam_epsilon", type=float, default=1e-8)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--max_train_steps", type=int, default=1000)
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--logging_steps", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16"],
    )
    p.add_argument(
        "--ddp_timeout_sec",
        type=int,
        default=1800,
        help="DDP/NCCL process group timeout in seconds (multi-GPU). Increase if model loading is slow.",
    )
    p.add_argument(
        "--max_sequence_length",
        type=int,
        default=256,
        help="SD3 max_sequence_length passed to encode_prompt (default 256).",
    )

    # Flow matching objective
    p.add_argument(
        "--sigma_min",
        type=float,
        default=0.0,
        help="Min sigma for training sampling (default 0.0).",
    )
    p.add_argument(
        "--sigma_max",
        type=float,
        default=1.0,
        help="Max sigma for training sampling (default 1.0).",
    )

    # OFT
    p.add_argument("--oft_block_size", type=int, default=32)
    p.add_argument("--use_cayley_neumann", action="store_true", default=True)
    p.add_argument("--no_use_cayley_neumann", action="store_false", dest="use_cayley_neumann")
    p.add_argument("--module_dropout", type=float, default=0.0)
    p.add_argument("--bias", type=str, default="none", choices=["none", "all", "oft_only"])
    p.add_argument(
        "--target_modules",
        type=str,
        default=",".join(default_target_modules()),
        help="Comma separated module suffixes, e.g. to_q,to_k,to_v,to_out.0",
    )

    # validation generation on every adapter save
    p.add_argument(
        "--val_num_samples",
        type=int,
        default=4,
        help="How many validation rows (prompts) to generate each save. <=0 means all rows.",
    )
    p.add_argument(
        "--val_distributed",
        type=str,
        default="none",
        choices=["none", "shard"],
        help='Validation generation mode. "shard" splits prompts across ranks (multi-GPU) for speed; seeds stay identical to single-GPU (seed=val_seed+i).',
    )
    p.add_argument(
        "--val_every_steps",
        type=int,
        default=None,
        help="Run validation every N optimizer steps. Default: same as save_steps. 0 disables validation.",
    )
    p.add_argument("--val_steps", type=int, default=40)
    p.add_argument("--val_guidance", type=float, default=4.5)
    p.add_argument(
        "--val_seed",
        type=int,
        default=1234,
        help="Base seed for validation generations; per-sample seed = val_seed + i.",
    )
    p.add_argument("--val_max_sequence_length", type=int, default=None)
    p.add_argument("--val_height", type=int, default=None)
    p.add_argument("--val_width", type=int, default=None)

    return p


def main() -> None:
    args = build_argparser().parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    adapter_root = out_dir / "adapter"
    adapter_root.mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    (out_dir / "validation").mkdir(parents=True, exist_ok=True)

    # Multi-GPU: SD3.5 weights are huge. If ranks load at different speeds, faster ranks can enter
    # DDP parameter sync (broadcast) early and hit the watchdog timeout. We make the timeout
    # configurable and disable broadcast_buffers to reduce unnecessary sync traffic.
    ddp_timeout = timedelta(seconds=int(args.ddp_timeout_sec))
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        kwargs_handlers=[
            InitProcessGroupKwargs(timeout=ddp_timeout),
            DistributedDataParallelKwargs(broadcast_buffers=False),
        ],
    )
    # Keep dataloader shuffling consistent across ranks (so each rank gets a distinct shard),
    # but use a per-rank RNG for sigma/noise sampling (to avoid identical noise across GPUs).
    set_seed(args.seed)
    rng = torch.Generator(device=accelerator.device).manual_seed(
        int(args.seed) + 1_000_003 * int(accelerator.process_index)
    )

    def _randn_like(x: torch.Tensor) -> torch.Tensor:
        # Some torch builds don't support randn_like(generator=...), so use randn(shape,...).
        return torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=rng)

    if accelerator.is_main_process:
        (out_dir / "run_args.json").write_text(
            json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8"
        )

    dtype = _resolve_dtype(args.mixed_precision, device_type=accelerator.device.type)

    pipe = StableDiffusion3Pipeline.from_pretrained(
        args.model_dir,
        torch_dtype=dtype,
        use_safetensors=bool(args.use_safetensors),
    )
    pipe.set_progress_bar_config(disable=True)

    # Components
    vae = pipe.vae
    transformer = pipe.transformer
    text_encoder = pipe.text_encoder
    text_encoder_2 = pipe.text_encoder_2
    text_encoder_3 = pipe.text_encoder_3
    scheduler = pipe.scheduler

    # Freeze everything except OFT adapter on transformer
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    text_encoder_3.requires_grad_(False)
    transformer.requires_grad_(False)

    peft_transformer = apply_oft_to_unet(
        transformer,
        oft_block_size=args.oft_block_size,
        use_cayley_neumann=args.use_cayley_neumann,
        module_dropout=args.module_dropout,
        bias=args.bias,
        target_modules=_parse_list(args.target_modules),
    )
    peft_transformer.train()

    # Dataset / loader
    train_examples = read_csv_examples(
        args.train_csv,
        image_col=args.image_col,
        text_col=args.text_col,
        negative_col=args.negative_col,
        image_root=args.image_root,
        delimiter=args.csv_delimiter,
    )
    dataset = CsvTextImageDataset(
        train_examples,
        resolution=args.resolution,
        center_crop=args.center_crop,
        random_flip=args.random_flip,
    )
    dl = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=csv_collate_fn,
        pin_memory=True,
    )

    val_limit = None if int(args.val_num_samples) <= 0 else int(args.val_num_samples)
    val_prompts = read_csv_prompts(
        args.val_csv,
        text_col=args.text_col,
        negative_col=args.negative_col,
        delimiter=args.csv_delimiter,
        limit=val_limit,
    )

    val_every_steps = args.val_every_steps
    if val_every_steps is None:
        val_every_steps = int(args.save_steps)
    val_every_steps = int(val_every_steps)
    if val_every_steps < 0:
        raise ValueError("--val_every_steps must be >= 0")

    trainable_params = [p for p in peft_transformer.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Warmup config (optimizer steps)
    warmup_steps = int(args.lr_warmup_steps)
    if (warmup_steps <= 0) and (args.lr_warmup_ratio is not None):
        r = float(args.lr_warmup_ratio)
        if not (0.0 <= r <= 1.0):
            raise ValueError("--lr_warmup_ratio must be in [0, 1]")
        warmup_steps = int(int(args.max_train_steps) * r)
    warmup_steps = max(0, min(int(warmup_steps), int(args.max_train_steps)))
    base_lrs = [float(pg["lr"]) for pg in optimizer.param_groups]

    def _apply_warmup_lr(next_step_1based: int) -> float:
        """
        Set optimizer LR for the upcoming optimizer step (1-based index).
        Returns the new LR of param_group[0] for convenience.
        """
        if warmup_steps <= 0:
            return float(optimizer.param_groups[0]["lr"])
        scale = min(float(next_step_1based) / float(warmup_steps), 1.0)
        for pg, base_lr in zip(optimizer.param_groups, base_lrs):
            pg["lr"] = float(base_lr) * float(scale)
        return float(optimizer.param_groups[0]["lr"])

    # Barrier before DDP wrapping to avoid starting large parameter broadcasts while some ranks are still loading.
    accelerator.wait_for_everyone()
    peft_transformer, optimizer, dl = accelerator.prepare(peft_transformer, optimizer, dl)

    # Move frozen modules
    pipe.to(accelerator.device)
    vae.to(accelerator.device, dtype=dtype)
    text_encoder.to(accelerator.device, dtype=dtype)
    text_encoder_2.to(accelerator.device, dtype=dtype)
    text_encoder_3.to(accelerator.device, dtype=dtype)

    # Scheduler configs for training-time sigma sampling
    shift = scheduler.config.get("shift", 1.0)
    num_train_timesteps = int(scheduler.config.get("num_train_timesteps", 1000))

    steps_per_epoch = math.ceil(len(dl) / args.gradient_accumulation_steps)
    num_epochs = math.ceil(args.max_train_steps / steps_per_epoch)

    global_step = 0
    # Accumulate loss over micro-batches within one optimizer step (gradient accumulation window).
    # We will reduce (sum) across GPUs at sync points and log the global mean.
    loss_sum = torch.zeros((), device=accelerator.device, dtype=torch.float32)
    loss_count = torch.zeros((), device=accelerator.device, dtype=torch.float32)
    progress = tqdm(
        range(args.max_train_steps),
        disable=not accelerator.is_local_main_process,
        desc="train",
    )

    metrics_path = out_dir / "logs" / "metrics.csv"
    t0 = time.perf_counter()
    last_step_t = t0
    if accelerator.is_main_process and not metrics_path.exists():
        metrics_path.write_text(
            "timestamp,global_step,loss,grad_norm_pre_clip,step_seconds,total_seconds\n",
            encoding="utf-8",
        )

    def _append_metrics(*, step: int, loss_value: float, grad_norm_pre_clip: Optional[float]):
        nonlocal last_step_t
        if not accelerator.is_main_process:
            return
        now = time.perf_counter()
        step_s = now - last_step_t
        total_s = now - t0
        last_step_t = now
        ts = datetime.now(timezone.utc).isoformat()
        gn = "" if grad_norm_pre_clip is None else f"{float(grad_norm_pre_clip):.6g}"
        line = f"{ts},{int(step)},{float(loss_value):.8g},{gn},{step_s:.6f},{total_s:.6f}\n"
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(line)
            f.flush()

    def _run_validation(save_step: int) -> None:
        if not val_prompts:
            return

        unwrapped = accelerator.unwrap_model(peft_transformer)
        unwrapped.eval()

        pipe.transformer = unwrapped
        pipe.to(accelerator.device)
        pipe.set_progress_bar_config(disable=True)

        # Baseline generation (base model without adapter): generate ONCE per run
        # so you can compare baseline vs adapter outputs on the same prompts/seeds.
        baseline_dir = out_dir / "validation" / "baseline"
        baseline_done = baseline_dir / "_DONE"
        baseline_dir.mkdir(parents=True, exist_ok=True)

        if args.val_distributed == "shard" and accelerator.num_processes > 1:
            rank = int(accelerator.process_index)
            baseline_meta = baseline_dir / f"prompts_rank{rank:02d}.jsonl"
            shard_indices = [
                i for i in range(len(val_prompts)) if (i % accelerator.num_processes) == rank
            ]
        else:
            baseline_meta = baseline_dir / "prompts.jsonl"
            shard_indices = list(range(len(val_prompts)))

        if not baseline_done.exists():
            with baseline_meta.open("w", encoding="utf-8") as mf:
                for i in shard_indices:
                    row = val_prompts[i]
                    prompt = row["text"]
                    neg = row.get("negative_text")
                    seed = int(args.val_seed) + int(i)
                    gen = torch.Generator(device=accelerator.device).manual_seed(seed)

                    kwargs = {}
                    if args.val_height is not None:
                        kwargs["height"] = int(args.val_height)
                    if args.val_width is not None:
                        kwargs["width"] = int(args.val_width)
                    kwargs["max_sequence_length"] = int(
                        args.val_max_sequence_length
                        if args.val_max_sequence_length is not None
                        else args.max_sequence_length
                    )

                    use_autocast = accelerator.device.type == "cuda" and dtype in (
                        torch.float16,
                        torch.bfloat16,
                    )
                    # Disable adapter to get the original (base) model output
                    if use_autocast:
                        with (
                            torch.inference_mode(),
                            torch.autocast(device_type="cuda", dtype=dtype),
                            unwrapped.disable_adapter(),
                        ):
                            out = pipe(
                                prompt=prompt,
                                negative_prompt=neg,
                                num_inference_steps=int(args.val_steps),
                                guidance_scale=float(args.val_guidance),
                                generator=gen,
                                **kwargs,
                            )
                    else:
                        with torch.inference_mode(), unwrapped.disable_adapter():
                            out = pipe(
                                prompt=prompt,
                                negative_prompt=neg,
                                num_inference_steps=int(args.val_steps),
                                guidance_scale=float(args.val_guidance),
                                generator=gen,
                                **kwargs,
                            )

                    img = out.images[0]
                    out_path = baseline_dir / f"{int(i):04d}.png"
                    img.save(str(out_path))
                    mf.write(
                        json.dumps(
                            {
                                "i": int(i),
                                "prompt": prompt,
                                "negative_prompt": neg,
                                "seed": int(seed),
                                "out": str(out_path),
                                "step": 0,
                                "tag": "baseline",
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

            accelerator.wait_for_everyone()
            if accelerator.is_main_process and args.val_distributed == "shard" and accelerator.num_processes > 1:
                merged = baseline_dir / "prompts.jsonl"
                with merged.open("w", encoding="utf-8") as out_f:
                    for r in range(int(accelerator.num_processes)):
                        rp = baseline_dir / f"prompts_rank{r:02d}.jsonl"
                        if rp.exists():
                            out_f.write(rp.read_text(encoding="utf-8"))
                baseline_done.write_text("ok\n", encoding="utf-8")
            elif accelerator.is_main_process and args.val_distributed == "none":
                baseline_done.write_text("ok\n", encoding="utf-8")
            accelerator.wait_for_everyone()

        step_dir = out_dir / "validation" / f"step_{int(save_step):06d}"
        step_dir.mkdir(parents=True, exist_ok=True)

        if args.val_distributed == "shard" and accelerator.num_processes > 1:
            rank = int(accelerator.process_index)
            meta_path = step_dir / f"prompts_rank{rank:02d}.jsonl"
            shard_indices = [
                i for i in range(len(val_prompts)) if (i % accelerator.num_processes) == rank
            ]
        else:
            meta_path = step_dir / "prompts.jsonl"
            shard_indices = list(range(len(val_prompts)))

        with meta_path.open("w", encoding="utf-8") as mf:
            for i in shard_indices:
                row = val_prompts[i]
                prompt = row["text"]
                neg = row.get("negative_text")
                seed = int(args.val_seed) + int(i)
                gen = torch.Generator(device=accelerator.device).manual_seed(seed)

                kwargs = {}
                if args.val_height is not None:
                    kwargs["height"] = int(args.val_height)
                if args.val_width is not None:
                    kwargs["width"] = int(args.val_width)
                kwargs["max_sequence_length"] = int(
                    args.val_max_sequence_length
                    if args.val_max_sequence_length is not None
                    else args.max_sequence_length
                )

                use_autocast = accelerator.device.type == "cuda" and dtype in (
                    torch.float16,
                    torch.bfloat16,
                )
                if use_autocast:
                    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=dtype):
                        out = pipe(
                            prompt=prompt,
                            negative_prompt=neg,
                            num_inference_steps=int(args.val_steps),
                            guidance_scale=float(args.val_guidance),
                            generator=gen,
                            **kwargs,
                        )
                else:
                    with torch.inference_mode():
                        out = pipe(
                            prompt=prompt,
                            negative_prompt=neg,
                            num_inference_steps=int(args.val_steps),
                            guidance_scale=float(args.val_guidance),
                            generator=gen,
                            **kwargs,
                        )
                img = out.images[0]
                out_path = step_dir / f"{int(i):04d}.png"
                img.save(str(out_path))
                mf.write(
                    json.dumps(
                        {
                            "i": int(i),
                            "prompt": prompt,
                            "negative_prompt": neg,
                            "seed": int(seed),
                            "out": str(out_path),
                            "step": int(save_step),
                            "tag": "adapter",
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        accelerator.wait_for_everyone()
        if accelerator.is_main_process and args.val_distributed == "shard" and accelerator.num_processes > 1:
            merged = step_dir / "prompts.jsonl"
            with merged.open("w", encoding="utf-8") as out_f:
                for r in range(int(accelerator.num_processes)):
                    rp = step_dir / f"prompts_rank{r:02d}.jsonl"
                    if rp.exists():
                        out_f.write(rp.read_text(encoding="utf-8"))
        accelerator.wait_for_everyone()

        unwrapped.train()

    for _epoch in range(num_epochs):
        for batch in dl:
            with accelerator.accumulate(peft_transformer):
                pixel_values = batch["pixel_values"].to(accelerator.device, dtype=dtype)
                texts: List[str] = batch["texts"]

                with torch.no_grad():
                    prompt_embeds, pooled_prompt_embeds = _encode_prompt_sd3(
                        pipe=pipe,
                        prompts=texts,
                        device=accelerator.device,
                        dtype=dtype,
                        max_sequence_length=int(args.max_sequence_length),
                    )

                # VAE -> latents (x0)
                with torch.no_grad():
                    latents = vae.encode(pixel_values).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                # Flow matching: sample sigma and create x_sigma = (1-sigma)*x0 + sigma*eps
                noise = _randn_like(latents)
                bsz = latents.shape[0]

                sigma = torch.rand(
                    (bsz,),
                    device=latents.device,
                    dtype=torch.float32,
                    generator=rng,
                )
                sigma = sigma * (float(args.sigma_max) - float(args.sigma_min)) + float(args.sigma_min)
                sigma = sigma.clamp(0.0, 1.0)
                sigma = _apply_sigma_shift(sigma, shift=shift)

                sigma_b = sigma.view(bsz, 1, 1, 1).to(dtype=latents.dtype)
                noisy_latents = (1.0 - sigma_b) * latents + sigma_b * noise

                # timestep in SD3 pipeline is sigma * num_train_timesteps (float)
                timesteps = (sigma * float(num_train_timesteps)).to(device=latents.device, dtype=torch.float32)

                # Predict "velocity" = eps - x0 (matches FlowMatchEulerDiscreteScheduler ODE update)
                model_pred = peft_transformer(
                    hidden_states=noisy_latents,
                    timestep=timesteps,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    joint_attention_kwargs=None,
                    return_dict=False,
                )[0]

                target = (noise - latents).to(dtype=model_pred.dtype)
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                accelerator.backward(loss)
                # Track micro-batch loss (local) in fp32 for stable aggregation.
                loss_sum = loss_sum + loss.detach().float()
                loss_count = loss_count + 1.0

                grad_norm_pre_clip: Optional[float] = None
                if accelerator.sync_gradients:
                    # Apply LR warmup on actual optimizer steps (post-accumulation).
                    _lr_now = _apply_warmup_lr(global_step + 1)
                    if args.max_grad_norm is not None and float(args.max_grad_norm) > 0:
                        grad_norm_pre_clip = accelerator.clip_grad_norm_(
                            trainable_params, float(args.max_grad_norm)
                        )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                # Global mean loss over the gradient accumulation window, reduced across GPUs.
                reduced_sum = accelerator.reduce(loss_sum, reduction="sum")
                reduced_count = accelerator.reduce(loss_count, reduction="sum")
                mean_loss = (reduced_sum / torch.clamp(reduced_count, min=1.0)).detach()
                # Reset window accumulators for next optimizer step.
                loss_sum = torch.zeros((), device=accelerator.device, dtype=torch.float32)
                loss_count = torch.zeros((), device=accelerator.device, dtype=torch.float32)

                global_step += 1
                progress.update(1)

                if accelerator.is_main_process and global_step % args.logging_steps == 0:
                    lr0 = float(optimizer.param_groups[0]["lr"])
                    progress.set_postfix({"loss": float(mean_loss.cpu()), "lr": lr0})

                _append_metrics(
                    step=global_step,
                    loss_value=float(mean_loss.cpu()),
                    grad_norm_pre_clip=None
                    if grad_norm_pre_clip is None
                    else float(grad_norm_pre_clip),
                )

                if global_step % args.save_steps == 0:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        adapter_dir = adapter_root / f"step_{int(global_step):06d}"
                        adapter_dir.mkdir(parents=True, exist_ok=True)
                        unwrapped = accelerator.unwrap_model(peft_transformer)
                        unwrapped.save_pretrained(str(adapter_dir))
                        (out_dir / "global_step.txt").write_text(
                            str(global_step), encoding="utf-8"
                        )
                    accelerator.wait_for_everyone()
                    if val_every_steps > 0 and (global_step % val_every_steps == 0):
                        if args.val_distributed == "shard" and accelerator.num_processes > 1:
                            _run_validation(global_step)
                        elif accelerator.is_main_process:
                            _run_validation(global_step)
                    accelerator.wait_for_everyone()

                if global_step >= args.max_train_steps:
                    break

        if global_step >= args.max_train_steps:
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        adapter_dir = adapter_root / f"step_{int(global_step):06d}"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        unwrapped = accelerator.unwrap_model(peft_transformer)
        unwrapped.save_pretrained(str(adapter_dir))
        (out_dir / "global_step.txt").write_text(str(global_step), encoding="utf-8")
    accelerator.wait_for_everyone()
    if val_every_steps > 0 and (global_step % val_every_steps == 0):
        if args.val_distributed == "shard" and accelerator.num_processes > 1:
            _run_validation(global_step)
        elif accelerator.is_main_process:
            _run_validation(global_step)
    accelerator.wait_for_everyone()

    accelerator.wait_for_everyone()


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()


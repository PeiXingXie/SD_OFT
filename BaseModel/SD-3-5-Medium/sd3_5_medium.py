from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional, Union


_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _default_model_dir(subdir: str, env_var: str) -> str:
    import os

    v = os.environ.get(env_var, "").strip()
    if v:
        return str(Path(v).expanduser())
    base = os.environ.get("OFT_MODELS_DIR", "").strip()
    if base:
        return str((Path(base).expanduser() / subdir).resolve())
    return str((_PROJECT_ROOT / "models" / subdir).resolve())


DEFAULT_SD35_MEDIUM_DIR = _default_model_dir("stable-diffusion-3.5-medium", "SD35_MODEL_DIR")


@dataclass
class SD35Medium:
    """
    Stable Diffusion 3.5 Medium (Diffusers) wrapper.

    - Default model_dir points to the local folder you provided.
    - By default uses CUDA if available, otherwise CPU.
    """

    model_dir: Union[str, Path] = DEFAULT_SD35_MEDIUM_DIR
    device: Optional[str] = None  # e.g. "cuda", "cuda:0", "cpu"
    torch_dtype: Optional[str] = None  # "bfloat16" | "float16" | "float32" | None (auto)
    use_safetensors: bool = True

    _pipe: object = None

    def load(self):
        """
        Lazily load the StableDiffusion3Pipeline.

        Returns:
            diffusers.StableDiffusion3Pipeline
        """

        try:
            import torch
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "Missing dependency: torch. Please install PyTorch first."
            ) from e

        try:
            from diffusers import StableDiffusion3Pipeline
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "Missing dependency: diffusers. Please install diffusers (and its deps) first."
            ) from e

        model_dir = str(self.model_dir)
        if not Path(model_dir).exists():
            raise FileNotFoundError(f"model_dir not found: {model_dir}")

        device = self.device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        dtype = None
        if self.torch_dtype is None:
            if device.startswith("cuda"):
                # Prefer bfloat16 on newer GPUs, otherwise float16.
                dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            else:
                dtype = torch.float32
        else:
            if self.torch_dtype == "bfloat16":
                dtype = torch.bfloat16
            elif self.torch_dtype == "float16":
                dtype = torch.float16
            elif self.torch_dtype == "float32":
                dtype = torch.float32
            else:
                raise ValueError(
                    f"Unsupported torch_dtype: {self.torch_dtype}. "
                    "Use one of: None, 'bfloat16', 'float16', 'float32'."
                )

        pipe = StableDiffusion3Pipeline.from_pretrained(
            model_dir,
            torch_dtype=dtype,
            use_safetensors=self.use_safetensors,
        )

        # Move to target device. (CPU offload is intentionally not enabled by default.)
        pipe = pipe.to(device)

        self._pipe = pipe
        self.device = device
        return pipe

    @property
    def pipe(self):
        if self._pipe is None:
            return self.load()
        return self._pipe

    def generate(
        self,
        prompt: str,
        *,
        negative_prompt: Optional[str] = None,
        num_inference_steps: int = 40,
        guidance_scale: float = 4.5,
        seed: Optional[int] = None,
        **kwargs,
    ):
        """
        Run text-to-image generation.

        Returns:
            PIL.Image.Image
        """

        import torch

        generator = None
        if seed is not None:
            # Generator should be on the same device as pipeline.
            generator = torch.Generator(device=self.device).manual_seed(int(seed))

        with torch.inference_mode():
            out = self.pipe(
                prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                **kwargs,
            )
        return out.images[0]

    def generate_to_file(
        self,
        prompt: str,
        output_path: Union[str, Path],
        *,
        negative_prompt: Optional[str] = None,
        num_inference_steps: int = 40,
        guidance_scale: float = 4.5,
        seed: Optional[int] = None,
        **kwargs,
    ) -> Path:
        """
        Convenience wrapper: generate and save to `output_path`.

        Returns:
            pathlib.Path to the saved image
        """

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        image = self.generate(
            prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            **kwargs,
        )
        image.save(str(output_path))
        return output_path


def _norm_model_dir(model_dir: Union[str, Path]) -> str:
    return str(Path(model_dir).expanduser().resolve())


@lru_cache(maxsize=4)
def get_sd35_medium(
    model_dir: Union[str, Path] = DEFAULT_SD35_MEDIUM_DIR,
    *,
    device: Optional[str] = None,
    torch_dtype: Optional[str] = None,
    use_safetensors: bool = True,
) -> SD35Medium:
    """
    Get a cached, loaded SD35Medium instance (singleton-style) to avoid re-loading weights.

    Notes:
    - Cache is per Python process.
    - Cache key includes (model_dir, device, torch_dtype, use_safetensors).
    - If you need to free GPU memory, call `get_sd35_medium.cache_clear()` and then `torch.cuda.empty_cache()`.
    """

    model = SD35Medium(
        model_dir=_norm_model_dir(model_dir),
        device=device,
        torch_dtype=torch_dtype,
        use_safetensors=use_safetensors,
    )
    model.load()
    return model


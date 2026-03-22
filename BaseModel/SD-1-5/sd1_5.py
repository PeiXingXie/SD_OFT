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


DEFAULT_SD15_DIR = _default_model_dir("stable-diffusion-v1-5", "SD15_MODEL_DIR")


@dataclass
class SD15:
    """
    Stable Diffusion v1.5 (Diffusers) wrapper.

    Notes:
    - Uses StableDiffusionPipeline (text-to-image).
    - Default model_dir points to your local folder.
    - By default uses CUDA if available, otherwise CPU.
    """

    model_dir: Union[str, Path] = DEFAULT_SD15_DIR
    device: Optional[str] = None  # "cuda" | "cuda:0" | "cpu"
    torch_dtype: Optional[str] = None  # "float16" | "bfloat16" | "float32" | None (auto)
    use_safetensors: bool = True

    _pipe: object = None

    def load(self):
        """
        Lazily load the StableDiffusionPipeline.

        Returns:
            diffusers.StableDiffusionPipeline
        """

        try:
            import torch
        except Exception as e:  # pragma: no cover
            raise RuntimeError("Missing dependency: torch. Please install PyTorch first.") from e

        try:
            from diffusers import StableDiffusionPipeline
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

        if self.torch_dtype is None:
            if device.startswith("cuda"):
                dtype = torch.float16
            else:
                dtype = torch.float32
        else:
            if self.torch_dtype == "float16":
                dtype = torch.float16
            elif self.torch_dtype == "bfloat16":
                dtype = torch.bfloat16
            elif self.torch_dtype == "float32":
                dtype = torch.float32
            else:
                raise ValueError(
                    f"Unsupported torch_dtype: {self.torch_dtype}. "
                    "Use one of: None, 'float16', 'bfloat16', 'float32'."
                )

        pipe = StableDiffusionPipeline.from_pretrained(
            model_dir,
            torch_dtype=dtype,
            use_safetensors=self.use_safetensors,
        )
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
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_images_per_prompt: Optional[int] = None,
        **kwargs,
    ):
        """
        Returns:
            PIL.Image.Image
        """

        import torch

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(int(seed))

        call_kwargs = dict(kwargs)
        if height is not None:
            call_kwargs["height"] = int(height)
        if width is not None:
            call_kwargs["width"] = int(width)
        if num_images_per_prompt is not None:
            call_kwargs["num_images_per_prompt"] = int(num_images_per_prompt)

        with torch.inference_mode():
            out = self.pipe(
                prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=int(num_inference_steps),
                guidance_scale=float(guidance_scale),
                generator=generator,
                **call_kwargs,
            )
        return out.images[0]

    def generate_to_file(
        self,
        prompt: str,
        output_path: Union[str, Path],
        *,
        negative_prompt: Optional[str] = None,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_images_per_prompt: Optional[int] = None,
        **kwargs,
    ) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        image = self.generate(
            prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            height=height,
            width=width,
            num_images_per_prompt=num_images_per_prompt,
            **kwargs,
        )
        image.save(str(output_path))
        return output_path


def _norm_model_dir(model_dir: Union[str, Path]) -> str:
    return str(Path(model_dir).expanduser().resolve())


@lru_cache(maxsize=4)
def get_sd15(
    model_dir: Union[str, Path] = DEFAULT_SD15_DIR,
    *,
    device: Optional[str] = None,
    torch_dtype: Optional[str] = None,
    use_safetensors: bool = True,
) -> SD15:
    """
    Get a cached, loaded SD1.5 instance to avoid re-loading weights.

    Cache is per Python process.
    """

    model = SD15(
        model_dir=_norm_model_dir(model_dir),
        device=device,
        torch_dtype=torch_dtype,
        use_safetensors=use_safetensors,
    )
    model.load()
    return model


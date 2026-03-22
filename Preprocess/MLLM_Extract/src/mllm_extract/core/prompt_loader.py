"""
Prompt loading utilities.

Supports:
- Inline prompts from YAML config: cfg["prompt"]["system_prompt"], cfg["prompt"]["user_prompt"]
- Python prompt files that define variables: system_prompt, user_prompt
  (compatible with `prompt_used.py` and `prompt/For*.py` in this repo)
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any


def load_prompt_from_py(prompt_file: str) -> tuple[str, str]:
    """
    Load `system_prompt` and `user_prompt` from a Python file.
    The file must define top-level variables: system_prompt, user_prompt.
    """

    p = Path(prompt_file).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"prompt file not found: {p}")
    spec = importlib.util.spec_from_file_location(f"_mllm_extract_prompt_{p.stem}", str(p))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to import prompt file: {p}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]

    system_prompt = getattr(mod, "system_prompt", None)
    user_prompt = getattr(mod, "user_prompt", None)
    if not isinstance(system_prompt, str) or not system_prompt.strip():
        raise RuntimeError(f"prompt file missing non-empty `system_prompt`: {p}")
    if not isinstance(user_prompt, str) or not user_prompt.strip():
        raise RuntimeError(f"prompt file missing non-empty `user_prompt`: {p}")
    return system_prompt, user_prompt


def resolve_prompt(
    *,
    root_dir: str,
    prompt_cfg: dict[str, Any],
    prompt_name: str | None,
    prompt_file: str | None,
) -> tuple[str, str]:
    """
    Priority:
    1) CLI prompt_file
    2) CLI prompt_name -> map to repo prompt files
    3) YAML prompt.prompt_file (optional)
    4) YAML inline system_prompt/user_prompt
    """

    root = Path(root_dir).expanduser().resolve()
    prompt_cfg = dict(prompt_cfg or {})

    if isinstance(prompt_file, str) and prompt_file.strip():
        return load_prompt_from_py(prompt_file.strip())

    name = (prompt_name or "").strip().lower()
    if name:
        mapping = {
            "common": root / "prompt" / "ForCommon.py",
            "pointillism": root / "prompt" / "ForPointillism.py",
            "used": root / "prompt_used.py",
            # evaluation prompts
            "semanticadherence": root / "prompt" / "SemanticAdherence.py",
            "semantic_adherence": root / "prompt" / "SemanticAdherence.py",
            "structuralplausibility": root / "prompt" / "StructuralPlausibility.py",
            "structural_plausibility": root / "prompt" / "StructuralPlausibility.py",
            # style match
            "stylematch": root / "prompt" / "StyleMatch.py",
            "style_match": root / "prompt" / "StyleMatch.py",
            # category classify
            "categoryclassify": root / "prompt" / "CategoryClassify.py",
            "category_classify": root / "prompt" / "CategoryClassify.py",
        }
        if name not in mapping:
            raise RuntimeError(f"unknown --prompt-name: {prompt_name!r}, expected one of {sorted(mapping.keys())}")
        return load_prompt_from_py(str(mapping[name]))

    cfg_prompt_file = str(prompt_cfg.get("prompt_file", "") or "").strip()
    if cfg_prompt_file:
        # allow relative to root
        p = Path(cfg_prompt_file)
        if not p.is_absolute():
            p = root / p
        return load_prompt_from_py(str(p))

    system_prompt = str(prompt_cfg.get("system_prompt", "") or "")
    user_prompt = str(prompt_cfg.get("user_prompt", "") or "")
    return system_prompt, user_prompt


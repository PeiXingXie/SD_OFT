"""
MLLM runner (batch images -> captions).

Usage:
  cd Preprocess/MLLM_Extract
  export PYTHONPATH=src
  python3 -m mllm_extract.cli.run_images --config configs/config_mllm.yaml --threads 8
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from tqdm import tqdm

from ..core.factory import build_mllm_client
from ..core.messages import build_messages
from ..core.prompt_loader import resolve_prompt
from ..core.client import AndesAPIError
from ..core.style_match_scoring import (
    build_style_match_stats_dict,
    compute_style_match_score,
    parse_style_match_base_json,
)
from ..io.csv_io import read_csv, write_csv
from ..io.image_io import load_image_as_data_url


def _norm_task(x: str | None) -> str:
    s = str(x or "").strip().lower()
    if s in ["", "caption", "captioning"]:
        return "caption"
    if s in ["semantic", "semanticadherence", "semantic_adherence", "semantic-adherence"]:
        return "semantic_adherence"
    if s in [
        "structural",
        "structuralplausibility",
        "structural_plausibility",
        "structural-plausibility",
    ]:
        return "structural_plausibility"
    if s in ["stylematch", "style_match", "style-match"]:
        return "style_match"
    if s in ["categoryclassify", "category_classify", "category-classify", "category"]:
        return "category_classify"
    return s


def _render_user_prompt(tpl: str, *, caption: str | None = None, sample_id: str | None = None) -> str:
    """
    Light-weight templating:
    - Replace {caption} and {id} tokens if present.
    (Using simple replace to avoid str.format() errors on stray braces.)
    """

    s = tpl or ""
    if caption is not None:
        s = s.replace("{caption}", str(caption))
    if sample_id is not None:
        s = s.replace("{id}", str(sample_id))
    return s


def _parse_eval_json(resp: Any) -> tuple[int | None, str | None]:
    """
    Parse model output for evaluation tasks. Expected JSON:
      {"score": <int 1-5>, "short_reason": "..."}
    """

    if resp is None:
        return None, None
    s = str(resp).strip()
    if not s:
        return None, None
    # try strict json first
    obj = None
    try:
        obj = json.loads(s)
    except Exception:
        # try to extract the first {...} block
        i = s.find("{")
        j = s.rfind("}")
        if i >= 0 and j > i:
            try:
                obj = json.loads(s[i : j + 1])
            except Exception:
                obj = None
    if not isinstance(obj, dict):
        return None, None
    score = obj.get("score", None)
    reason = obj.get("short_reason", None)
    try:
        score_i = int(score) if score is not None else None
    except Exception:
        score_i = None
    reason_s = None if reason is None else str(reason).strip()
    return score_i, reason_s


def _sleep_backoff(attempt: int) -> None:
    time.sleep(min(10.0, float(2 ** max(0, attempt - 1))))


def _as_str_cell(v: Any) -> str:
    """
    Convert a dataframe cell value to a safe string.
    - NaN/None/""/"nan" -> ""
    - otherwise -> str(v).strip()
    """

    return "" if _is_empty_cell(v) else str(v).strip()


def _as_id_cell(v: Any) -> str:
    """
    Normalize id values:
    - NaN/None -> ""
    - numeric like 12.0 -> "12"
    - otherwise -> trimmed string
    """

    if _is_empty_cell(v):
        return ""
    try:
        # float or numpy float
        fv = float(v)
        if abs(fv - int(fv)) < 1e-9:
            return str(int(fv))
    except Exception:
        pass
    return str(v).strip()


def _validate_eval_output(score: int | None, reason: str | None) -> tuple[int, str]:
    """
    For evaluation tasks, require a well-formed score and a non-empty short_reason.
    """

    if score is None:
        raise ValueError("invalid_eval_output: missing score")
    if int(score) < 1 or int(score) > 5:
        raise ValueError(f"invalid_eval_output: score out of range: {score}")
    rs = (reason or "").strip()
    if not rs:
        raise ValueError("invalid_eval_output: missing short_reason")
    return int(score), rs


def _parse_category_json(resp: Any) -> str | None:
    """
    Parse model output for CategoryClassify. Expected JSON:
      {"category": "..."}
    """

    if resp is None:
        return None
    s = str(resp).strip()
    if not s:
        return None
    obj = None
    try:
        obj = json.loads(s)
    except Exception:
        i = s.find("{")
        j = s.rfind("}")
        if i >= 0 and j > i:
            try:
                obj = json.loads(s[i : j + 1])
            except Exception:
                obj = None
    if not isinstance(obj, dict):
        return None
    cat = obj.get("category", None)
    if cat is None:
        return None
    return str(cat).strip()


def _validate_category_output(cat: str | None) -> str:
    allowed = {"portrait", "landscape", "still_life", "animal", "architecture"}
    if cat is None:
        raise ValueError("invalid_category_output: missing category")
    c = str(cat).strip()
    if c not in allowed:
        raise ValueError(f"invalid_category_output: {c!r}, expected one of {sorted(allowed)}")
    return c


def _task_prefix(task: str) -> str:
    t = _norm_task(task)
    if t == "semantic_adherence":
        return "semantic"
    if t == "structural_plausibility":
        return "structural"
    if t == "style_match":
        return "style_match"
    if t == "category_classify":
        return "category"
    return t or "caption"


def _normalize_raw_response_for_csv(x: Any) -> str:
    """
    Make raw_response safe for CSV:
    - Prefer minified JSON if parsable
    - Otherwise flatten newlines into literal \\n
    """

    if x is None:
        return ""
    s = str(x)
    if not s.strip():
        return ""
    try:
        obj = json.loads(s)
        return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        return s.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "\\n")


def _default_config_path() -> str:
    cwd_guess = Path("configs") / "config_mllm.yaml"
    if cwd_guess.exists():
        return str(cwd_guess)
    # run_images.py -> cli -> mllm_extract -> src -> <MLLM_Extract root>
    root = Path(__file__).resolve().parents[3]
    return str(root / "configs" / "config_mllm.yaml")


def _clamp_threads(x, default: int = 1) -> int:
    try:
        v = int(x)
    except Exception:
        v = int(default)
    return max(1, v)


def _normalize_caption(x: Any) -> str:
    """
    Best-effort normalize model output to a single-line caption.
    - Strip surrounding whitespace
    - Take the first non-empty line
    - Strip surrounding quotes if present
    """

    s = "" if x is None else str(x)
    s = s.strip()
    if not s:
        return ""
    # keep first non-empty line
    for line in s.splitlines():
        line = line.strip()
        if line:
            s = line
            break
    # remove wrapping quotes
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1].strip()
    return s


def _is_empty_cell(v: Any) -> bool:
    if v is None:
        return True
    try:
        if isinstance(v, float) and pd.isna(v):
            return True
    except Exception:
        pass
    s = str(v).strip()
    if not s:
        return True
    if s.lower() == "nan":
        return True
    return False


def _scan_dir(image_dir: str, *, recursive: bool, glob_pat: str, exts: list[str]) -> pd.DataFrame:
    p = Path(image_dir).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"input.image_dir not found: {p}")
    exts_norm = {e.lower() for e in (exts or [])}
    if recursive:
        it = p.rglob(glob_pat or "*")
    else:
        it = p.glob(glob_pat or "*")
    rows: list[dict[str, str]] = []
    for fp in it:
        if not fp.is_file():
            continue
        if exts_norm and fp.suffix.lower() not in exts_norm:
            continue
        rows.append({"id": fp.stem, "path": str(fp)})
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=_default_config_path(), help="config yaml path")
    ap.add_argument("--threads", type=int, default=1, help="parallel threads (default 1=serial)")
    ap.add_argument("--log-interval-s", type=int, default=10, help="parallel status log interval seconds")
    ap.add_argument("--retries", type=int, default=3, help="per-row retries on failure (default 3)")
    ap.add_argument("--no-resume", action="store_true", help="disable resume (force rerun all rows)")
    # overrides
    ap.add_argument("--input-dir", default=None, help="override input.image_dir (dir mode)")
    ap.add_argument("--input-csv", default=None, help="override input.input_csv (csv mode)")
    ap.add_argument("--output-csv", default=None, help="override output.output_csv")
    ap.add_argument("--output-jsonl", default=None, help="override output.output_jsonl (empty to disable)")
    ap.add_argument("--output-dir", default=None, help="override output directory (prefix for csv/jsonl if not explicitly set)")

    # task selection (default: caption extraction)
    ap.add_argument(
        "--task",
        default=None,
        help="task type: caption | semantic_adherence | structural_plausibility | style_match | category_classify (default from config.run.task or caption)",
    )

    # prompt selection
    ap.add_argument("--prompt-name", default=None, help="use built-in prompt: common|pointillism|used")
    ap.add_argument("--prompt-file", default=None, help="load prompt from a .py file defining system_prompt/user_prompt")

    # CSV column overrides (only used when input.mode=csv)
    ap.add_argument("--id-col", default=None, help="override input sample id column in CSV (defaults to input.sample_id_column)")
    ap.add_argument(
        "--path-col",
        default=None,
        help="override image path column in CSV (defaults to input.image_path_column). Can be omitted if using --input-dir to resolve path by id",
    )
    ap.add_argument(
        "--caption-col",
        default=None,
        help="override caption column in CSV (used by evaluation tasks). Defaults to input.caption_column or output.caption_column",
    )

    # api overrides (Andes gateway style)
    ap.add_argument("--api-base-url", default=None, help="override api.base_url")
    ap.add_argument("--api-app-id", default=None, help="override api.app_id")
    ap.add_argument("--api-model", default=None, help="override api.model")
    ap.add_argument("--api-secret-key", default=None, help="override api.secret_key (prefer env in prod)")
    ap.add_argument("--api-secret-key-env", default="ANDES_SK", help="env var name to read secret key from when missing")
    ap.add_argument("--api-timeout-sec", type=int, default=None, help="override api.timeout_sec")
    ap.add_argument("--api-retry-on-code", type=int, default=None, help="override api.retry_on_code")
    ap.add_argument("--api-retry-sleep-sec", type=int, default=None, help="override api.retry_sleep_sec")
    ap.add_argument("--api-use-reasoning", default=None, help="override api.useReasoning (true/false)")
    ap.add_argument("--api-temperature", type=float, default=None, help="override api.temperature")
    ap.add_argument("--api-top-p", type=float, default=None, help="override api.topP")

    # run overrides
    ap.add_argument("--start", type=int, default=None, help="override run.start")
    ap.add_argument("--end", type=int, default=None, help="override run.end")
    ap.add_argument("--image-detail", default=None, help="override run.image_detail (low|high|auto)")
    ap.add_argument(
        "--non-retriable-api-codes",
        default=None,
        help="comma-separated api codes treated as non-retriable (default from config.run.non_retriable_api_codes, fallback -30002)",
    )
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8")) or {}

    input_cfg = dict(cfg.get("input", {}) or {})
    run_cfg = dict(cfg.get("run", {}) or {})
    prompt_cfg = dict(cfg.get("prompt", {}) or {})
    api_cfg = dict(cfg.get("api", {}) or {})
    out_cfg = dict(cfg.get("output", {}) or {})

    if args.input_dir:
        input_cfg["image_dir"] = args.input_dir
    if args.input_csv:
        input_cfg["input_csv"] = args.input_csv
        input_cfg["mode"] = "csv"
    if args.output_csv:
        out_cfg["output_csv"] = args.output_csv
    if args.output_jsonl is not None:
        out_cfg["output_jsonl"] = str(args.output_jsonl)
    if args.output_dir:
        # if caller didn't explicitly set output file names, put them under this directory
        od = str(args.output_dir).strip()
        if od:
            od_p = Path(od)
            # csv: only override when --output-csv not explicitly provided
            if not args.output_csv:
                cur = str(out_cfg.get("output_csv", "") or "").strip() or "captions.csv"
                out_cfg["output_csv"] = str(od_p / Path(cur).name)
            # jsonl: only override when --output-jsonl not explicitly provided
            if args.output_jsonl is None:
                curj = str(out_cfg.get("output_jsonl", "") or "").strip() or "captions.jsonl"
                out_cfg["output_jsonl"] = str(od_p / Path(curj).name)

    # Apply API overrides
    def _set_if(flag_v, key: str):
        if flag_v is None:
            return
        api_cfg[key] = flag_v

    _set_if(args.api_base_url, "base_url")
    _set_if(args.api_app_id, "app_id")
    _set_if(args.api_model, "model")
    _set_if(args.api_timeout_sec, "timeout_sec")
    _set_if(args.api_retry_on_code, "retry_on_code")
    _set_if(args.api_retry_sleep_sec, "retry_sleep_sec")
    _set_if(args.api_temperature, "temperature")
    _set_if(args.api_top_p, "topP")
    if args.api_use_reasoning is not None:
        v = str(args.api_use_reasoning).strip().lower()
        if v in ["1", "true", "yes", "y", "on"]:
            api_cfg["useReasoning"] = True
        elif v in ["0", "false", "no", "n", "off"]:
            api_cfg["useReasoning"] = False
        else:
            raise RuntimeError("--api-use-reasoning must be true/false")
    if args.api_secret_key is not None:
        api_cfg["secret_key"] = str(args.api_secret_key)
    # Env fallback for secret key when empty
    if not str(api_cfg.get("secret_key", "") or "").strip():
        env_name = str(args.api_secret_key_env or "ANDES_SK").strip()
        if env_name:
            api_cfg["secret_key"] = os.environ.get(env_name, "")

    # Apply run overrides
    if args.start is not None:
        run_cfg["start"] = int(args.start)
    if args.end is not None:
        run_cfg["end"] = int(args.end)
    if args.image_detail is not None:
        run_cfg["image_detail"] = str(args.image_detail)

    # Determine task:
    # - CLI --task > config.run.task > infer from prompt-name > caption
    task = _norm_task(args.task if args.task is not None else run_cfg.get("task", None))
    pn = _norm_task(args.prompt_name)
    if (args.task is None) and (task == "caption") and (pn in ["semantic_adherence", "structural_plausibility", "style_match", "category_classify"]):
        task = pn

    # Non-retriable api codes: avoid endless retries on content safety, etc.
    non_retriable_codes: set[int] = set()
    cfg_codes = run_cfg.get("non_retriable_api_codes", None)
    if isinstance(cfg_codes, list):
        for x in cfg_codes:
            try:
                non_retriable_codes.add(int(x))
            except Exception:
                pass
    # default fallback if nothing configured
    if not non_retriable_codes:
        non_retriable_codes.add(-30002)
    if args.non_retriable_api_codes is not None:
        non_retriable_codes = set()
        for part in str(args.non_retriable_api_codes).split(","):
            part = part.strip()
            if not part:
                continue
            non_retriable_codes.add(int(part))

    mode = str(input_cfg.get("mode", "dir")).strip().lower()

    # Build input df
    if mode == "csv":
        in_csv = str(input_cfg.get("input_csv", "")).strip()
        if not in_csv:
            raise RuntimeError("input.mode=csv but input.input_csv is empty")
        df = read_csv(in_csv)

        # Determine columns
        image_col = str(args.path_col or input_cfg.get("image_path_column", "path")).strip()
        id_src_col = str(args.id_col or input_cfg.get("sample_id_column", "")).strip()
        caption_src_col = str(
            args.caption_col
            or input_cfg.get("caption_column", "")
            or out_cfg.get("caption_column", "caption")
            or "caption"
        ).strip()

        # Ensure canonical id/path columns exist.
        if "path" not in df.columns:
            if image_col and image_col in df.columns:
                df["path"] = df[image_col]
            else:
                df["path"] = ""

        if "id" not in df.columns:
            if id_src_col and id_src_col in df.columns:
                # normalize numeric ids like 12.0 -> "12"
                df["id"] = df[id_src_col].apply(_as_id_cell)
            else:
                # fallback to path stem when available
                df["id"] = df["path"].astype(str).apply(lambda x: Path(x).stem if str(x).strip() else "")
        else:
            # If caller explicitly specifies --id-col and it exists, override df["id"] for matching.
            # This avoids the common case where CSV already has a human-readable `id` column,
            # but matching to input-dir should use another column (e.g. valid_id like 0001.png).
            if args.id_col is not None and id_src_col and (id_src_col in df.columns):
                if id_src_col != "id" and "__orig_id" not in df.columns:
                    df["__orig_id"] = df["id"].astype(str)
                df["id"] = df[id_src_col].apply(_as_id_cell)
            else:
                df["id"] = df["id"].apply(_as_id_cell)

        # If evaluation tasks: need caption column
        if task in ["semantic_adherence", "structural_plausibility"]:
            if caption_src_col not in df.columns:
                raise RuntimeError(f"Missing caption column in input CSV: {caption_src_col} (use --caption-col or input.caption_column)")

        # If no path provided in CSV, allow resolving via input.image_dir by matching id -> file stem.
        need_resolve = df["path"].apply(_is_empty_cell).any()
        image_dir = str(input_cfg.get("image_dir", "") or "").strip()
        if need_resolve and image_dir:
            scan_df = _scan_dir(
                image_dir,
                recursive=bool(input_cfg.get("recursive", True)),
                glob_pat=str(input_cfg.get("glob", "*.*")),
                exts=list(input_cfg.get("exts", [".jpg", ".jpeg", ".png", ".webp", ".gif"])),
            )
            id_to_path: dict[str, str] = {}
            for _, r in scan_df.iterrows():
                sid = str(r.get("id", "")).strip()
                sp = str(r.get("path", "")).strip()
                if sid and sp and sid not in id_to_path:
                    id_to_path[sid] = sp
            # Fill missing path values
            def _fill_path(row) -> str:
                cur = _as_str_cell(row.get("path", ""))
                if cur:
                    return cur
                sid = _as_id_cell(row.get("id", ""))
                if not sid:
                    return ""
                # allow id with extension
                if sid in id_to_path:
                    return id_to_path[sid]
                stem = Path(sid).stem
                return id_to_path.get(stem, "")

            df["path"] = df.apply(_fill_path, axis=1)
        # normalize path to safe strings (avoid "nan")
        df["path"] = df["path"].apply(_as_str_cell)

    else:
        image_dir = str(input_cfg.get("image_dir", "")).strip()
        if not image_dir:
            raise RuntimeError("input.mode=dir but input.image_dir is empty")
        df = _scan_dir(
            image_dir,
            recursive=bool(input_cfg.get("recursive", True)),
            glob_pat=str(input_cfg.get("glob", "*.*")),
            exts=list(input_cfg.get("exts", [".jpg", ".jpeg", ".png", ".webp", ".gif"])),
        )

    # Slice
    start = int(run_cfg.get("start", 0) or 0)
    end = run_cfg.get("end", None)
    if end is not None:
        end = int(end)
        df = df.iloc[start:end].reset_index(drop=True)
    else:
        df = df.iloc[start:].reset_index(drop=True)

    # Output paths / resume
    out_csv_path = str(out_cfg.get("output_csv", "outputs/captions.csv"))
    out_csv_p = Path(out_csv_path)
    out_csv_p.parent.mkdir(parents=True, exist_ok=True)

    caption_col = str(out_cfg.get("caption_column", "caption"))
    raw_col = str(out_cfg.get("raw_response_column", "__raw_response"))
    err_col = str(out_cfg.get("error_column", "__error"))

    # Eval columns:
    # - If explicitly configured, respect it.
    # - Otherwise default to task-specific columns so multiple tasks can share one CSV safely.
    prefix = _task_prefix(task)
    score_col = str(out_cfg.get("score_column", "") or "").strip() or f"{prefix}_score"
    reason_col = str(out_cfg.get("short_reason_column", "") or "").strip() or f"{prefix}_short_reason"

    task_col = str(out_cfg.get("task_column", "") or "").strip() or "task"

    # style match columns (default to explicit, task-specific names; can be overridden in YAML)
    style_base_col = str(out_cfg.get("style_match_base_column", "") or "").strip() or f"{prefix}_base_json"
    style_stats_col = str(out_cfg.get("style_match_stats_column", "") or "").strip() or f"{prefix}_stats_json"
    style_score_col = str(out_cfg.get("style_match_score_column", "") or "").strip() or f"{prefix}_scalar_score"

    # category classify column
    category_col = str(out_cfg.get("category_column", "") or "").strip() or f"{prefix}_category"

    # caption input column (for evaluation tasks)
    caption_in_col = str(
        args.caption_col
        or input_cfg.get("caption_column", "")
        or out_cfg.get("caption_column", "caption")
        or "caption"
    ).strip()

    def _resolve_missing_paths_inplace(df_in: pd.DataFrame) -> pd.DataFrame:
        """
        If df has empty `path`, and input.image_dir is provided, resolve by matching id -> scanned file stem.
        Works for both fresh runs and resume runs.
        """

        if "path" not in df_in.columns or "id" not in df_in.columns:
            return df_in
        need_resolve = df_in["path"].apply(_is_empty_cell).any()
        image_dir = str(input_cfg.get("image_dir", "") or "").strip()
        if (not need_resolve) or (not image_dir):
            return df_in
        scan_df = _scan_dir(
            image_dir,
            recursive=bool(input_cfg.get("recursive", True)),
            glob_pat=str(input_cfg.get("glob", "*.*")),
            exts=list(input_cfg.get("exts", [".jpg", ".jpeg", ".png", ".webp", ".gif"])),
        )
        id_to_path: dict[str, str] = {}
        for _, r in scan_df.iterrows():
            sid = str(r.get("id", "")).strip()
            sp = str(r.get("path", "")).strip()
            if sid and sp and sid not in id_to_path:
                id_to_path[sid] = sp

        def _fill_path(row) -> str:
            cur = _as_str_cell(row.get("path", ""))
            if cur:
                return cur
            sid = _as_id_cell(row.get("id", ""))
            if not sid:
                return ""
            if sid in id_to_path:
                return id_to_path[sid]
            stem = Path(sid).stem
            return id_to_path.get(stem, "")

        df_in["path"] = df_in.apply(_fill_path, axis=1)
        df_in["path"] = df_in["path"].apply(_as_str_cell)
        return df_in

    if (not bool(args.no_resume)) and out_csv_p.exists():
        df = read_csv(str(out_csv_p))
        # ensure required cols exist
        if "id" not in df.columns or "path" not in df.columns:
            raise RuntimeError(f"Resume output_csv missing required columns: {out_csv_p}")
        # If resuming in CSV mode and caller passes --id-col, re-apply id override for matching.
        if mode == "csv":
            id_src_col = str(args.id_col or input_cfg.get("sample_id_column", "")).strip()
            if args.id_col is not None and id_src_col and (id_src_col in df.columns):
                if id_src_col != "id" and "__orig_id" not in df.columns:
                    df["__orig_id"] = df["id"].astype(str)
                df["id"] = df[id_src_col].apply(_as_id_cell)
        df = _resolve_missing_paths_inplace(df)

    # Ensure output columns exist (supports using the same CSV across tasks by adding new columns)
    base_cols = [caption_col, raw_col, err_col]
    if task in ["semantic_adherence", "structural_plausibility"]:
        base_cols.extend([score_col, reason_col])
    if task == "style_match":
        base_cols.extend([style_base_col, style_stats_col, style_score_col])
    if task == "category_classify":
        base_cols.append(category_col)
    base_cols.append(task_col)
    for c in base_cols:
        if c not in df.columns:
            df[c] = ""

    # Normalize id/path to safe strings (avoid 'nan' path and improve id matching)
    df["id"] = df["id"].apply(_as_id_cell)
    df["path"] = df["path"].apply(_as_str_cell)

    # Ensure string-like output columns are object dtype and have no NaN (avoid pandas dtype warnings)
    string_cols = [caption_col, raw_col, err_col]
    if task in ["semantic_adherence", "structural_plausibility"]:
        string_cols.append(reason_col)
    if task == "style_match":
        string_cols.extend([style_base_col, style_stats_col])
    for c in string_cols:
        if c in df.columns:
            df[c] = df[c].astype(object)
            df[c] = df[c].apply(_as_str_cell)

    # Ensure required input columns exist for the chosen task (for resume runs too)
    if task in ["semantic_adherence", "structural_plausibility"]:
        if caption_in_col not in df.columns:
            raise RuntimeError(f"Missing caption column in CSV for task={task}: {caption_in_col} (use --caption-col)")
    if task == "category_classify":
        if caption_in_col not in df.columns:
            raise RuntimeError(f"Missing caption column in CSV for task={task}: {caption_in_col} (use --caption-col)")

    # Client
    client = build_mllm_client(api_cfg)

    # Prompts (inline / prompt file / built-in name)
    # root_dir = <MLLM_Extract root>
    root_dir = str(Path(__file__).resolve().parents[3])
    system_prompt, user_prompt = resolve_prompt(
        root_dir=root_dir,
        prompt_cfg=prompt_cfg,
        prompt_name=args.prompt_name,
        prompt_file=args.prompt_file,
    )
    read_images = bool(run_cfg.get("read_images", True))
    if task == "category_classify":
        # Text-only task: do not require images.
        read_images = False
    image_detail = str(run_cfg.get("image_detail", "auto"))

    threads = _clamp_threads(args.threads, default=1)
    retries = max(0, int(args.retries))
    write_every_n = max(1, int(run_cfg.get("write_every_n", 20)))

    # JSONL (optional)
    out_jsonl_path = str(out_cfg.get("output_jsonl", "") or "").strip()
    jsonl_enabled = bool(out_jsonl_path)
    jsonl_include_raw = bool(out_cfg.get("include_raw_in_jsonl", True))
    out_jsonl_p = Path(out_jsonl_path) if jsonl_enabled else None
    if out_jsonl_p:
        out_jsonl_p.parent.mkdir(parents=True, exist_ok=True)

    def _append_jsonl(rec: dict) -> None:
        if not out_jsonl_p:
            return
        with open(out_jsonl_p, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Logging
    def _ts() -> str:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    def _log(msg: str) -> None:
        print(f"[{_ts()}] {msg}", flush=True)

    _log(f"run_images start: total={len(df)} threads={threads} retries={retries} resume={not bool(args.no_resume)} out_csv={out_csv_p}")
    _log(f"task={task}")

    def _should_skip_idx(idx: int) -> bool:
        """
        Resume policy:
        - Caption task: skip only when caption is non-empty AND error is empty.
        - Eval task: skip only when score is non-empty AND error is empty.
        - Otherwise (target empty/NaN, or error non-empty), retry.
        - Exception: if error is marked non-retriable, skip.
        """
        if bool(args.no_resume):
            return False
        err_val = df.at[idx, err_col] if err_col in df.columns else None
        if isinstance(err_val, str) and err_val.strip().lower().startswith("non_retriable:"):
            return True
        if task in ["semantic_adherence", "structural_plausibility"]:
            target_ok = not _is_empty_cell(df.at[idx, score_col] if score_col in df.columns else None)
        elif task == "style_match":
            target_ok = not _is_empty_cell(df.at[idx, style_score_col] if style_score_col in df.columns else None)
        elif task == "category_classify":
            target_ok = not _is_empty_cell(df.at[idx, category_col] if category_col in df.columns else None)
        else:
            target_ok = not _is_empty_cell(df.at[idx, caption_col] if caption_col in df.columns else None)
        err_empty = _is_empty_cell(err_val)
        return target_ok and err_empty

    # Serial
    if threads <= 1:
        for idx in tqdm(range(len(df)), desc=task):
            if _should_skip_idx(idx):
                continue
            path = _as_str_cell(df.at[idx, "path"])
            img_url = None
            if read_images and (not path.strip()):
                prev = _as_str_cell(df.at[idx, err_col])
                df.at[idx, err_col] = (prev + " | " if prev else "") + "missing_image_path"
                continue
            if read_images and path.strip():
                try:
                    img_url = load_image_as_data_url(path)
                except Exception as e:
                    prev = _as_str_cell(df.at[idx, err_col])
                    df.at[idx, err_col] = (prev + " | " if prev else "") + f"image_load_error: {e}"
                    # Without image, caption would be meaningless; mark as failed and move on.
                    continue
            sid = str(df.at[idx, "id"])
            cap_in = ""
            if task in ["semantic_adherence", "structural_plausibility", "category_classify"]:
                cap_in = str(df.at[idx, caption_in_col] if caption_in_col in df.columns else "")
                if _is_empty_cell(cap_in):
                    prev = str(df.at[idx, err_col] or "")
                    df.at[idx, err_col] = (prev + " | " if prev else "") + f"missing_caption: col={caption_in_col}"
                    continue
                # keep a stable caption column in outputs for downstream usage
                if caption_col not in df.columns:
                    df[caption_col] = ""
                if _is_empty_cell(df.at[idx, caption_col]):
                    df.at[idx, caption_col] = cap_in
            rendered_user_prompt = _render_user_prompt(user_prompt, caption=cap_in, sample_id=sid)
            msgs = build_messages(system_prompt, rendered_user_prompt, image_data_url=img_url, image_detail=image_detail)
            try:
                non_retriable_hit = False
                for attempt in range(1, max(1, retries + 1) + 1):
                    try:
                        resp = client.chat_completions(msgs)
                        df.at[idx, raw_col] = _normalize_raw_response_for_csv(resp)
                        if task in ["semantic_adherence", "structural_plausibility"]:
                            score, reason = _parse_eval_json(resp)
                            score_i, reason_s = _validate_eval_output(score, reason)
                            df.at[idx, score_col] = int(score_i)
                            df.at[idx, reason_col] = str(reason_s)
                        elif task == "style_match":
                            base = parse_style_match_base_json(resp)
                            res = compute_style_match_score(base)
                            stats = build_style_match_stats_dict(res)
                            df.at[idx, style_base_col] = json.dumps(base, ensure_ascii=False)
                            df.at[idx, style_stats_col] = json.dumps(stats, ensure_ascii=False)
                            df.at[idx, style_score_col] = float(res.final_score_after_caps)
                        elif task == "category_classify":
                            cat = _parse_category_json(resp)
                            cat_v = _validate_category_output(cat)
                            df.at[idx, category_col] = cat_v
                        else:
                            cap = _normalize_caption(resp)
                            df.at[idx, caption_col] = cap
                        df.at[idx, task_col] = task
                        # success: clear previous errors
                        df.at[idx, err_col] = ""
                        break
                    except AndesAPIError as e:
                        if (e.api_code is not None) and (int(e.api_code) in non_retriable_codes):
                            # e.g. content safety blocked: do not retry forever
                            df.at[idx, raw_col] = ""
                            if task in ["semantic_adherence", "structural_plausibility"]:
                                df.at[idx, score_col] = ""
                                df.at[idx, reason_col] = ""
                            elif task == "style_match":
                                df.at[idx, style_base_col] = ""
                                df.at[idx, style_stats_col] = ""
                                df.at[idx, style_score_col] = ""
                            elif task == "category_classify":
                                df.at[idx, category_col] = ""
                            else:
                                df.at[idx, caption_col] = ""
                            df.at[idx, err_col] = f"non_retriable:api_code={e.api_code} api_msg={e.api_msg or ''}".strip()
                            non_retriable_hit = True
                            break
                    except Exception as e:
                        if attempt <= retries:
                            _sleep_backoff(attempt)
                        else:
                            raise
                if non_retriable_hit:
                    # write jsonl record if enabled (for downstream filtering), then continue
                    if jsonl_enabled:
                        rec = {"id": str(df.at[idx, "id"]), "path": path}
                        if task in ["semantic_adherence", "structural_plausibility"]:
                            rec["caption"] = str(df.at[idx, caption_col] or "")
                            rec["score"] = ""
                            rec["short_reason"] = ""
                        elif task == "style_match":
                            rec["style_match_base"] = ""
                            rec["style_match_stats"] = ""
                            rec["style_match_score"] = ""
                        elif task == "category_classify":
                            rec["caption"] = str(df.at[idx, caption_col] or "")
                            rec["category"] = ""
                        else:
                            rec["caption"] = ""
                        rec["task"] = task
                        if jsonl_include_raw:
                            rec["raw_response"] = ""
                        rec["error"] = str(df.at[idx, err_col] or "")
                        _append_jsonl(rec)
                    continue
                if jsonl_enabled:
                    rec = {"id": str(df.at[idx, "id"]), "path": path, "caption": str(df.at[idx, caption_col] or "")}
                    if task in ["semantic_adherence", "structural_plausibility"]:
                        rec["score"] = str(df.at[idx, score_col] or "")
                        rec["short_reason"] = str(df.at[idx, reason_col] or "")
                    elif task == "style_match":
                        rec["style_match_base"] = str(df.at[idx, style_base_col] or "")
                        rec["style_match_stats"] = str(df.at[idx, style_stats_col] or "")
                        rec["style_match_score"] = str(df.at[idx, style_score_col] or "")
                    elif task == "category_classify":
                        rec["category"] = str(df.at[idx, category_col] or "")
                    rec["task"] = task
                    if jsonl_include_raw:
                        rec["raw_response"] = str(df.at[idx, raw_col] or "")
                    err = str(df.at[idx, err_col] or "").strip()
                    if err:
                        rec["error"] = err
                    _append_jsonl(rec)
            except Exception as e:
                prev = _as_str_cell(df.at[idx, err_col])
                df.at[idx, err_col] = (prev + " | " if prev else "") + f"mllm_error: {e}"
            if (idx + 1) % write_every_n == 0:
                write_csv(df, str(out_csv_p))
        write_csv(df, str(out_csv_p))
        return

    # Parallel
    total = len(df)
    lock = threading.Lock()
    active: dict[int, str] = {}
    done = ok = fail = 0
    stop = threading.Event()
    interval = max(1, int(args.log_interval_s))
    t0 = time.time()

    def monitor() -> None:
        while not stop.wait(interval):
            with lock:
                d, o, f = done, ok, fail
                act = list(active.values())[:8]
            elapsed = max(1, int(time.time() - t0))
            head = f"run_images: {d}/{total} ok={o} fail={f} active={len(active)} elapsed={elapsed}s"
            if act:
                head += " | active=" + "; ".join(act)
            _log(head)

    mon_th = threading.Thread(target=monitor, daemon=True)
    mon_th.start()

    def worker(idx: int, row: dict) -> dict:
        tid = threading.get_ident()
        with lock:
            active[tid] = f"idx={idx}"
        out = {"idx": idx, "set": {}, "errs": []}
        try:
            if _should_skip_idx(idx):
                out["ok"] = True
                out["skipped"] = True
                return out
            path = _as_str_cell(row.get("path", ""))
            img_url = None
            if read_images and (not path.strip()):
                out["errs"].append("missing_image_path")
                out["ok"] = False
                return out
            if read_images and path.strip():
                try:
                    img_url = load_image_as_data_url(path)
                except Exception as e:
                    out["errs"].append(f"image_load_error: {e}")
                    out["ok"] = False
                    return out
            sid = str(row.get("id", ""))
            cap_in = ""
            if task in ["semantic_adherence", "structural_plausibility", "category_classify"]:
                cap_in = str(row.get(caption_in_col, ""))
                if _is_empty_cell(cap_in):
                    out["errs"].append(f"missing_caption: col={caption_in_col}")
                    out["ok"] = False
                    return out
                # keep caption column in output stable
                if caption_col != caption_in_col:
                    out["set"][caption_col] = cap_in
            rendered_user_prompt = _render_user_prompt(user_prompt, caption=cap_in, sample_id=sid)
            msgs = build_messages(system_prompt, rendered_user_prompt, image_data_url=img_url, image_detail=image_detail)
            resp = None
            last_err: str | None = None
            for attempt in range(1, max(1, retries + 1) + 1):
                try:
                    resp = client.chat_completions(msgs)
                    # Validate/parse within the retry loop so malformed outputs can retry.
                    if task in ["semantic_adherence", "structural_plausibility"]:
                        score, reason = _parse_eval_json(resp or "")
                        score_i, reason_s = _validate_eval_output(score, reason)
                        out["set"][raw_col] = _normalize_raw_response_for_csv(resp)
                        out["set"][score_col] = int(score_i)
                        out["set"][reason_col] = str(reason_s)
                        out["set"][err_col] = ""
                        out["set"][task_col] = task
                        out["ok"] = True
                        return out
                    if task == "style_match":
                        base = parse_style_match_base_json(resp or "")
                        res = compute_style_match_score(base)
                        stats = build_style_match_stats_dict(res)
                        out["set"][raw_col] = _normalize_raw_response_for_csv(resp)
                        out["set"][style_base_col] = json.dumps(base, ensure_ascii=False)
                        out["set"][style_stats_col] = json.dumps(stats, ensure_ascii=False)
                        out["set"][style_score_col] = float(res.final_score_after_caps)
                        out["set"][err_col] = ""
                        out["set"][task_col] = task
                        out["ok"] = True
                        return out
                    if task == "category_classify":
                        cat = _parse_category_json(resp or "")
                        cat_v = _validate_category_output(cat)
                        out["set"][raw_col] = _normalize_raw_response_for_csv(resp)
                        out["set"][category_col] = cat_v
                        out["set"][err_col] = ""
                        out["set"][task_col] = task
                        out["ok"] = True
                        return out
                    # caption task: keep existing behavior (do not enforce non-empty caption)
                    out["set"][raw_col] = _normalize_raw_response_for_csv(resp)
                    out["set"][caption_col] = _normalize_caption(resp or "")
                    out["set"][err_col] = ""
                    out["set"][task_col] = task
                    out["ok"] = True
                    return out
                except AndesAPIError as e:
                    if (e.api_code is not None) and (int(e.api_code) in non_retriable_codes):
                        out["set"][raw_col] = ""
                        if task in ["semantic_adherence", "structural_plausibility"]:
                            out["set"][score_col] = ""
                            out["set"][reason_col] = ""
                        elif task == "style_match":
                            out["set"][style_base_col] = ""
                            out["set"][style_stats_col] = ""
                            out["set"][style_score_col] = ""
                        elif task == "category_classify":
                            out["set"][category_col] = ""
                        else:
                            out["set"][caption_col] = ""
                        out["set"][err_col] = f"non_retriable:api_code={e.api_code} api_msg={e.api_msg or ''}".strip()
                        out["ok"] = True
                        out["blocked"] = True
                        return out
                    last_err = f"andes_api_error: {e}"
                    if attempt <= retries:
                        _sleep_backoff(attempt)
                except Exception as e:
                    last_err = f"mllm_error: {e}"
                    if attempt <= retries:
                        _sleep_backoff(attempt)
                    else:
                        raise
            # Exhausted retries without success/non-retriable return
            out["errs"].append(last_err or "retry_exhausted")
            out["ok"] = False
            return out
        except Exception as e:
            out["errs"].append(f"mllm_error: {e}")
            out["ok"] = False
            return out
        finally:
            with lock:
                active.pop(tid, None)

    try:
        with ThreadPoolExecutor(max_workers=threads) as ex:
            futs = [ex.submit(worker, idx, df.iloc[idx].to_dict()) for idx in range(total)]
            for j, fut in enumerate(as_completed(futs), start=1):
                res = fut.result()
                idx = int(res["idx"])
                for k, v in (res.get("set") or {}).items():
                    df.at[idx, k] = v
                if res.get("errs"):
                    prev = _as_str_cell(df.at[idx, err_col])
                    df.at[idx, err_col] = (prev + " | " if prev else "") + " | ".join(res["errs"])
                with lock:
                    done += 1
                    if res.get("ok"):
                        ok += 1
                    else:
                        fail += 1
                if jsonl_enabled and res.get("ok") and not res.get("skipped"):
                    rec = {"id": str(df.at[idx, "id"]), "path": str(df.at[idx, "path"]), "caption": str(df.at[idx, caption_col] or "")}
                    if task in ["semantic_adherence", "structural_plausibility"]:
                        rec["score"] = str(df.at[idx, score_col] or "")
                        rec["short_reason"] = str(df.at[idx, reason_col] or "")
                    elif task == "style_match":
                        rec["style_match_base"] = str(df.at[idx, style_base_col] or "")
                        rec["style_match_stats"] = str(df.at[idx, style_stats_col] or "")
                        rec["style_match_score"] = str(df.at[idx, style_score_col] or "")
                    if jsonl_include_raw:
                        rec["raw_response"] = str(df.at[idx, raw_col] or "")
                    err = str(df.at[idx, err_col] or "").strip()
                    if err:
                        rec["error"] = err
                    _append_jsonl(rec)
                if j % write_every_n == 0:
                    write_csv(df, str(out_csv_p))
    finally:
        stop.set()
        try:
            mon_th.join(timeout=1.0)
        except Exception:
            pass

    write_csv(df, str(out_csv_p))
    _log(f"run_images done. ok={ok} fail={fail} out_csv={out_csv_p}")


if __name__ == "__main__":
    main()


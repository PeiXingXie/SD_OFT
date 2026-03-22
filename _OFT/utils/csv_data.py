from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass(frozen=True)
class CsvExample:
    image_path: Path
    text: str
    negative_text: Optional[str] = None


def _resolve_image_path(raw: str, *, csv_path: Path, image_root: Optional[str]) -> Path:
    p = Path(raw)
    if p.is_absolute():
        return p
    if image_root is not None:
        return (Path(image_root) / p).resolve()
    return (csv_path.parent / p).resolve()


def read_csv_examples(
    csv_file: str,
    *,
    image_col: str,
    text_col: str,
    negative_col: Optional[str] = None,
    image_root: Optional[str] = None,
    delimiter: str = ",",
    limit: Optional[int] = None,
) -> List[CsvExample]:
    """
    Read examples from a CSV file.

    Required columns:
      - image_col: image path (absolute, or relative to image_root/csv parent)
      - text_col: prompt text

    Optional:
      - negative_col: negative prompt
    """
    path = Path(csv_file)
    if not path.exists():
        raise FileNotFoundError(f"csv_file not found: {path}")

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")

        cols = set(reader.fieldnames)
        for c in [image_col, text_col]:
            if c not in cols:
                raise ValueError(
                    f"CSV missing required column '{c}'. "
                    f"Available columns: {sorted(cols)}"
                )
        if negative_col is not None and negative_col not in cols:
            raise ValueError(
                f"CSV missing negative_col '{negative_col}'. "
                f"Available columns: {sorted(cols)}"
            )

        out: List[CsvExample] = []
        for row in reader:
            img_raw = (row.get(image_col) or "").strip()
            txt = (row.get(text_col) or "").strip()
            if not img_raw or not txt:
                # skip empty/incomplete rows
                continue

            img_path = _resolve_image_path(
                img_raw, csv_path=path, image_root=image_root
            )
            if not img_path.exists():
                raise FileNotFoundError(
                    f"Image not found: {img_path} (from '{img_raw}' in {path})"
                )

            neg = None
            if negative_col is not None:
                neg = (row.get(negative_col) or "").strip() or None

            out.append(CsvExample(image_path=img_path, text=txt, negative_text=neg))
            if limit is not None and len(out) >= int(limit):
                break

    if not out:
        raise ValueError(f"No valid rows found in CSV: {path}")
    return out


def read_csv_prompts(
    csv_file: str,
    *,
    text_col: str,
    negative_col: Optional[str] = None,
    delimiter: str = ",",
    limit: Optional[int] = None,
) -> List[Dict[str, Optional[str]]]:
    """
    Read prompt rows for validation generation.
    Returns list of dicts: {text, negative_text}.
    """
    path = Path(csv_file)
    if not path.exists():
        raise FileNotFoundError(f"csv_file not found: {path}")

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")

        cols = set(reader.fieldnames)
        if text_col not in cols:
            raise ValueError(
                f"CSV missing required column '{text_col}'. "
                f"Available columns: {sorted(cols)}"
            )
        if negative_col is not None and negative_col not in cols:
            raise ValueError(
                f"CSV missing negative_col '{negative_col}'. "
                f"Available columns: {sorted(cols)}"
            )

        out: List[Dict[str, Optional[str]]] = []
        for row in reader:
            txt = (row.get(text_col) or "").strip()
            if not txt:
                continue
            neg = None
            if negative_col is not None:
                neg = (row.get(negative_col) or "").strip() or None
            out.append({"text": txt, "negative_text": neg})
            if limit is not None and len(out) >= int(limit):
                break

    if not out:
        raise ValueError(f"No valid prompts found in CSV: {path}")
    return out


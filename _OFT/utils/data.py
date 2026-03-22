from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image


def _is_image(p: Path) -> bool:
    return p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def _load_caption_from_txt(txt_path: Path) -> str:
    return txt_path.read_text(encoding="utf-8").strip()


def _load_metadata_jsonl(jsonl_path: Path) -> List[Dict[str, str]]:
    items: List[Dict[str, str]] = []
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if not isinstance(obj, dict) or "file_name" not in obj or "text" not in obj:
            raise ValueError(
                f"Invalid jsonl line (need keys file_name,text): {line[:200]}"
            )
        items.append({"file_name": str(obj["file_name"]), "text": str(obj["text"])})
    return items


@dataclass(frozen=True)
class Example:
    image_path: Path
    text: str


def discover_examples(train_data_dir: str) -> List[Example]:
    """
    Discover training examples from a directory.

    Supported:
      - image + same-stem .txt caption
      - metadata.jsonl (with file_name + text)
    """
    root = Path(train_data_dir)
    if not root.exists():
        raise FileNotFoundError(f"train_data_dir not found: {root}")

    jsonl = root / "metadata.jsonl"
    if jsonl.exists():
        items = _load_metadata_jsonl(jsonl)
        ex: List[Example] = []
        for it in items:
            img = (root / it["file_name"]).resolve()
            if not img.exists():
                raise FileNotFoundError(f"Image not found: {img}")
            ex.append(Example(image_path=img, text=it["text"]))
        if not ex:
            raise ValueError(f"No examples found in metadata: {jsonl}")
        return ex

    # image + caption txt
    images = sorted([p for p in root.rglob("*") if p.is_file() and _is_image(p)])
    if not images:
        raise ValueError(
            f"No images found under: {root} (expected png/jpg/webp/bmp, or metadata.jsonl)"
        )
    ex2: List[Example] = []
    for img in images:
        txt = img.with_suffix(".txt")
        if not txt.exists():
            raise FileNotFoundError(f"Missing caption txt for image: {img} -> {txt}")
        caption = _load_caption_from_txt(txt)
        if not caption:
            raise ValueError(f"Empty caption: {txt}")
        ex2.append(Example(image_path=img, text=caption))
    return ex2


def build_transforms(
    *,
    resolution: int,
    center_crop: bool,
    random_flip: bool,
):
    import torchvision.transforms as T

    tx: List[object] = [
        T.Resize(resolution, interpolation=T.InterpolationMode.BILINEAR),
    ]
    if center_crop:
        tx.append(T.CenterCrop(resolution))
    else:
        tx.append(T.RandomCrop(resolution))
    if random_flip:
        tx.append(T.RandomHorizontalFlip())

    # to [-1, 1] float tensor
    tx.extend(
        [
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    return T.Compose(tx)


class TextImageDataset:
    def __init__(
        self,
        examples: List[Example],
        *,
        resolution: int = 512,
        center_crop: bool = True,
        random_flip: bool = True,
    ):
        self.examples = examples
        self.tx = build_transforms(
            resolution=resolution, center_crop=center_crop, random_flip=random_flip
        )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        ex = self.examples[int(idx)]
        image = Image.open(ex.image_path).convert("RGB")
        pixel_values = self.tx(image)
        return {
            "pixel_values": pixel_values,
            "text": ex.text,
            "image_path": str(ex.image_path),
        }


def collate_fn(batch: List[Dict[str, object]]) -> Dict[str, object]:
    import torch

    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    texts = [b["text"] for b in batch]
    return {"pixel_values": pixel_values, "texts": texts}


class CsvTextImageDataset:
    """
    Same as TextImageDataset but accepts CsvExample from utils/csv_data.py.
    """

    def __init__(
        self,
        examples,
        *,
        resolution: int = 512,
        center_crop: bool = True,
        random_flip: bool = True,
    ):
        self.examples = examples
        self.tx = build_transforms(
            resolution=resolution, center_crop=center_crop, random_flip=random_flip
        )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        ex = self.examples[int(idx)]
        image = Image.open(ex.image_path).convert("RGB")
        pixel_values = self.tx(image)
        return {
            "pixel_values": pixel_values,
            "text": ex.text,
            "negative_text": getattr(ex, "negative_text", None),
            "image_path": str(ex.image_path),
        }


def csv_collate_fn(batch: List[Dict[str, object]]) -> Dict[str, object]:
    import torch

    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    texts = [b["text"] for b in batch]
    negative_texts = [b.get("negative_text") for b in batch]
    return {"pixel_values": pixel_values, "texts": texts, "negative_texts": negative_texts}


"""Dataset scanning and the FROZEN unified preprocessing for all methods.

Protocol (locked, do not change after sealed evaluation):
  * whole image resized to 256x256, aspect ratio NOT preserved, no crop, no pad
  * PIL ``Image.BILINEAR`` -- identical resampling for every method and for GT
  * both the prediction and the reference are resized with the same call
  * metrics computed per image, then averaged (never average-then-metric)

Native-resolution and official-protocol results are kept as a *supplementary*
track only; they must never be mixed into the main table.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")

#: canonical dataset -> (subdir under the dataset root, split)
DATASETS: Dict[str, str] = {
    "UIEB": "UIEB/test",
    "LSUI": "LSUI/test",
    "UFO": "UFO/test",
    "UIEB-VAL": "UIEB/val",
    "LSUI-VAL": "LSUI/val",
    "UFO-VAL": "UFO/val",
}


def is_image(name: str) -> bool:
    return name.lower().endswith(IMG_EXT)


def list_images(root: str) -> List[str]:
    return sorted(f for f in os.listdir(root) if is_image(f))


def stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def sha256_of(path: str, blocksize: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(blocksize)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def load_rgb(path: str) -> np.ndarray:
    """uint8 RGB HxWx3."""
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


def to_unified(arr: np.ndarray, size: int = 256) -> np.ndarray:
    """FROZEN: whole image -> (size, size), PIL BILINEAR, no crop/pad."""
    im = Image.fromarray(np.asarray(arr, dtype=np.uint8), mode="RGB")
    return np.asarray(im.resize((size, size), Image.BILINEAR), dtype=np.uint8)


@dataclass
class Pair:
    name: str
    input_path: str
    ref_path: str


@dataclass
class PairSet:
    name: str
    pairs: List[Pair] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.pairs)

    def names(self) -> List[str]:
        return [p.name for p in self.pairs]

    def manifest(self, hash_files: bool = False) -> dict:
        out = {
            "name": self.name,
            "n_pairs": len(self.pairs),
            "notes": self.notes,
            "names": self.names(),
        }
        if hash_files:
            out["sha256"] = {
                p.name: [sha256_of(p.input_path), sha256_of(p.ref_path)]
                for p in self.pairs
            }
        return out


def _index_by_variant(files: List[str]) -> Dict[str, str]:
    """Map several normalised spellings of a stem to the actual filename."""
    idx: Dict[str, str] = {}
    for f in files:
        s = stem(f)
        for key in _variants(s):
            idx.setdefault(key, f)
    return idx


def _variants(s: str) -> List[str]:
    out = [s, s.lower()]
    for suf in ("_img_", "_img", "_pred", "_enhanced", "_ref", "_gt", "_h"):
        if s.endswith(suf):
            out.append(s[: -len(suf)])
            out.append(s[: -len(suf)].lower())
    lead = "".join(ch for ch in s if ch.isdigit())
    if lead:
        out.append(lead)
    return out


def resolve_pairs(input_dir: str, ref_dir: str, name: str) -> PairSet:
    """Pair inputs with references tolerating extension / suffix differences."""
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(input_dir)
    if not os.path.isdir(ref_dir):
        raise FileNotFoundError(ref_dir)
    ins = list_images(input_dir)
    refs = list_images(ref_dir)
    ref_idx = _index_by_variant(refs)
    ps: List[Pair] = []
    missing: List[str] = []
    for f in ins:
        target = None
        for key in _variants(stem(f)):
            if key in ref_idx:
                target = ref_idx[key]
                break
        if target is None:
            missing.append(f)
            continue
        ps.append(Pair(stem(f), os.path.join(input_dir, f), os.path.join(ref_dir, target)))
    notes = [
        f"inputs={len(ins)} refs={len(refs)} matched={len(ps)} unmatched={len(missing)}",
        f"input_dir={input_dir}",
        f"ref_dir={ref_dir}",
    ]
    if missing:
        notes.append(f"UNMATCHED_SAMPLE={missing[:8]}")
    return PairSet(name=name, pairs=ps, notes=notes)


def probe(root: str, maxdepth: int = 2) -> List[str]:
    """Human-readable shallow tree; used for layout audits."""
    lines: List[str] = []
    root = os.path.abspath(root)
    base_depth = root.rstrip(os.sep).count(os.sep)
    for dirpath, dirnames, filenames in os.walk(root):
        depth = dirpath.rstrip(os.sep).count(os.sep) - base_depth
        if depth > maxdepth:
            dirnames[:] = []
            continue
        dirnames.sort()
        rel = os.path.relpath(dirpath, root)
        n_img = sum(1 for f in filenames if is_image(f))
        other = len(filenames) - n_img
        lines.append(f"{'  ' * depth}{rel}/  imgs={n_img} other={other}")
    return lines

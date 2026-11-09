"""Frozen-protocol evaluation shared by the ablation trainer and the report tools.

Why a separate module
---------------------
``code/tools/run_baseline.py`` contains the authoritative ``score()`` used for
every number in this repository, but importing it drags in seven third-party
model builders and pins ``torch.set_num_threads``.  The ablation trainer needs
the *same numbers* without that baggage, so the scoring is re-implemented here
and then **checked against the original** by ``check_agrees_with_run_baseline``.

If those two ever disagree, this module is wrong.  ``python -m obsui.ablate_eval``
runs the check.

Protocol (frozen, identical to ``docs/research_plan.md`` section 3)
-------------------------------------------------------------------
* whole image -> 256x256, PIL ``Image.BILINEAR``, no crop, no pad, no aspect
  preservation (``dataio.to_unified``);
* model I/O in float ``[0, 1]``;
* PSNR on float ``[0, 1]`` with ``data_range=1.0`` (identical to
  ``metrics.psnr(..., data_range=255.0)`` on uint8);
* SSIM = ``skimage.metrics.structural_similarity(channel_axis=-1, win_size=5,
  data_range=1.0)``;
* per-image metrics first, then the arithmetic mean over images;
* every image counts -- nothing is dropped for scoring badly.
"""
from __future__ import annotations

import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from skimage.metrics import structural_similarity as sk_ssim

from obsui import dataio, metrics

SK_SSIM_WIN = 5
DATA_ROOT_DEFAULT = "/home/user/UnderWaterDataset"


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #
def score_pair(pred_u8: np.ndarray, ref_u8: np.ndarray) -> Dict[str, float]:
    """Frozen per-image metrics for one already-256x256 uint8 RGB pair."""
    p = ref_u8.astype(np.float64) / 255.0
    q = pred_u8.astype(np.float64) / 255.0
    return {
        "psnr": float(metrics.psnr(pred_u8, ref_u8, data_range=255.0)),
        "ssim": float(sk_ssim(p, q, channel_axis=-1, win_size=SK_SSIM_WIN, data_range=1.0)),
    }


def aggregate(rows: Sequence[Dict[str, float]]) -> Dict[str, float]:
    if not rows:
        raise ValueError("no rows to aggregate")
    out: Dict[str, float] = {"n": len(rows)}
    for k in ("psnr", "ssim"):
        v = np.array([r[k] for r in rows], dtype=np.float64)
        out[k] = float(v.mean())
        out[k + "_std"] = float(v.std(ddof=1)) if len(v) > 1 else 0.0
    return out


# --------------------------------------------------------------------------- #
# inference
# --------------------------------------------------------------------------- #
def to_tensor(u8: np.ndarray) -> torch.Tensor:
    """HxWx3 uint8 -> 1x3xHxW float in [0, 1] (the backbone's input convention)."""
    x = torch.from_numpy(np.ascontiguousarray(u8)).float().div_(255.0)
    return x.permute(2, 0, 1).unsqueeze(0)


def to_uint8(t: torch.Tensor) -> np.ndarray:
    """Model output in [0, 1] -> HxWx3 uint8 at its native size."""
    x = t.detach().float().cpu()
    if x.dim() == 4:
        x = x[0]
    x = x.clamp(0.0, 1.0)
    arr = (x * 255.0).round().clamp(0, 255).byte().permute(1, 2, 0).numpy()
    return np.ascontiguousarray(arr, dtype=np.uint8)


def load_unified(split_root: str, names: Sequence[str]) -> List[np.ndarray]:
    """Read and freeze-resize one split's inputs (references handled separately)."""
    return [dataio.to_unified(dataio.load_rgb(os.path.join(split_root, n))) for n in names]


def make_pairs(dataset: str, split: str,
               data_root: str = DATA_ROOT_DEFAULT) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    """(name, input_256, reference_256) for one split, matched by stem.

    ``resolve_pairs`` tolerates suffix/extension differences and is the same
    matcher the baseline runner uses; the sorted-index pairing that MambaCue's
    own loader uses was separately verified to agree on 0 % of files for
    UIEB/LSUI train/val/test, so the two loaders see the same pairs.
    """
    root = os.path.join(data_root, dataset, split)
    ps = dataio.resolve_pairs(os.path.join(root, "input"), os.path.join(root, "GT"), dataset)
    if not ps.pairs:
        raise SystemExit(f"no pairs under {root}")
    out = []
    for p in ps.pairs:
        a = dataio.to_unified(dataio.load_rgb(p.input_path))
        b = dataio.to_unified(dataio.load_rgb(p.ref_path))
        out.append((p.name, a, b))
    return out


@torch.no_grad()
def evaluate_model(model: torch.nn.Module,
                   pairs: Sequence[Tuple[str, np.ndarray, np.ndarray]],
                   device: torch.device,
                   batch: int = 8) -> List[Dict[str, float]]:
    """Per-image rows for one model over one split. Model must already be in eval()."""
    was_training = model.training
    model.eval()
    rows: List[Dict[str, float]] = []
    try:
        for i in range(0, len(pairs), batch):
            chunk = pairs[i:i + batch]
            xs = torch.cat([to_tensor(a) for _, a, _ in chunk], dim=0).to(device)
            preds = model(xs, prior_map=None, return_aux=False)
            if isinstance(preds, (tuple, list)):
                preds = preds[0]
            preds = preds.clamp(0.0, 1.0).float().cpu()
            for j, (name, _a, ref) in enumerate(chunk):
                m = score_pair(to_uint8(preds[j]), ref)
                m["name"] = name
                rows.append(m)
    finally:
        if was_training:
            model.train()
    return rows


# --------------------------------------------------------------------------- #
# agreement check against the authoritative implementation
# --------------------------------------------------------------------------- #
def check_agrees_with_run_baseline(trials: int = 12, seed: int = 0) -> None:
    """Assert ``score_pair`` reproduces ``run_baseline.score`` bit-for-bit.

    This is the only thing that licenses using this module instead of the
    original for a reported number.
    """
    import sys

    tools = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools")
    if tools not in sys.path:
        sys.path.insert(0, tools)
    import importlib.util

    spec = importlib.util.spec_from_file_location("_run_baseline", os.path.join(tools, "run_baseline.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # module-level code only sets torch threads

    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(trials):
        ref = rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8)
        noise = rng.normal(0.0, 18.0, size=ref.shape)
        pred = np.clip(ref.astype(np.float64) + noise, 0, 255).astype(np.uint8)
        mine = score_pair(pred, ref)
        theirs = mod.score(pred, ref)
        for k in ("psnr", "ssim"):
            d = abs(mine[k] - theirs[k])
            worst = max(worst, d)
            if d > 1e-9:
                raise AssertionError(
                    f"score_pair disagrees with run_baseline.score on {k}: "
                    f"{mine[k]!r} vs {theirs[k]!r} (|d|={d:.3e})")
    print(f"[ablate_eval] agrees with run_baseline.score on {trials} synthetic pairs (max|d|={worst:.2e})")


if __name__ == "__main__":
    check_agrees_with_run_baseline()

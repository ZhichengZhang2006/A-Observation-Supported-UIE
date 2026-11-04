"""Pack a MambaCue run's side-by-side prediction tiles into one npz per dataset.

Each ``pred/<name>.png`` produced by MambaCue is ``hstack(raw, gt, pred)`` at
256x256 per panel.  We split it with ``t = w // 3`` (the UFO tiles are 240x960,
so ``w == 3*h`` must NOT be assumed) and store uint8 arrays plus names.

Output: ``<out>/<DATASET>.npz`` with keys
    raw   (N,256,256,3) uint8   degraded input, unified PIL-bilinear resize
    gt    (N,256,256,3) uint8   reference, same resize
    pred  (N,256,256,3) uint8   MambaCue output
    names (N,) <U64
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
from PIL import Image

RUNS = {
    "UIEB": "output/L1+SSIM+Prep+Lab+grad+Angle/four/color-illumination-contrast/both/UIEB/20261009_092226",
    "LSUI": "output/L1+SSIM+Prep+Lab+grad+Angle/four/color-illumination-contrast/both/LSUI/20261009_112952",
    "UFO": "output/L1+SSIM+Prep+Lab+grad+Angle/four/color-illumination-contrast/both/UFO/20261010_020343",
}


def split_tile(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    h, w = arr.shape[:2]
    t = w // 3
    if t <= 0:
        raise ValueError(f"degenerate tile {path} shape={arr.shape}")
    return arr[:, :t], arr[:, t : 2 * t], arr[:, 2 * t : 3 * t]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mambacue", default="/home/user/MambaCue")
    ap.add_argument("--out", default="data/packed")
    ap.add_argument("--datasets", nargs="+", default=list(RUNS))
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    for ds in args.datasets:
        run = os.path.join(args.mambacue, RUNS[ds], "pred")
        files = sorted(glob.glob(os.path.join(run, "*.png")))
        if not files:
            print(f"[skip] {ds}: no tiles under {run}")
            continue
        raws, gts, preds, names = [], [], [], []
        for f in files:
            r, g, p = split_tile(f)
            if not (r.shape == g.shape == p.shape):
                print(f"[warn] {ds} {os.path.basename(f)} shapes {r.shape} {g.shape} {p.shape}")
                continue
            raws.append(r)
            gts.append(g)
            preds.append(p)
            names.append(os.path.splitext(os.path.basename(f))[0])
        out = os.path.join(args.out, f"{ds}.npz")
        np.savez_compressed(
            out,
            raw=np.stack(raws),
            gt=np.stack(gts),
            pred=np.stack(preds),
            names=np.array(names, dtype="U64"),
        )
        print(f"[ok] {ds} n={len(names)} shape={raws[0].shape} -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

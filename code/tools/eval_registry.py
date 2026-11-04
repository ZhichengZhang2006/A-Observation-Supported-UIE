"""Build sota_registry: evaluate every third-party prediction set under the frozen
256x256 unified protocol against server GT.

Protocol (frozen, from AutoResearch_A_Observation_Supported_UIE.md line 144):
  whole image -> resize to 256x256 (no crop, no pad), same resampling for all methods.
  Here: cv2.INTER_AREA for downscale and INTER_LINEAR for upscale, per metrics.resize_unified.

Metrics are computed per image, then averaged (image as independent unit).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from obsui import metrics  # noqa: E402

IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


def stem(p: str) -> str:
    return os.path.splitext(os.path.basename(p))[0]


def index_by_stem(d: str) -> Dict[str, str]:
    out = {}
    for fn in sorted(os.listdir(d)):
        if fn.lower().endswith(IMG_EXT):
            out[stem(fn)] = os.path.join(d, fn)
    return out


def load_bgr(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("cannot read " + path)
    return img


def evaluate(pred_dir: str, ref_dir: str, skip_names=("inference.json",)) -> dict:
    refs = index_by_stem(ref_dir)
    preds = index_by_stem(pred_dir)
    rows = []
    missing_ref = []
    missing_pred = []
    for s, rp in refs.items():
        pp = preds.get(s)
        if pp is None:
            missing_pred.append(s)
            continue
        ref = load_bgr(rp)
        pred = load_bgr(pp)
        if pred.shape[:2] != ref.shape[:2]:
            pred = cv2.resize(pred, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_AREA)
        # unified protocol: both resized to 256x256 exactly
        ref_u = cv2.resize(ref, (256, 256), interpolation=cv2.INTER_AREA)
        pred_u = cv2.resize(pred, (256, 256), interpolation=cv2.INTER_AREA)
        m = metrics.evaluate_pair(pred_u, ref_u)
        rows.append({"stem": s, **m})
    for s in preds:
        if s not in refs:
            missing_ref.append(s)
    agg = metrics.aggregate(rows) if rows else {}
    return {
        "n": len(rows),
        "n_ref": len(refs),
        "n_pred": len(preds),
        "missing_pred": missing_pred[:20],
        "n_missing_pred": len(missing_pred),
        "missing_ref": missing_ref[:20],
        "n_missing_ref": len(missing_ref),
        "psnr": agg.get("psnr"),
        "ssim": agg.get("ssim"),
        "mse": agg.get("mse"),
        "psnr_std": agg.get("psnr_std"),
        "ssim_std": agg.get("ssim_std"),
        "rows": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-root", default=r"C:\Users\Zhingcheng Zhang\Desktop\MambaCue-Result\ALL\results")
    ap.add_argument("--ref-root", default=os.path.join(HERE, "..", "..", "_ops", "data_refs"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default=os.path.join(HERE, "..", "..", "_ops", "registry.json"))
    args = ap.parse_args()

    results = {}
    for ds in ("UIEB", "LSUI", "UFO"):
        ref_dir = os.path.join(args.ref_root, ds, args.split, "GT")
        if not os.path.isdir(ref_dir):
            print("MISSING ref dir", ref_dir)
            continue
        ds_pred_root = os.path.join(args.pred_root, ds)
        if not os.path.isdir(ds_pred_root):
            print("MISSING pred dir", ds_pred_root)
            continue
        results[ds] = {}
        for meth in sorted(os.listdir(ds_pred_root)):
            mdir = os.path.join(ds_pred_root, meth)
            if not os.path.isdir(mdir):
                continue
            try:
                r = evaluate(mdir, ref_dir)
            except Exception as exc:  # noqa: BLE001
                print(f"[ERR] {ds}/{meth}: {type(exc).__name__}: {exc}")
                continue
            results[ds][meth] = r
            print(f"{ds:5s} {meth:14s} n={r['n']:4d} psnr={r['psnr']:.4f} ssim={r['ssim']:.4f} "
                  f"(miss_pred={r['n_missing_pred']} miss_ref={r['n_missing_ref']})")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=1, ensure_ascii=False)
    print("\nwrote", args.out)

    # best per dataset
    print("\n=== BEST per dataset (256x256 unified) ===")
    for ds, d in results.items():
        if not d:
            continue
        best_p = max(d.items(), key=lambda kv: kv[1]["psnr"] or 0)
        best_s = max(d.items(), key=lambda kv: kv[1]["ssim"] or 0)
        print(f"{ds}: PSNR {best_p[0]} {best_p[1]['psnr']:.4f} | SSIM {best_s[0]} {best_s[1]['ssim']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Per-image features and errors for the H2 risk model (v3, image level).

Why image level and not blocks
-------------------------------
The v1/v2 study reduced everything to 16x16 blocks, but six of the seven support
features were *constant per image* (they were broadcast across the image's blocks).
Block-level cross-validation with image grouping then cannot distinguish "the
support statistic predicts error" from "the image identity predicts error": the
feature is an image-level constant, so any within-image variation in the target
that happens to correlate with it is unidentifiable.  v3 therefore models the
quantity a deployment actually needs -- *which images will come back wrong* -- at
the image level, with one row per image.

Feature sets
------------
``ordinary``  9 statistics computable from the observation alone, including the
              fitted optical depth ``tau_mean`` (H1 showed that controlling for
              depth removes most of the marginal support/error correlation, so a
              baseline without it would be a strawman).
``support``   7 leverage-weighted information statistics for colour / band /
              pixel direction families, from ``obsui.support``.
``ensemble``  2 disagreement statistics between the two frozen backbones --
              the cheapest ordinary epistemic-uncertainty proxy available without
              training an MC-dropout model.  Added by the model script, not here.

Nothing in either feature set reads the reference image.  The reference enters
only through the target.

Usage
-----
    python code/tools/extract_features_v3.py --split train --backbone phaseformer
Writes ``artifacts/features_v3/<split>_<backbone>.npz``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import obsui  # noqa: E402,F401  (pins BLAS threads before numpy)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

import obsui.dataio as dataio  # noqa: E402
import obsui.metrics as M  # noqa: E402
import obsui.support as S  # noqa: E402

DATASET_ROOT = "/home/user/UnderWaterDataset"
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PRED_ROOT = os.path.join(PROJECT_ROOT, "predictions")
ART_ROOT = os.path.join(PROJECT_ROOT, "artifacts", "features_v3")

ORDINARY = ["lum_mean", "lum_std", "rg_mean", "gb_mean", "grad", "noise_lap",
            "sat_frac", "entropy", "tau_mean"]
SUPPORT = ["lq_colour", "lq_band", "lq_pixel", "lev_colour", "frac_qeff_colour",
           "lq_colour_min", "lq_colour_max"]

EPS = 0.05


def _entropy(lum: np.ndarray) -> float:
    q = np.clip((lum * 15.999).astype(np.int32), 0, 15)
    p = np.bincount(q.ravel(), minlength=16).astype(np.float64)
    p /= p.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def ordinary_features(I: np.ndarray) -> np.ndarray:
    lum = I @ np.array([0.2126, 0.7152, 0.0722])
    rg = I[:, :, 0] - I[:, :, 1]
    gb = I[:, :, 1] - I[:, :, 2]
    gx = cv2.Sobel(lum, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(lum, cv2.CV_64F, 0, 1, ksize=3)
    grad = np.sqrt(gx ** 2 + gy ** 2)
    lap = cv2.Laplacian(lum, cv2.CV_64F, ksize=3)
    sat = I.max(axis=2) - I.min(axis=2)
    return np.array([
        float(lum.mean()), float(lum.std()), float(rg.mean()), float(gb.mean()),
        float(grad.mean()), float(np.abs(lap).mean()), float((sat > 0.1).mean()),
        _entropy(lum),
    ], dtype=np.float64)


def support_features(I: np.ndarray, J: np.ndarray, fit: dict) -> np.ndarray:
    h, w = I.shape[:2]
    an = S.analyse(I, J, fit)
    rows = S.direction_table(an, h, w, k_low=4, k_px=8)
    fam: dict = {}
    for r in rows:
        r["log10q_eff"] = float(np.log10(max(r["q_eff"], 1e-30)))
        fam.setdefault(r["kind"], []).append(r)

    def stat(kind, key="log10q_eff", fn=np.mean):
        vals = [r[key] for r in fam.get(kind, [])]
        return float(fn(vals)) if vals else float("nan")

    dc = fam.get("dc", [])
    qeff_ratio = float(np.median([r["q_eff"] / max(r["q"], 1e-12) for r in dc])) if dc else 0.0
    return np.array([
        stat("dc"), stat("low"), stat("pixel"), stat("dc", "leverage"),
        qeff_ratio, stat("dc", fn=np.min), stat("dc", fn=np.max),
    ], dtype=np.float64)


def targets(pred: np.ndarray, ref: np.ndarray) -> dict:
    # data_range=1.0 explicitly: these arrays are [0,1] floats, and the legacy
    # default in metrics.py is 255.0.
    psnr = M.psnr(pred, ref, data_range=1.0)
    lum_w = np.array([0.2126, 0.7152, 0.0722])
    ssim = M.ssim_channel(pred @ lum_w, ref @ lum_w, data_range=1.0)
    lr = np.log(np.maximum(pred[..., 0] + EPS, 1e-6)) - np.log(np.maximum(pred[..., 1] + EPS, 1e-6))
    lg = np.log(np.maximum(pred[..., 1] + EPS, 1e-6)) - np.log(np.maximum(pred[..., 2] + EPS, 1e-6))
    rr = np.log(np.maximum(ref[..., 0] + EPS, 1e-6)) - np.log(np.maximum(ref[..., 1] + EPS, 1e-6))
    rg_ = np.log(np.maximum(ref[..., 1] + EPS, 1e-6)) - np.log(np.maximum(ref[..., 2] + EPS, 1e-6))
    chroma = float(0.5 * (np.abs(lr - rr).mean() + np.abs(lg - rg_).mean()))
    ref_lum = ref @ np.array([0.2126, 0.7152, 0.0722])
    err = np.abs(pred - ref).mean(axis=2)
    return {
        "psnr": float(psnr), "ssim": float(ssim), "chroma": chroma,
        "rel": float((err / (ref_lum + EPS)).mean()), "abs": float(err.mean()),
    }


def _one(args):
    name, input_path, ref_path, pred_path = args
    try:
        # to_unified returns uint8 in [0,255]; predictions are loaded as
        # [0,1] floats.  Normalise here, once, or every target and every
        # support statistic is computed across two different scales.
        I = dataio.to_unified(dataio.load_rgb(input_path)).astype(np.float64) / 255.0
        R = dataio.to_unified(dataio.load_rgb(ref_path)).astype(np.float64) / 255.0
        if not os.path.exists(pred_path):
            return {"name": name, "error": "missing prediction"}
        J = np.asarray(Image.open(pred_path).convert("RGB"), dtype=np.float64) / 255.0
        if J.shape != I.shape:
            return {"name": name, "error": f"pred shape {J.shape} != {I.shape}"}
        h, w = I.shape[:2]
        fit = S.fit_water(I, J, h, w, noise_mode="misfit")
        of = ordinary_features(I)
        of[-1] = float(np.mean(fit["tau"]))
        sf = support_features(I, J, fit)
        tg = targets(J, R)
        if not (0.0 < tg["psnr"] < 60.0):
            return {"name": name, "error": f"implausible psnr {tg['psnr']:.2f} -- "
                                          "check the [0,255] vs [0,1] scale"}
        return {"name": name, "ordinary": of.tolist(), "support": sf.tolist(),
                "targets": tg}
    except Exception as exc:  # one bad file must not lose the whole split
        return {"name": name, "error": f"{type(exc).__name__}: {exc}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "val", "test"])
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--dataset", default="UIEB")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    tag = args.dataset if args.split == "test" else f"{args.dataset}-{args.split}"
    in_dir = os.path.join(DATASET_ROOT, f"{args.dataset}/{args.split}", "input")
    ref_dir = os.path.join(DATASET_ROOT, f"{args.dataset}/{args.split}", "GT")
    pred_dir = os.path.join(PRED_ROOT, tag, args.backbone)
    pairs = dataio.resolve_pairs(in_dir, ref_dir, args.dataset)
    if args.limit:
        pairs.pairs = pairs.pairs[: args.limit]

    jobs = [(p.name, p.input_path, p.ref_path, os.path.join(pred_dir, p.name + ".png"))
            for p in pairs.pairs]
    print(f"[{args.split}/{args.backbone}] {len(jobs)} images, {args.workers} workers", flush=True)
    t0 = time.time()
    rows = []
    if args.workers <= 1:
        for j in jobs:
            rows.append(_one(j))
    else:
        import multiprocessing as mp
        with mp.Pool(args.workers, initializer=S._pin_threads, initargs=(1,)) as pool:
            for r in pool.imap_unordered(_one, jobs, chunksize=1):
                rows.append(r)
                if len(rows) % 50 == 0:
                    print(f"  {len(rows)}/{len(jobs)}", flush=True)
    ok = [r for r in rows if "error" not in r]
    bad = [r for r in rows if "error" in r]
    ok.sort(key=lambda r: r["name"])
    os.makedirs(ART_ROOT, exist_ok=True)
    out = os.path.join(ART_ROOT, f"{args.dataset}_{args.split}_{args.backbone}.npz")
    np.savez_compressed(
        out,
        names=np.array([r["name"] for r in ok]),
        ordinary=np.array([r["ordinary"] for r in ok], dtype=np.float64),
        support=np.array([r["support"] for r in ok], dtype=np.float64),
        targets=np.array([[r["targets"][k] for k in ("psnr", "ssim", "chroma", "rel", "abs")]
                          for r in ok], dtype=np.float64),
    )
    print(f"  wrote {out} n={len(ok)} errors={len(bad)} ({time.time() - t0:.1f}s)", flush=True)
    if bad:
        print(json.dumps(bad[:5], indent=1), flush=True)


if __name__ == "__main__":
    main()

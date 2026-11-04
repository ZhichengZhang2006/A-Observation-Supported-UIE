"""H2 -- the falsification test the whole project stands or falls on.

H1 showed that observation support correlates with enhancement error, after
controlling for degradation level, transmission, brightness and gradient.  That is
necessary but not sufficient.  The question a referee will ask is the one the
novelty audit wrote down as the decisive falsifier:

    does the support statistic add predictive power *over ordinary uncertainty
    and degradation predictors*, or is it a rebadged version of them?

This script answers it by fitting the same error model on two nested feature sets
and comparing held-out R^2, with blocks grouped by image so no image contributes
to both sides of a split.  The split unit matters: blocks from one image share a
water model and a scene, so a random block split would leak and inflate both
models.

    F0  ordinary: brightness, contrast, colour cast, gradient energy, noise
        estimate, saturation, entropy, and **fitted optical depth** -- the things a
        practitioner would compute without this paper.  Optical depth is in F0
        deliberately: H1 measured that ``rho(support, error)`` drops from 0.173 to
        0.068 once depth is controlled, so a baseline without it would be a
        strawman and the incremental claim would be unearned.
    F1  F0 + observation support: per-block leverage-weighted information for
        colour / band / pixel directions, plus the gauge ambiguity budget.

Both sets are computed from the **observation and the method's own output only**.
No reference image enters either feature set, so this is a measurement a
deployment could make.  The reference enters only as the target.

If F1 does not beat F0 by a margin that survives the grouped cross-validation,
the support statistic is not carrying independent information and this project's
mechanism claim must be withdrawn.  That is written here, before the number is
seen, so the result cannot be rationalised afterwards.

Targets, and why there are three
--------------------------------
``abs``  |error| in image units.  Shrinks with signal level, so dark blocks look
         good while being wrong in relative terms.
``rel``  |error| / (reference + 0.05).  The honest per-block version.
``chr``  log-opponent chromatic difference.  This is the "unsupported colour" the
         project is about, and the one the mechanism must move.

The headline is ``chr``; ``abs`` is kept because a reader will want it and because
reporting only the flattering target would be exactly the failure mode this work
criticises.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import obsui  # noqa: E402,F401

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import obsui.support as S  # noqa: E402

BS = 16
EPS = 0.05

ORDINARY = ["lum_mean", "lum_std", "rg_mean", "gb_mean", "grad", "noise_lap",
            "sat_frac", "entropy", "tau_mean"]
SUPPORT = ["lq_colour", "lq_band", "lq_pixel", "lev_colour", "frac_qeff_colour",
           "lq_colour_min", "lq_colour_max"]


def _block_mean(a, h, w, bs=BS):
    """Average an ``(h,w)`` or ``(h,w,c)`` map over non-overlapping blocks."""
    if a.ndim == 2:
        a = a[:, :, None]
    hh, ww = (h // bs) * bs, (w // bs) * bs
    v = a[:hh, :ww].reshape(hh // bs, bs, ww // bs, bs, a.shape[2]).mean(axis=(1, 3))
    return v.reshape(-1, a.shape[2])


def _block_std(a, h, w, bs=BS):
    hh, ww = (h // bs) * bs, (w // bs) * bs
    v = a[:hh, :ww].reshape(hh // bs, bs, ww // bs, bs).std(axis=(1, 3))
    return v.reshape(-1)


def _ordinary_features(I, h, w, tau=None):
    """Everything computable without a Fisher calculation, per block.

    ``tau`` is the fitted optical depth, passed in so it lands in the ordinary set.
    It comes from the same fit as the support features, which is deliberate: the
    baseline gets every non-support quantity that fit produces, so any remaining
    gain is attributable to the *information* quantity and not to having fitted a
    water model at all.
    """
    lum = I @ np.array([0.2126, 0.7152, 0.0722])
    rg = I[:, :, 0] - I[:, :, 1]
    gb = I[:, :, 1] - I[:, :, 2]
    gx = cv2.Sobel(lum, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(lum, cv2.CV_64F, 0, 1, ksize=3)
    grad = np.sqrt(gx ** 2 + gy ** 2)
    lap = cv2.Laplacian(lum, cv2.CV_64F, ksize=3)
    sat = (I.max(axis=2) - I.min(axis=2))
    # entropy on 16 grey levels, per block, via the histogram of the block
    q = np.clip((lum * 15.999).astype(np.int32), 0, 15)
    hh, ww = (h // BS) * BS, (w // BS) * BS
    qb = q[:hh, :ww].reshape(hh // BS, BS, ww // BS, BS)
    ent = np.zeros((hh // BS, ww // BS))
    for bi in range(hh // BS):
        for bj in range(ww // BS):
            p = np.bincount(qb[bi, :, bj, :].ravel(), minlength=16).astype(np.float64)
            p /= p.sum()
            p = p[p > 0]
            ent[bi, bj] = -(p * np.log2(p)).sum()

    cols = [
        _block_mean(lum, h, w)[:, 0],
        _block_std(lum, h, w),
        _block_mean(rg, h, w)[:, 0],
        _block_mean(gb, h, w)[:, 0],
        _block_mean(grad, h, w)[:, 0],
        _block_mean(np.abs(lap), h, w)[:, 0],
        _block_mean(sat, h, w)[:, 0],
        ent.reshape(-1),
    ]
    if tau is not None:
        cols.append(_block_mean(np.asarray(tau).reshape(h, w), h, w)[:, 0])
    return np.stack(cols, axis=1)


def _support_features(I, J, h, w, fit):
    """Per-block observation support, from the observation and a claimed scene.

    ``J`` is the method's own output -- the question asked is "given this
    observation, how well is the scene *this method claims* pinned down?".  Using
    the reference here would make the feature unavailable at test time and would
    also beg the question.
    """
    an = S.analyse(I, J, fit)
    # ``analyse`` returns per-pixel quantities flattened as ``(N, 3)`` with N =
    # h*w; put them back on the image grid so the block reduction is meaningful.
    lq = an["log10q_eff"].reshape(h, w, 3)
    lev = an["leverage"].reshape(h, w, 3)

    # Direction families are what make this interpretable: colour is a global
    # offset, band is a spatial frequency, pixel is a single sample.
    # ``direction_info`` returns raw ``q`` and ``q_eff``; the log is taken here so
    # the features are on the same order-one scale as everything else.
    k_low = 4
    rows = S.direction_table(an, h, w, k_low=k_low, k_px=8)
    fam = {}
    for r in rows:
        r["log10q_eff"] = float(np.log10(max(r["q_eff"], 1e-30)))
        fam.setdefault(r["kind"], []).append(r)

    def fam_stat(kind, key="log10q_eff"):
        vals = [r[key] for r in fam.get(kind, [])]
        return float(np.mean(vals)) if vals else float("nan")

    lq_c = fam_stat("dc")
    lq_b = fam_stat("low")
    lq_p = fam_stat("pixel")
    lev_c = fam_stat("dc", "leverage")
    qeff_ratio = float(np.nanmedian([r["q_eff"] / max(r["q"], 1e-12)
                                     for r in fam.get("dc", [])]) or 0.0)

    cols = [
        np.full(h * w // (BS * BS), lq_c),
        np.full(h * w // (BS * BS), lq_b),
        np.full(h * w // (BS * BS), lq_p),
        np.full(h * w // (BS * BS), lev_c),
        np.full(h * w // (BS * BS), qeff_ratio),
        _block_mean(lq, h, w)[:, 0],
        _block_mean(lev, h, w)[:, 0],
    ]
    return np.stack(cols, axis=1), {"lq_colour": lq_c, "lq_band": lq_b,
                                    "lq_pixel": lq_p, "lev_colour": lev_c,
                                    "qeff_ratio_colour": qeff_ratio}


def _targets(Jh, Js, h, w):
    lum_h = Jh @ np.array([0.2126, 0.7152, 0.0722])
    lum_s = Js @ np.array([0.2126, 0.7152, 0.0722])
    err = np.abs(Jh - Js).mean(axis=2)
    rel = (np.abs(Jh - Js) / (Js + EPS)).mean(axis=2)

    def chroma(a):
        r, g, bl = a[:, :, 0], a[:, :, 1], a[:, :, 2]
        return np.stack([np.log((r + EPS) / (g + EPS)),
                         np.log((g + EPS) / (bl + EPS))], axis=2)

    chr_ = np.abs(chroma(Jh) - chroma(Js)).mean(axis=2)
    return {
        "abs": _block_mean(err, h, w)[:, 0],
        "rel": _block_mean(rel, h, w)[:, 0],
        "chr": _block_mean(chr_, h, w)[:, 0],
    }


def _ridge_fit(X, y, lam=1e-2):
    X1 = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
    A = X1.T @ X1 + lam * np.eye(X1.shape[1])
    w = np.linalg.solve(A, X1.T @ y)
    return w


def _ridge_pred(w, X):
    X1 = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
    return X1 @ w


def _r2(y, yh):
    ss_res = float(((y - yh) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss_res / max(ss_tot, 1e-18)


def _grouped_cv(X, y, groups, n_folds=5, lam=1e-2, centre=True):
    """Cross-validate with whole images as folds.

    Blocks within an image share a water model and a scene, so a random block
    split would leak and would flatter every model.

    ``centre`` removes each image's mean from both the features and the target.
    That is deliberate and it changes the question to the one being asked:
    "within a given image, does support tell me *where* the error is?"  Without
    centring, a pooled R^2 is dominated by differences in average error *between*
    images, which no block-level feature can predict from a held-out image, and
    every model scores below zero.  The mode is recorded in the artifact so a
    reader is never in doubt which question was answered.
    """
    ug = np.unique(groups)
    rng = np.random.default_rng(0)
    rng.shuffle(ug)

    Xc, yc = X, y
    if centre:
        Xc = X.copy()
        yc = y.copy()
        for g in ug:
            m = groups == g
            Xc[m] -= Xc[m].mean(axis=0, keepdims=True)
            yc[m] -= yc[m].mean()

    pred = np.full_like(yc, np.nan)
    for f in range(n_folds):
        hold = ug[f::n_folds]
        m = np.isin(groups, hold)
        if m.all() or (~m).sum() < 10:
            continue
        w = _ridge_fit(Xc[~m], yc[~m], lam)
        pred[m] = _ridge_pred(w, Xc[m])
    ok = ~np.isnan(pred)
    if ok.sum() < 10:
        return float("nan")
    return _r2(yc[ok], pred[ok])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="data/packed/UIEB.npz")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="artifacts/h2.json")
    args = ap.parse_args()

    dat = np.load(args.npz, allow_pickle=True)
    GT, RAW, PRED = dat["gt"], dat["raw"], dat["pred"]
    names = [str(x) for x in dat["names"]]
    n = len(names) if args.limit <= 0 else min(args.limit, len(names))

    X0_all, X1_all, Y_all, G_all = [], [], [], []
    per_image = []
    for i in range(n):
        gt = GT[i].astype(np.float64) / 255.0
        raw = RAW[i].astype(np.float64) / 255.0
        pred = PRED[i].astype(np.float64) / 255.0
        h, w = gt.shape[:2]

        fit = S.fit_water(raw, pred, h, w, k_depth=9, noise_mode="misfit")
        tau_mean = fit["tau"].mean(axis=1)          # per-pixel, (N,)
        f0 = _ordinary_features(raw, h, w, tau=tau_mean)
        f1, sup = _support_features(raw, pred, h, w, fit)
        tg = _targets(pred, gt, h, w)

        nb = f0.shape[0]
        if f1.shape[0] != nb:
            m = min(nb, f1.shape[0])
            f0, f1 = f0[:m], f1[:m]
            tg = {k: v[:m] for k, v in tg.items()}
            nb = m

        X0_all.append(f0)
        X1_all.append(np.concatenate([f0, f1], axis=1))
        Y_all.append(tg)
        G_all.append(np.full(nb, i))
        per_image.append({"name": names[i], **sup,
                          "chi2_over_dof": float(fit["chi2"]
                                                 / max(fit.get("dof", fit["n_obs"]), 1))})
        if (i + 1) % 20 == 0 or i == n - 1:
            print(f"  [{i+1}/{n}] {names[i]:14s} lq_colour={sup['lq_colour']:.2f} "
                  f"lq_pixel={sup['lq_pixel']:.2f}", flush=True)

    X0 = np.concatenate(X0_all, axis=0)
    X1 = np.concatenate(X1_all, axis=0)
    G = np.concatenate(G_all, axis=0)
    Y = {k: np.concatenate([y[k] for y in Y_all], axis=0) for k in Y_all[0]}

    # standardise using global statistics; folds are large enough that this is not
    # a meaningful leak, and it keeps the ridge penalty comparable across features
    mu0, sd0 = X0.mean(0), X0.std(0) + 1e-9
    X0s = (X0 - mu0) / sd0
    mu1, sd1 = X1.mean(0), X1.std(0) + 1e-9
    X1s = (X1 - mu1) / sd1

    print(f"\nblocks={X0.shape[0]} images={n}")
    print("within-image CV (per-image means removed): "
          "'where in this image is the error largest?'")
    print(f"{'target':<6s} {'R2(F0)':>9s} {'R2(F1)':>9s} {'delta':>8s}")
    res = {}
    for tgt, y in Y.items():
        r0 = _grouped_cv(X0s, y, G, n_folds=5, centre=True)
        r1 = _grouped_cv(X1s, y, G, n_folds=5, centre=True)
        res[tgt] = {"r2_ordinary": r0, "r2_with_support": r1, "delta": r1 - r0}
        print(f"{tgt:<6s} {r0:>9.4f} {r1:>9.4f} {r1-r0:>+8.4f}")

    # Pooled (not centred) is reported too, because a reader will ask.  It answers
    # "which image is worse", which a held-out image cannot be predicted for from
    # block features -- so the numbers are expected to be poor and are labelled.
    print()
    print("pooled CV (not centred): 'which image is worse?' -- expected poor for "
          "any block-level feature")
    for tgt, y in Y.items():
        r0 = _grouped_cv(X0s, y, G, n_folds=5, centre=False)
        r1 = _grouped_cv(X1s, y, G, n_folds=5, centre=False)
        res[tgt]["pooled_r2_ordinary"] = r0
        res[tgt]["pooled_r2_with_support"] = r1
        print(f"{tgt:<6s} {r0:>9.4f} {r1:>9.4f} {r1-r0:>+8.4f}")

    # A single pooled number hides whether support helps on the images that matter.
    # Per-image Spearman between the block-mean support and the block-mean error is
    # the image-level version of the H1 claim.
    print()
    fr = np.array([p["lq_colour"] for p in per_image])
    allres = {k: np.array([np.mean(y[k]) for y in Y_all]) for k in Y_all[0]}
    from scipy.stats import spearmanr
    for k, v in allres.items():
        rho, p = spearmanr(fr, v)
        print(f"  per-image rho(lq_colour, {k}) = {rho:+.4f}  (p={p:.4f})")
        res.setdefault("per_image", {})[k] = {"rho": float(rho), "p": float(p)}

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"npz": args.npz, "n_images": n,
                   "n_blocks": int(X0.shape[0]),
                   "features_ordinary": ORDINARY,
                   "features_support": SUPPORT,
                   "cv": "5-fold grouped by image",
                   "targets": res,
                   "per_image": per_image}, fh, indent=1)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

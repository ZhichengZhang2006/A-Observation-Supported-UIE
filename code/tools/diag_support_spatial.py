"""Is the support signal spatially flat?  A diagnostic, not an experiment.

The H3 ablation produced identical numbers for ``support_g``, ``const_g``,
``random_g`` and ``degrade_g`` -- to two decimals.  Four different spatial patterns
cannot give the same answer unless the thing being varied is not varying, so before
reading that result as science, check whether the instrument has any spatial
content at all.

If ``g_support`` is dominated by a global constant (which it will be if the fit
absorbs most per-pixel variation into the water model), then a spatially selective
intervention has nothing to select between, and the whole "act where support is
low" idea is untestable in this form.  That would be a finding about the
measurement, and it changes what the mechanism chapter can claim.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import obsui  # noqa: E402,F401

import numpy as np  # noqa: E402

import obsui.support as S  # noqa: E402

LUM = np.array([0.2126, 0.7152, 0.0722])
EPS = 0.05


def main():
    dat = np.load("data/packed/UIEB.npz", allow_pickle=True)
    GT, RAW, PRED = dat["gt"], dat["raw"], dat["pred"]
    names = [str(x) for x in dat["names"]]
    wm = S.default_water()

    rows = []
    for i in range(min(12, len(names))):
        gt = GT[i].astype(np.float64) / 255.0
        raw = RAW[i].astype(np.float64) / 255.0
        pred = PRED[i].astype(np.float64) / 255.0
        h, w = gt.shape[:2]

        fit = S.fit_water(raw, pred, h, w, k_depth=9, noise_mode="misfit")
        an = S.analyse(raw, pred, fit)
        lq = an["log10q_eff"].reshape(h, w, 3).mean(axis=2)
        sup = S.support_score(lq)
        g = 1.0 - sup

        # how much of g's variance is a global constant vs spatial structure?
        frac_const = float(np.var(g.mean(axis=(0, 1)) * np.ones_like(g))
                           / max(np.var(g), 1e-30))
        # chroma error map, for reference
        def chroma(a):
            r, gg, b = a[..., 0], a[..., 1], a[..., 2]
            return np.stack([np.log((r + EPS) / (gg + EPS)),
                             np.log((gg + EPS) / (b + EPS))], axis=-1)
        err = np.abs(chroma(pred) - chroma(gt)).mean(axis=2)

        # block-level correlation between g and error (the spatial claim)
        bs = 16
        hh, ww = (h // bs) * bs, (w // bs) * bs
        gb = g[:hh, :ww].reshape(hh // bs, bs, ww // bs, bs).mean(axis=(1, 3)).ravel()
        eb = err[:hh, :ww].reshape(hh // bs, bs, ww // bs, bs).mean(axis=(1, 3)).ravel()
        from scipy.stats import spearmanr
        rho, p = spearmanr(gb, eb)

        rows.append({
            "name": names[i],
            "g_mean": float(g.mean()), "g_std": float(g.std()),
            "g_p05": float(np.percentile(g, 5)),
            "g_p95": float(np.percentile(g, 95)),
            "g_range": float(np.percentile(g, 95) - np.percentile(g, 5)),
            "frac_variance_from_constant": frac_const,
            "block_rho_g_err": float(rho), "block_p": float(p),
            "chi2_over_dof": float(fit["chi2"] / max(fit.get("dof", 1), 1)),
            "lq_std": float(lq.std()),
        })
        print(f"{names[i]:14s} g={g.mean():.4f}+-{g.std():.4f} "
              f"p05={np.percentile(g,5):.4f} p95={np.percentile(g,95):.4f} "
              f"rho(g,err)={rho:+.3f} chi2/dof={fit['chi2']/max(fit.get('dof',1),1):.2f}",
              flush=True)

    gm = np.array([r["g_mean"] for r in rows])
    gs = np.array([r["g_std"] for r in rows])
    rr = np.array([r["block_rho_g_err"] for r in rows])
    print()
    print(f"across images: g_mean={gm.mean():.3f} (sd {gm.std():.3f}), "
          f"within-image g_std={gs.mean():.4f} (sd {gs.std():.4f})")
    print(f"median within-image block rho(g, chroma err) = {np.median(rr):+.4f}")
    print(f"images with rho>0: {(rr>0).sum()}/{len(rr)}")
    print()
    print("if within-image g_std is ~0, a spatially selective intervention has")
    print("nothing to select between -- the mechanism cannot be tested this way.")

    with open("artifacts/support_spatial.json", "w", encoding="utf-8") as fh:
        json.dump({"rows": rows, "g_mean_across": float(gm.mean()),
                   "g_std_within": float(gs.mean()),
                   "median_block_rho": float(np.median(rr))}, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())

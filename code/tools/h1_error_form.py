"""Absolute error is the wrong target: it scales with the signal, not with support.

The 40-image run gives ``rho(q_eff, |Jh-J*|) = +0.18`` at region level, stable
under partialling out change magnitude, optical depth, brightness and gradient
energy.  Taken at face value that *refutes* H1: high support would go with high
error.

It does not, and the reason is a property of the metric rather than of the
physics.  ``|Jh - J*|`` is an absolute difference, so it is small wherever the
scene is dark, and dark is exactly where the observation carries least
information.  An enhancer can be 50 % wrong in a shadow and still contribute
almost nothing to the absolute error.  So the quantity that has to be predicted
is the *relative* deviation, and the one this project actually cares about is the
chromatic part of it: unsupported colour, not unsupported photons.

This script separates the three:
    abs   |Jh - J*|                       (dominated by signal level)
    rel   |Jh - J*| / (J* + 0.05)         (scale free)
    chr   |chroma(Jh) - chroma(J*)|       (hue/saturation, log-opponent)
    lum   |luma(Jh) - luma(J*)|           (structure + exposure)
and repeats the partial-correlation control for each.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import obsui  # noqa: E402,F401

import numpy as np  # noqa: E402

import obsui.support as S  # noqa: E402
from tools.h1_diagnose import blocks, grad_energy, rank, spearman  # noqa: E402
from tools.h1_confound import partial_spearman  # noqa: E402

EPS = 0.05


def luma(a: np.ndarray) -> np.ndarray:
    return 0.2126 * a[..., 0] + 0.7152 * a[..., 1] + 0.0722 * a[..., 2]


def chroma_opp(a: np.ndarray) -> np.ndarray:
    """Log-opponent chroma channels, (H, W, 2).  Log ratios are what a colour
    cast actually is; a linear R-B difference reports dark pixels as neutral."""
    return np.stack([np.log((a[..., 0] + EPS) / (a[..., 1] + EPS)),
                     np.log((a[..., 1] + EPS) / (a[..., 2] + EPS))], axis=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--k-depth", type=int, default=16)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    raw, gt, pred = d["raw"], d["gt"], d["pred"]
    names = [str(x) for x in d["names"]]
    n = len(raw) if args.limit <= 0 else min(args.limit, len(raw))
    print(f"dataset={os.path.basename(args.npz)} n={n}")

    keys = ("abs", "rel", "chr", "lumm", "lq", "tau", "lum", "grad", "dlt", "fwd")
    cols = {k: [] for k in keys}
    for i in range(n):
        I = raw[i].astype(np.float64) / 255.0
        Jh = pred[i].astype(np.float64) / 255.0
        Jt = gt[i].astype(np.float64) / 255.0
        h, w = I.shape[:2]
        try:
            fit = S.fit_water(I, Jh, h, w, k_depth=args.k_depth, noise_mode="image")
            an = S.analyse(I, Jh, fit)
        except Exception as exc:
            print(f"  [{i}] FAILED {names[i]}: {type(exc).__name__}: {exc}")
            continue
        rec = S.render(Jh.reshape(-1, 3), fit["tau"], fit["water"])
        fwd = (np.abs(rec - I.reshape(-1, 3)) / np.maximum(fit["sigma"], 1e-6)).mean(axis=1).reshape(h, w)

        ce, ct = chroma_opp(Jh), chroma_opp(Jt)
        cols["abs"].append(blocks(np.abs(Jh - Jt).mean(axis=2)).ravel())
        cols["rel"].append(blocks((np.abs(Jh - Jt) / (Jt + EPS)).mean(axis=2)).ravel())
        cols["chr"].append(blocks(np.abs(ce - ct).mean(axis=2)).ravel())
        cols["lumm"].append(blocks(np.abs(luma(Jh) - luma(Jt))).ravel())
        cols["lq"].append(blocks(an["log10q_eff"].reshape(h, w, 3).mean(axis=2)).ravel())
        cols["tau"].append(blocks(an["tau"].reshape(h, w, 3).mean(axis=2)).ravel())
        cols["lum"].append(blocks(I.mean(axis=2)).ravel())
        cols["grad"].append(grad_energy(I.mean(axis=2)).ravel())
        cols["dlt"].append(blocks(np.abs(Jh - I).mean(axis=2)).ravel())
        cols["fwd"].append(blocks(fwd).ravel())
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{n}")

    R = {k: np.concatenate(v) for k, v in cols.items()}
    print(f"\nblocks={len(R['rel'])}")
    print(f"{'target':8s} {'rho(lq)':>9s} {'rho(tau)':>9s} {'rho(lum)':>9s} {'rho(grad)':>10s} "
          f"{'partial|lq':>11s} {'partial|fwd':>12s}")
    out = {"n_blocks": int(len(R["rel"])), "table": {}}
    for t in ("abs", "rel", "chr", "lumm"):
        Z = np.column_stack([R["dlt"], R["tau"], R["lum"], R["grad"]])
        p = partial_spearman(R["lq"], R[t], Z)
        pf = partial_spearman(R["fwd"], R[t], Z)
        print(f"{t:8s} {spearman(R['lq'], R[t]):+9.4f} {spearman(R['tau'], R[t]):+9.4f} "
              f"{spearman(R['lum'], R[t]):+9.4f} {spearman(R['grad'], R[t]):+10.4f} "
              f"{p:+11.4f} {pf:+12.4f}")
        out["table"][t] = {
            "rho_lq": spearman(R["lq"], R[t]),
            "rho_tau": spearman(R["tau"], R[t]),
            "rho_lum": spearman(R["lum"], R[t]),
            "rho_grad": spearman(R["grad"], R[t]),
            "partial_lq": p, "partial_fwd": pf,
        }

    # ratio form: is the chromatic ambiguity *relative to the structural one*
    # the thing that tracks error?  This is the "unsupported colour" quantity.
    ratio = R["chr"] / np.maximum(R["lumm"], 1e-3)
    print(f"\nchroma/luma error ratio: rho(lq)={spearman(R['lq'], ratio):+.4f}  "
          f"rho(tau)={spearman(R['tau'], ratio):+.4f}")
    out["chroma_over_luma"] = {"rho_lq": spearman(R["lq"], ratio),
                              "rho_tau": spearman(R["tau"], ratio)}
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=1)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

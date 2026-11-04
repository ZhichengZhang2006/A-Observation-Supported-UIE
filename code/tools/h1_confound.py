"""Why is support *negatively* correlated with enhancer error?

The first H1 run gave the wrong sign: high ``q_eff`` went with high error, both
across images (rho = -0.60 against PSNR) and across regions (rho = -0.25).  A
sign flip like that is either a real effect or a confound, and the difference
matters, so this script measures the confound directly.

Candidate explanation: ``q_eff`` tracks *how much the enhancer changed the
image*.  Where the observation is informative the enhancer is confident and
edits boldly; where it is not, the enhancer (trained for fidelity) mostly leaves
the input alone and is therefore close to the reference by default.  If so, the
correlation with error is inherited from the change magnitude, and the honest
test is the partial correlation of error with support *given* the change.

Reported for each region:
    err   |Jh - J*|          ground-truth error
    dlt   |Jh - I |          how much was changed
    lq    log10 q_eff        support
    tau, lum, grad           cheap physical confounds
    fwd   |I - G(Jh)| / sig  forward-model residual (is the output observation-consistent?)
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

BS = 16


def partial_spearman(x, y, z):
    """Spearman of x and y after removing the rank-linear effect of every column of z."""
    rx, ry = rank(x), rank(y)
    Z = np.column_stack([np.ones(len(rx))] + [rank(z[:, j]) for j in range(z.shape[1])])
    bx, *_ = np.linalg.lstsq(Z, rx, rcond=None)
    by, *_ = np.linalg.lstsq(Z, ry, rcond=None)
    ex, ey = rx - Z @ bx, ry - Z @ by
    den = np.sqrt((ex ** 2).sum() * (ey ** 2).sum())
    return float((ex * ey).sum() / den) if den > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--k-depth", type=int, default=16)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    raw, gt, pred = d["raw"], d["gt"], d["pred"]
    names = [str(x) for x in d["names"]]
    n = len(raw) if args.limit <= 0 else min(args.limit, len(raw))
    print(f"dataset={os.path.basename(args.npz)} n={n}")

    cols = {k: [] for k in ("err", "dlt", "lq", "tau", "lum", "grad", "fwd", "q")}
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
        I3, J3 = I.reshape(-1, 3), Jh.reshape(-1, 3)
        rec = S.render(J3, fit["tau"], fit["water"])
        fwd = (np.abs(rec - I3) / np.maximum(fit["sigma"], 1e-6)).mean(axis=1).reshape(h, w)

        cols["err"].append(blocks(np.abs(Jh - Jt).mean(axis=2)).ravel())
        cols["dlt"].append(blocks(np.abs(Jh - I).mean(axis=2)).ravel())
        cols["lq"].append(blocks(an["log10q_eff"].reshape(h, w, 3).mean(axis=2)).ravel())
        cols["q"].append(blocks(an["q"].reshape(h, w, 3).mean(axis=2)).ravel())
        cols["tau"].append(blocks(an["tau"].reshape(h, w, 3).mean(axis=2)).ravel())
        cols["lum"].append(blocks(I.mean(axis=2)).ravel())
        cols["grad"].append(grad_energy(I.mean(axis=2)).ravel())
        cols["fwd"].append(blocks(fwd).ravel())
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{n}")

    R = {k: np.concatenate(v) for k, v in cols.items()}
    print(f"\nblocks={len(R['err'])}")
    print("correlation with err (higher = predicts error):")
    for k in ("lq", "q", "dlt", "tau", "lum", "grad", "fwd"):
        print(f"  rho({k:4s}, err) = {spearman(R[k], R['err']):+.4f}")
    print("\ncorrelation with support:")
    for k in ("dlt", "tau", "lum", "grad", "err"):
        print(f"  rho(lq, {k:4s}) = {spearman(R['lq'], R[k]):+.4f}")

    print("\npartial correlation of err with lq, controlling for:")
    combos = {
        "dlt": ["dlt"],
        "dlt+lum+grad": ["dlt", "lum", "grad"],
        "dlt+tau+lum+grad+cast": ["dlt", "tau", "lum", "grad"],
    }
    res = {}
    for tag, keys in combos.items():
        Z = np.column_stack([R[k] for k in keys])
        v = partial_spearman(R["lq"], R["err"], Z)
        res[tag] = v
        print(f"  {tag:24s} -> {v:+.4f}")

    print("\nfor reference, the same partial test for the change magnitude itself:")
    Z = np.column_stack([R["lq"], R["tau"], R["lum"], R["grad"]])
    print(f"  partial rho(dlt, err | lq,tau,lum,grad) = {partial_spearman(R['dlt'], R['err'], Z):+.4f}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"n_blocks": int(len(R["err"])),
                       "rho": {k: spearman(R[k], R["err"]) for k in
                               ("lq", "q", "dlt", "tau", "lum", "grad", "fwd")},
                       "partial_lq_err": res}, fh, indent=1)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

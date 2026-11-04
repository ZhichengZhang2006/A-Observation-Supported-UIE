"""H1 -- does observation support predict the error of a real enhancer?

The claim under test is the only one that keeps this work distinct from an
ordinary uncertainty map (see ``docs/novelty_audit.md``): the quantity must
carry predictive power *without* being fitted to a network's own residuals.

Setup
-----
For every test image we have the degraded observation ``I``, the enhancer output
``J_hat`` and the reference ``J*``.  We compute support on ``(I, J_hat)`` --
which is a legitimate *operational* estimate of the ambiguity set around the
output, not the feasible set itself, and the report must say so -- and then ask
whether the resulting map predicts ``|J_hat - J*|``.

Two levels are reported:

* **image level** -- 89 points, correlation between mean effective information
  and PSNR.  This is the coarse version.
* **region level** -- the same question on 16x16 blocks pooled over all images.
  This is where the interesting part is: the confound control.  A dark, strongly
  cast region has both low information and high error, so a raw correlation is
  worthless.  The question is whether support survives controlling for
  brightness, colour cast, optical depth, and local gradient energy.

Usage::

    python code/tools/h1_diagnose.py --npz data/packed/UIEB.npz --limit 24
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

BS = 16  # block size for the region-level analysis


def blocks(a: np.ndarray, b: int = BS) -> np.ndarray:
    """Mean-pool a (H, W) map into (H/b, W/b) block means."""
    h, w = a.shape
    return a[: h // b * b, : w // b * b].reshape(h // b, b, w // b, b).mean(axis=(1, 3))


def grad_energy(lum: np.ndarray, b: int = BS) -> np.ndarray:
    gx = np.diff(lum, axis=1, prepend=lum[:, :1])
    gy = np.diff(lum, axis=0, prepend=lum[:1, :])
    return blocks(np.sqrt(gx ** 2 + gy ** 2), b)


def rank(x: np.ndarray) -> np.ndarray:
    """Average-tie ranks, so Spearman needs no scipy."""
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x), float)
    r[order] = np.arange(len(x), dtype=float)
    # average ties
    xs = x[order]
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        if j > i:
            r[order[i:j + 1]] = 0.5 * (i + j)
        i = j + 1
    return r


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx, ry = rank(x) - rank(x).mean(), rank(y) - rank(y).mean()
    den = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / den) if den > 0 else 0.0


def ols(y: np.ndarray, x: np.ndarray):
    """Return (coef, r2, t-stats) for y ~ [1, x]."""
    A = np.column_stack([np.ones(len(y)), x])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    dof = max(len(y) - A.shape[1], 1)
    s2 = float((resid ** 2).sum() / dof)
    try:
        cov = s2 * np.linalg.inv(A.T @ A)
        se = np.sqrt(np.maximum(np.diag(cov), 1e-30))
    except np.linalg.LinAlgError:
        se = np.full(A.shape[1], np.nan)
    r2 = 1.0 - float((resid ** 2).sum()) / max(float(((y - y.mean()) ** 2).sum()), 1e-30)
    return coef, r2, coef / se


def main() -> None:
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
    print(f"dataset={os.path.basename(args.npz)} n={n}/{len(raw)} k_depth={args.k_depth}")

    img_rows = []
    reg = {k: [] for k in ("logq", "err", "lum", "cast", "tau", "grad", "q")}

    for i in range(n):
        I = raw[i].astype(np.float64) / 255.0
        Jh = pred[i].astype(np.float64) / 255.0
        Jt = gt[i].astype(np.float64) / 255.0
        h, w = I.shape[:2]
        try:
            fit = S.fit_water(I, Jh, h, w, k_depth=args.k_depth, noise_mode="image")
            an = S.analyse(I, Jh, fit)
        except Exception as exc:  # a failed fit must be counted, not skipped silently
            print(f"  [{i:3d}] FIT FAILED {names[i]}: {type(exc).__name__}: {exc}")
            img_rows.append({"name": names[i], "ok": False, "error": str(exc)})
            continue

        logq = an["log10q_eff"].reshape(h, w, 3).mean(axis=2)
        q = an["q"].reshape(h, w, 3).mean(axis=2)
        tau = an["tau"].reshape(h, w, 3).mean(axis=2)
        err = np.abs(Jh - Jt).mean(axis=2)
        lum = I.mean(axis=2)
        cast = I[..., 0] - I[..., 2]

        img_rows.append({
            "name": names[i], "ok": True,
            "logq": float(logq.mean()),
            "psnr": float(10 * np.log10(1.0 / max(((Jh - Jt) ** 2).mean(), 1e-12))),
            "mae": float(err.mean()),
            "rmse": float(np.sqrt(((Jh - Jt) ** 2).mean())),
            "tau": float(tau.mean()),
            "chi2_dof": float(fit["chi2"] / max(fit["n_obs"], 1)),
        })

        reg["logq"].append(blocks(logq).ravel())
        reg["q"].append(blocks(q).ravel())
        reg["err"].append(blocks(err).ravel())
        reg["lum"].append(blocks(lum).ravel())
        reg["cast"].append(blocks(np.abs(cast)).ravel())
        reg["tau"].append(blocks(tau).ravel())
        reg["grad"].append(grad_energy(lum).ravel())

        if (i + 1) % 5 == 0:
            sub = [r for r in img_rows if r.get("ok")]
            if len(sub) > 2:
                rho = spearman(np.array([r["logq"] for r in sub]),
                               np.array([r["psnr"] for r in sub]))
                print(f"  [{i + 1:3d}/{n}] running image-level rho(logq, PSNR) = {rho:+.3f}")

    ok = [r for r in img_rows if r.get("ok")]
    print(f"\nimages ok={len(ok)} failed={len(img_rows) - len(ok)}")

    # ---------------- image level ----------------
    lq = np.array([r["logq"] for r in ok])
    ps = np.array([r["psnr"] for r in ok])
    mae = np.array([r["mae"] for r in ok])
    rho_psnr = spearman(lq, ps)
    rho_mae = spearman(lq, -mae)
    print(f"[image] rho(log10 q_eff, PSNR) = {rho_psnr:+.4f}   n={len(ok)}")
    print(f"[image] rho(log10 q_eff, -MAE) = {rho_mae:+.4f}")
    print(f"[image] spread: logq [{lq.min():.2f},{lq.max():.2f}]  "
          f"PSNR [{ps.min():.2f},{ps.max():.2f}]  MAE [{mae.min():.4f},{mae.max():.4f}]")

    # ---------------- region level ----------------
    R = {k: np.concatenate(v) for k, v in reg.items() if v}
    print(f"\n[region] {len(R['err'])} blocks of {BS}x{BS}")
    rho_raw = spearman(R["logq"], -R["err"])
    print(f"[region] rho(log10 q_eff, -error)          = {rho_raw:+.4f}")
    for k in ("lum", "cast", "tau", "grad"):
        print(f"[region] rho({k:4s}, -error)                = {spearman(R[k], -R['err']):+.4f}")

    # confound control: does support add anything over the cheap physical stats?
    # errors are heavy-tailed, so the regression is on ranks (a monotone question
    # is what the claim actually needs).
    y = rank(-R["err"])
    base_cols = [rank(R["lum"]), rank(R["cast"]), rank(R["tau"]), rank(R["grad"])]
    Xb = np.column_stack(base_cols)
    cb, r2b, tb = ols(y, Xb)
    Xf = np.column_stack(base_cols + [rank(R["logq"])])
    cf, r2f, tf = ols(y, Xf)
    print(f"\n[control] R2 without support = {r2b:.4f}")
    print(f"[control] R2 with    support = {r2f:.4f}   (increment {r2f - r2b:+.4f})")
    print(f"[control] support t-stat     = {tf[-1]:+.1f}   (coef {cf[-1]:+.3f})")
    print(f"[control] baseline t-stats   = {np.round(tb[1:], 1).tolist()} "
          f"(lum, cast, tau, grad)")

    # within-image control: the between-image variation is the easiest part, and
    # a region-level claim has to hold *inside* one photograph as well.
    wid = []
    for i in range(n):
        if not img_rows[i].get("ok"):
            continue
        e = reg["err"][len(wid)] if len(wid) < len(reg["err"]) else None
        if e is None:
            break
        wid.append(float(len(e)))
    # recompute properly: group by image index stored in order
    offs = np.cumsum([0] + [len(v) for v in reg["err"]])
    within = []
    for i in range(len(reg["err"])):
        sl = slice(offs[i], offs[i + 1])
        if offs[i + 1] - offs[i] < 8:
            continue
        lqi, ei = R["logq"][sl], R["err"][sl]
        if lqi.std() < 1e-9 or ei.std() < 1e-9:
            continue
        within.append(spearman(lqi, -ei))
    within = np.array(within)
    print(f"\n[within-image] rho per image: mean={within.mean():+.4f} "
          f"median={np.median(within):+.4f} frac<0 (support predicts error)="
          f"{(within < 0).mean():.3f}  n={len(within)}")

    out = {
        "npz": args.npz, "n_images": len(ok), "n_failed": len(img_rows) - len(ok),
        "blocks": BS, "k_depth": args.k_depth,
        "image_level": {"rho_logq_psnr": rho_psnr, "rho_logq_negmae": rho_mae},
        "region_level": {
            "rho_logq_negerr": rho_raw,
            "rho_lum": spearman(R["lum"], -R["err"]),
            "rho_cast": spearman(R["cast"], -R["err"]),
            "rho_tau": spearman(R["tau"], -R["err"]),
            "rho_grad": spearman(R["grad"], -R["err"]),
            "r2_base": r2b, "r2_full": r2f, "support_t": float(tf[-1]),
        },
        "within_image": {"mean": float(within.mean()), "median": float(np.median(within)),
                         "frac_negative": float((within < 0).mean()), "n": int(len(within))},
        "images": img_rows,
    }
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=1)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

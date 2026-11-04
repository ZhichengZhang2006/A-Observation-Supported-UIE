"""Why does a *noisier* observation appear to constrain a colour offset *more*?

`run_controlled.py --only q2` produced, on one image at 96x96:

    UNKNOWN nz= 1.0 width=0.5000 censored=True
    UNKNOWN nz= 2.0 width=0.5000 censored=True
    UNKNOWN nz= 4.0 width=0.0176 censored=False

Adding noise cannot turn an unconstrained direction into a tightly constrained
one.  Two candidate explanations, and they call for opposite responses:

(a) **The bound is real.**  Larger noise changes the fitted water enough that the
    offset becomes genuinely visible.  I cannot see a mechanism for this, and if
    it is real it is a significant result about the estimator.

(b) **The bound is an artefact.**  The profile objective becomes steep for a
    numerical reason -- a near-singular denominator in the implied-transmission
    solve (``T = (I/g - b) / (J - b)`` blows up as ``J_c -> b_c``), or the fit
    lands far from the optimum so the chi-square surface is dominated by model
    error rather than by the perturbation.  Then the "tight bound" measures the
    singularity, not the data.

The diagnostic that separates them: report the fit quality (``chi2/dof``) and the
margin ``min |J_c - b_c|`` alongside the width.  A tight width that comes with
``chi2/dof >> 1`` or a vanishing margin is (b): the estimator is reporting its own
breakdown as a measurement.

This is written as a script rather than as a paragraph because the answer decides
whether a number goes in the paper.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import obsui  # noqa: E402,F401

import numpy as np  # noqa: E402

import obsui.controlled as C  # noqa: E402
import obsui.support as S  # noqa: E402


def main():
    import cv2

    d = np.load("data/packed/UIEB.npz", allow_pickle=True)
    J = cv2.resize(d["gt"][0], (96, 96), interpolation=cv2.INTER_AREA)
    J = J.astype(np.float64) / 255.0
    h, w = J.shape[:2]

    v = np.zeros((h * w, 3))
    v[:, 0] = 1.0            # peak-normalised: a is the per-pixel channel offset

    print(f"image=116_img_ size={h}x{w}")
    print(f"{'nz':>5s} {'k':>3s} {'chi2/dof':>9s} {'b_inf':>22s} {'min|J-b|':>9s} "
          f"{'width':>8s} {'cens':>5s} {'rmse':>8s}")
    for nz in (1.0, 2.0, 4.0):
        cond = C.Condition(name=f"nz{nz:g}", read_sigma=0.004 * nz,
                           shot_alpha=0.008 * nz)
        obs = C.render(J, cond, np.random.RandomState(cond.seed + 17))
        I = obs["I"].astype(np.float64)
        for k in (4, 9):
            try:
                fit = S.fit_water(I, J, h, w, k_depth=k, noise_mode="declared",
                                  read_sigma=cond.read_sigma,
                                  shot_alpha=cond.shot_alpha)
                wm = fit["water"]
                chi2_dof = float(fit["chi2"] / max(fit["n_obs"], 1.0))
                # the margin that governs the singularity: how close is any
                # channel's scene value to the fitted veiling radiance?
                margin = float(np.abs(J - wm.b_inf.reshape(1, 1, 3)).min())
                pw = S.profile_width(I, J, fit, v, h, w, a_max=0.5, n_grid=6, df=1)
                print(f"{nz:5.1f} {k:3d} {chi2_dof:9.3f} "
                      f"[{wm.b_inf[0]:.3f},{wm.b_inf[1]:.3f},{wm.b_inf[2]:.3f}]"
                      f" {margin:9.4f} {pw['width']:8.4f} "
                      f"{str(pw['censored']):>5s} {fit['rmse']:8.4f}", flush=True)
            except Exception as exc:
                print(f"{nz:5.1f} {k:3d} FAILED {type(exc).__name__}: {exc}", flush=True)

    # The chi-square curve itself: is it steep because of information, or because
    # the surface has a kink?  A smooth quadratic is information; a jump is a
    # numerical boundary being crossed.
    print("\nchi2(a) curve at nz=4, k=9 (a is per-pixel channel offset):")
    cond = C.Condition(name="nz4", read_sigma=0.016, shot_alpha=0.032)
    obs = C.render(J, cond, np.random.RandomState(cond.seed + 17))
    I = obs["I"].astype(np.float64)
    fit = S.fit_water(I, J, h, w, k_depth=9, noise_mode="declared",
                      read_sigma=cond.read_sigma, shot_alpha=cond.shot_alpha)
    base = S.profile_chi2(I, J, fit, v, 0.0, h, w)
    print(f"  base chi2 = {base:.4f}  (threshold for 1 dof = {S.chi2_threshold(1):.4f})")
    for a in (0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5):
        c = S.profile_chi2(I, J, fit, v, a, h, w)
        print(f"  a={a:6.3f}  chi2={c:12.4f}  delta={c - base:12.4f}")


if __name__ == "__main__":
    main()

"""Diagnose the DC profile curve against depth variability.

Prints the raw profiled chi2 at a ladder of amplitudes so a non-monotone
`profile_width` can be traced to either the physics or the optimizer.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import obsui  # noqa: E402,F401

import numpy as np  # noqa: E402

import obsui.support as S  # noqa: E402
from tools.test_support import make_pair  # noqa: E402

H = W = 128
VDC = S.components(H, W, 4)["dc_0"]
GRID = (0.0, 0.02, 0.05, 0.1, 0.2, 0.4, 0.6)

print(f"{'dv':>5} {'rmse':>8} {'chi2/dof':>9} {'thresh':>7} " + " ".join(f"a={a:<5.2f}" for a in GRID))
for dv in (0.0, 0.5, 1.0, 1.5):
    I, J, d, wm, tau = make_pair(bd=(1.2, 0.85, 0.6), b_inf=(0.05, 0.08, 0.10),
                                 seed=7, depth_var=dv)
    fit = S.fit_water(I, J, H, W, noise_mode="declared")
    base = fit["chi2"]
    thresh = base + S.chi2_threshold(1)
    vals = []
    for a in GRID:
        c = S.profile_chi2(I, J, fit, VDC, a, H, W)
        vals.append(c - base)
    print(f"{dv:5.1f} {fit['rmse']:8.5f} {fit['chi2'] / max(fit['n_obs'], 1):9.3f} {thresh:7.2f} "
          + " ".join(f"{v:<7.2f}" for v in vals))

"""Is the non-monotone DC profile a physics result or a basis-resolution artefact?

`depth_basis(h, w, k)` is a k-cell bilinear hat basis.  If the true optical depth
varies faster than the basis can represent, the fit is misspecified (large
chi2/dof) and the profiled width is meaningless.  This script sweeps the basis
order against the depth variability to separate the two.
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

print(f"{'dv':>5} {'k':>3} {'rmse':>8} {'chi2/dof':>9} {'width':>7} {'cens':>6} {'lev':>7}")
for dv in (0.0, 0.5, 1.0, 1.5):
    for k in (4, 9, 16, 25):
        I, J, d, wm, tau = make_pair(bd=(1.2, 0.85, 0.6), b_inf=(0.05, 0.08, 0.10),
                                     seed=7, depth_var=dv)
        fit = S.fit_water(I, J, H, W, k_depth=k, noise_mode="declared")
        an = S.analyse(I, J, fit)
        lev = S.direction_info(an, VDC)["leverage"]
        r = S.profile_width(I, J, fit, VDC, H, W, a_max=0.6, n_grid=6)
        print(f"{dv:5.1f} {k:3d} {fit['rmse']:8.5f} {fit['chi2'] / max(fit['n_obs'], 1):9.3f} "
              f"{r['width']:7.4f} {str(r['censored']):>6} {lev:7.4f}")

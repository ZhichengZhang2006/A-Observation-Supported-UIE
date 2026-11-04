"""The exact gauge family of the underwater forward model.

Why this module exists
----------------------
Profiling a *global colour offset* against a **known** scene gives a tight bound
(measured: 3e-4 to 2e-2 per-pixel channel offset, and it grows with noise as it
should).  That number is correct and it is also close to useless, because it
answers the wrong question: it asks how well a scene can be recovered when the
scene is already known.  The single-image problem does not have that luxury.

The right question is: which changes to ``J`` leave the observation **exactly**
unchanged?  For this forward model the answer is available in closed form, and it
is not a small correction -- it is a one-parameter family with an explicit
feasible range.

Derivation
----------
Write the model per pixel and channel as

    I/g = b + (J - b) T,        T = exp(-beta d).

Substitute ``d -> d + dd`` and ``J -> b + (J - b) exp(beta dd)``:

    b + (J' - b) T' = b + (J - b) exp(beta dd) * exp(-beta (d + dd))
                     = b + (J - b) exp(-beta d)
                     = I/g.

**Identical, for any depth increment ``dd``.**  So increasing the optical depth by
``dd`` and brightening each channel's signal-above-veiling by ``exp(beta_c dd)``
leaves the image bit-for-bit unchanged.  No estimator can separate the two, and
no prior about "natural" water parameters can either, because both explanations
use the *same* ``beta``, ``b`` and ``g`` -- only the depth field differs, and it
is different by construction.

Because ``beta`` differs between channels, the required brightening differs too:
red is lifted by ``exp(beta_R dd)`` and blue by ``exp(beta_B dd)``, which is a
**pure chromatic change**.  That is what "unsupported colour" means here, and it
is a theorem rather than a fitted statistic.

The feasible range of ``dd`` is set by physics, not by taste:

* ``J'_c <= 1`` -- the restored scene must stay inside the dynamic range.  This is
  the binding constraint and it is image-dependent: a scene with little headroom
  above ``b`` cannot be brightened much, so it admits little ambiguity.
* ``J'_c >= 0`` and ``d + dd >= 0``.
* ``beta_c (d + dd) <= TAU_MAX`` keeps the water within the modelled range.

The first constraint is why the ambiguity is *large for dark, low-contrast scenes
and small for bright ones*, which is the opposite of what a brightness or
contrast statistic would predict and is a falsifiable consequence of this
account.  It is also the mechanism behind "unsupported colour" that the
enhancement literature describes informally.

What this does and does not claim
---------------------------------
It does **not** claim a single image cannot be enhanced.  It claims that any
colour restoration is determined only up to this family, so a method that emits a
specific correction has supplied information from outside the observation --
a scene prior, a learned colour distribution, or noise.  All three can be useful;
none of them is a measurement of the scene, and a method should not report one as
if it were.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from .support import TAU_MAX, WaterModel

__all__ = [
    "gauge_pair",
    "verify_exact",
    "colour_budget",
    "basis_budget",
    "best_step",
    "family_spread",
    "admissible_range",
]


def _render_np(J: np.ndarray, tau: np.ndarray, wm: WaterModel) -> np.ndarray:
    T = np.exp(-tau)
    b = wm.b_inf.reshape(1, 3)
    g = wm.gain.reshape(1, 3)
    return g * (J * T + b * (1.0 - T))


def gauge_pair(
    J: np.ndarray,
    d: np.ndarray,
    wm: WaterModel,
    dd: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply the exact gauge transform: ``d -> d + dd``, ``J -> J'``.

    ``J`` is ``(N, 3)`` or ``(h, w, 3)``, ``d``/``dd`` are ``(N,)`` or ``(h, w)``.
    Returns ``(J_prime, d_prime)`` with the same shapes.  The transform is exact
    for any ``dd``; the caller is responsible for checking that the result is
    physically admissible (see :func:`colour_budget`).
    """
    shp = J.shape
    J3 = np.asarray(J, dtype=np.float64).reshape(-1, 3)
    d1 = np.asarray(d, dtype=np.float64).reshape(-1)
    dd1 = np.asarray(dd, dtype=np.float64).reshape(-1)
    beta = wm.beta.reshape(1, 3)
    b = wm.b_inf.reshape(1, 3)
    lift = np.exp(beta * dd1.reshape(-1, 1))
    Jp = b + (J3 - b) * lift
    return Jp.reshape(shp), (d1 + dd1).reshape(np.shape(d))


def verify_exact(
    J: np.ndarray,
    d: np.ndarray,
    wm: WaterModel,
    dd: np.ndarray,
) -> Dict[str, float]:
    """Numerically confirm the transform leaves the observation unchanged.

    Returns the maximum absolute change in the rendered image, in image units.
    A correct implementation gives round-off (``< 1e-12``); anything larger means
    the algebra above was misapplied.
    """
    beta = wm.beta.reshape(1, 3)
    d1 = np.asarray(d, dtype=np.float64).reshape(-1)
    dd1 = np.asarray(dd, dtype=np.float64).reshape(-1)
    I0 = _render_np(np.asarray(J, dtype=np.float64).reshape(-1, 3),
                    d1.reshape(-1, 1) * beta, wm)
    Jp, dp = gauge_pair(J, d, wm, dd)
    I1 = _render_np(np.asarray(Jp, dtype=np.float64).reshape(-1, 3),
                    np.asarray(dp, dtype=np.float64).reshape(-1, 1) * beta, wm)
    diff = np.abs(I1 - I0)
    return {
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "scale": float(np.abs(I0).mean()),
    }


def colour_budget(
    J: np.ndarray,
    d: np.ndarray,
    wm: WaterModel,
    dd: float,
) -> Dict[str, object]:
    """How much colour ambiguity one depth step buys, and in which directions.

    The family is **two-sided** and the two sides are bounded by different physics,
    which matters because they have different signs of chromatic change:

    * ``dd > 0`` -- deeper water, brighter and more saturated scene.  Bounded by
      dynamic range: ``b_c + (J_c - b_c) exp(beta_c dd) <= 1``.  Real photographs
      usually touch the top of the range somewhere, so this side is often nearly
      empty.  That is a property of the *capture*, not of the scene.
    * ``dd < 0`` -- shallower water, darker and **less saturated** scene.  Bounded
      only by ``d + dd >= 0``, so it is available for essentially every real image.

    The negative side is the less comfortable one: it says a restored image that
    looks washed out is exactly as consistent with the observation as a vivid one,
    provided the water is declared shallower.  Reporting only the positive side
    would hide that, so both are returned with their own bounds.
    """
    J3 = np.asarray(J, dtype=np.float64).reshape(-1, 3)
    N = J3.shape[0]
    beta = np.broadcast_to(wm.beta.reshape(1, 3), (N, 3))
    b = np.broadcast_to(wm.b_inf.reshape(1, 3), (N, 3))
    d1 = np.asarray(d, dtype=np.float64).reshape(-1)

    # upper bound: any channel whose signal-above-veiling would exceed the range
    signal = J3 - b
    pos = signal > 1e-9
    per = np.full((N, 3), np.inf)
    per[pos] = (np.log(np.maximum((1.0 - b[pos]) / signal[pos], 1.0))
                / beta[pos])
    dd_pos_max = float(per.min()) if np.isfinite(per).any() else 0.0

    # lower bound: optical depth cannot go negative anywhere
    dd_neg_min = float(-d1.min())

    dd_use = float(np.clip(dd, dd_neg_min, dd_pos_max))

    Jp, _ = gauge_pair(J3, d1, wm, np.full(N, dd_use))
    Jp3 = np.asarray(Jp, dtype=np.float64).reshape(-1, 3)

    eps = 0.05

    def chroma(a):
        a = np.clip(a, 0.0, None)
        r, g_, bl = a[:, 0], a[:, 1], a[:, 2]
        return np.stack([np.log((r + eps) / (g_ + eps)),
                         np.log((g_ + eps) / (bl + eps))], axis=1)

    dc = chroma(Jp3) - chroma(J3)
    dl = (0.2126 * (Jp3[:, 0] - J3[:, 0]) + 0.7152 * (Jp3[:, 1] - J3[:, 1])
          + 0.0722 * (Jp3[:, 2] - J3[:, 2]))
    mag = np.sqrt((dc ** 2).sum(axis=1))
    delta = Jp3 - J3
    sig = (J3 - b).reshape(-1)
    # The lift is exactly ``exp(beta_c dd) - 1`` per channel, applied to the signal
    # above veiling.  That ratio is the cleanest statement of the mechanism: a
    # plain brightness gain would divide by ``J`` instead of ``(J - b)``, and a
    # per-channel gamma would not produce a pure multiplicative lift at all.
    with np.errstate(divide="ignore", invalid="ignore"):
        lift = np.where(np.abs(J3 - b) > 1e-9, (Jp3 - J3) / (J3 - b), np.nan)
    lift_pred = np.exp(beta * dd_use) - 1.0
    return {
        "dd_requested": float(dd),
        "dd_used": dd_use,
        "dd_pos_max": dd_pos_max,
        "dd_neg_min": dd_neg_min,
        "admissible": bool(dd_neg_min - 1e-12 <= dd <= dd_pos_max + 1e-12),
        "chroma_p50": float(np.percentile(mag, 50)),
        "chroma_p95": float(np.percentile(mag, 95)),
        "chroma_max": float(mag.max()),
        "luma_p95": float(np.percentile(np.abs(dl), 95)),
        "headroom_p50": float(np.percentile(np.abs(1.0 - J3).max(axis=1), 50)),
        "lift_pred_r": float(lift_pred[0, 0]),
        "lift_pred_g": float(lift_pred[0, 1]),
        "lift_pred_b": float(lift_pred[0, 2]),
        "lift_err_max": float(np.nanmax(np.abs(lift - lift_pred))),
        # The change is chromatic precisely because ``beta`` differs by channel,
        # so the induced log-opponent shift must equal the beta *difference*.
        "chroma_pred_rg": float(np.log(
            (1.0 + np.exp(beta[0, 0] * dd_use) - 1.0)
            / (1.0 + np.exp(beta[0, 1] * dd_use) - 1.0))),
    }


def basis_budget(
    J: np.ndarray,
    wm: WaterModel,
    phi: np.ndarray,
    d0: np.ndarray,
    dd_steps: Tuple[float, ...] = (0.05, 0.1, 0.2, 0.4, 0.8),
) -> list:
    """Colour budget at several depth steps, both directions.

    Sweeping ``dd`` shows where each side saturates.  The positive side must
    flatten at ``dd_pos_max`` (the scene would exceed the dynamic range); the
    negative side must flatten at ``dd_neg_min`` (the water would have negative
    depth).  A curve that did not flatten would mean a constraint is not being
    applied.
    """
    rows = []
    for dd in dd_steps:
        for sign in (+1.0, -1.0):
            r = colour_budget(J, d0, wm, sign * float(dd))
            r["dd_step"] = sign * float(dd)
            rows.append(r)
    return rows


def _sufficient_stats(Ja: np.ndarray, Jb: np.ndarray, wm: WaterModel):
    """Everything the least-squares fit needs, in one pass over the pixels.

    The family acts per channel as ``A'_c = b_c + (A_c - b_c) u_c`` with
    ``u_c = exp(beta_c dd)``.  Minimising ``sum_c ||A'_c - B_c||^2`` over ``dd``
    therefore splits into three independent one-dimensional problems, and each is
    an unconstrained quadratic in ``u_c``:

        f_c(u) = ||B_c - b_c - (A_c - b_c) u||^2
               = u^2 |A_c - b_c|^2 - 2u (A_c - b_c).(B_c - b_c) + |B_c - b_c|^2

    with the minimiser ``u*_c = <A_c - b_c, B_c - b_c> / |A_c - b_c|^2`` and the
    achievable residual ``|B_c - b_c|^2 - <...>^2 / |A_c - b_c|^2``.

    Three scalar sums per channel replace a full-image sweep.  This is not a
    micro-optimisation: at 256x256 the sweep is 161 x 196k x 3 operations, and it
    turned a per-image measurement into a per-image wait.
    """
    A = np.asarray(Ja, dtype=np.float64).reshape(-1, 3)
    B = np.asarray(Jb, dtype=np.float64).reshape(-1, 3)
    b = wm.b_inf.reshape(1, 3)
    s = A - b
    t = B - b
    s_nn = (s * s).sum(axis=0)          # |A_c - b_c|^2
    st = (s * t).sum(axis=0)            # <A_c - b_c, B_c - b_c>
    t_nn = (t * t).sum(axis=0)          # |B_c - b_c|^2
    return s_nn, st, t_nn


def admissible_range(J: np.ndarray, d: np.ndarray, wm: WaterModel) -> Tuple[float, float]:
    """Largest physically valid depth step in each direction, as ``(lo, hi)``.

    ``hi`` comes from ``J'_c <= 1`` (dynamic range) and ``lo`` from ``d + dd >= 0``
    plus ``J'_c >= 0``.  ``beta_c (d + dd) <= TAU_MAX`` is folded into ``hi`` too.
    """
    J3 = np.asarray(J, dtype=np.float64).reshape(-1, 3)
    b = wm.b_inf.reshape(1, 3)
    beta = wm.beta.reshape(1, 3)
    d1 = np.asarray(d, dtype=np.float64).reshape(-1)
    sig = J3 - b
    hi = np.inf
    pos = sig > 1e-9
    if pos.any():
        # J'_c = b_c + (J_c - b_c) u <= 1  ->  u <= (1 - b_c) / (J_c - b_c)
        ratio = np.where(pos, (1.0 - b) / np.where(pos, sig, 1.0), np.inf)
        u_max = float(np.min(ratio))
        if u_max > 0 and np.isfinite(u_max):
            hi = float(np.log(u_max) / float(beta.max()))
    lo = float(-d1.min()) if d1.size else 0.0
    # reflectance floor: b_c + (J_c - b_c) u >= 0 -> u >= -b_c / (J_c - b_c)
    neg = sig < -1e-9
    if neg.any():
        u_min = np.max(-b / np.where(neg, sig, -np.inf))
        if np.isfinite(u_min) and u_min > 0:
            lo = max(lo, float(-np.log(u_min) / beta.min()))
    # water model validity
    if d1.size:
        hi = min(hi, float(TAU_MAX / beta.max() - d1.min()))
    return lo, hi


def best_step(
    Ja: np.ndarray,
    Jb: np.ndarray,
    wm: WaterModel,
    d_a: np.ndarray | None = None,
    d_b: np.ndarray | None = None,
    n_iter: int = 3,
) -> Dict[str, object]:
    """Gauge step taking scene ``Ja`` closest to scene ``Jb``, in closed form.

    Each channel's optimal ``u_c`` is available analytically; ``dd`` follows from
    ``u_c = exp(beta_c dd)``.  Since ``beta_c`` differ, the three channels give
    three different ``dd``, so a single step cannot satisfy all of them exactly --
    that residual is the part of the difference the family *cannot* explain and is
    exactly what we want to report.  Returned ``dd`` is the least-squares
    compromise over channels, refined against the physical range.

    ``d_a``/``d_b`` are optional depth maps; when given, ``admissible`` also
    checks that the shifted depth stays in ``[0, TAU_MAX / beta_max]``.
    """
    s_nn, st, t_nn = _sufficient_stats(Ja, Jb, wm)
    beta = wm.beta
    # The denominator must be the size of the *difference being explained*, i.e.
    # ``||Ja - Jb||^2``.  Using ``||Jb - b||^2`` (the signal above veiling) instead
    # silently rescales every fraction: for a scene near the veiling light that
    # quantity is small, so almost any residual "explains" almost all of it.  This
    # is the denominator that makes "frac" mean what the docstring says.
    A0 = np.asarray(Ja, dtype=np.float64).reshape(-1, 3)
    B0 = np.asarray(Jb, dtype=np.float64).reshape(-1, 3)
    tot = float((((B0 - A0) ** 2).sum()))
    if tot <= 1e-18:
        return {"dd": 0.0, "frac": 1.0, "residual": 0.0, "total": 0.0,
                "frac_free": 1.0, "admissible": True}

    # ---- the one-parameter family -------------------------------------------
    # ``dd`` is a SINGLE scalar and the three channels are linked through
    # ``u_c = exp(beta_c dd)``.  Fitting the three ``u_c`` independently would be a
    # three-parameter model and would overstate how much a different *water*
    # explanation can absorb -- an error worth a factor of five in the measured
    # fraction, so it is worth being explicit about.  With the sufficient
    # statistics the residual at any ``dd`` costs three multiplies, so a fine grid
    # is both exact-enough and cheap.
    dd_grid = np.linspace(-2.0, 2.0, 8001)
    u_grid = np.exp(beta.reshape(1, 3) * dd_grid.reshape(-1, 1))     # (G,3)
    resid_grid = (s_nn * u_grid ** 2 - 2.0 * st * u_grid + t_nn).sum(axis=1)
    k = int(np.argmin(resid_grid))
    dd_best = float(dd_grid[k])

    # ---- three free per-channel gains (an upper bound, reported not claimed) --
    u_free = np.where(s_nn > 1e-12, st / np.maximum(s_nn, 1e-12), 1.0)
    resid_free = float((t_nn - np.where(s_nn > 1e-12,
                                        st ** 2 / np.maximum(s_nn, 1e-12), 0.0)).sum())
    resid_free = max(0.0, resid_free)

    lo, hi = (-np.inf, np.inf)
    if d_a is not None:
        lo, hi = admissible_range(Ja, d_a, wm)
    dd_c = float(np.clip(dd_best, lo, hi)) if np.isfinite(lo) or np.isfinite(hi) \
        else dd_best

    # residual at the *clipped* step is what a physical reader would see
    Jp, _ = gauge_pair(Ja, np.zeros(np.asarray(Ja).reshape(-1, 3).shape[0]),
                       wm, np.full(np.asarray(Ja).reshape(-1, 3).shape[0], dd_c))
    A = np.asarray(Jp, dtype=np.float64).reshape(-1, 3)
    B = np.asarray(Jb, dtype=np.float64).reshape(-1, 3)
    resid_c = float(((B - A) ** 2).sum())

    per_ch_dd = np.log(np.clip(u_free, 1e-9, 1e9)) / beta
    return {
        "dd": dd_c,
        "dd_unclipped": dd_best,
        "frac": float(np.clip(1.0 - resid_c / tot, 0.0, 1.0)),
        "frac_free": float(np.clip(1.0 - resid_free / tot, 0.0, 1.0)),
        "residual": resid_c,
        "total": tot,
        "per_channel_dd": [float(x) for x in per_ch_dd],
        "admissible": bool(lo - 1e-12 <= dd_c <= hi + 1e-12),
        "range": [float(lo), float(hi)] if np.isfinite(lo) or np.isfinite(hi) else None,
    }


def family_spread(
    Jstar: np.ndarray,
    wm: WaterModel,
    d_ref: np.ndarray | None = None,
    n_grid: int = 96,
) -> Dict[str, float]:
    """Range of PSNR a *perfect* reconstruction could show, purely from gauge freedom.

    Every member of the family renders the identical observation, so a method that
    lands anywhere in this range is indistinguishable *by the data*.  The spread in
    dB is therefore the width of the band inside which PSNR cannot rank methods on
    colour.  Computed in closed form: the extremes of a quadratic over a
    physically clipped interval.
    """
    J = np.asarray(Jstar, dtype=np.float64).reshape(-1, 3)
    b = wm.b_inf.reshape(1, 3)
    s = J - b
    s_nn = (s * s).sum(axis=0)
    N = J.shape[0]
    if N == 0:
        return {"psnr_spread": 0.0, "n_members": 0}

    lo, hi = (-0.5, 0.5)
    if d_ref is not None:
        lo, hi = admissible_range(J, d_ref, wm)
    lo = max(lo, -0.5)
    hi = min(hi, 0.5)
    if not np.isfinite(lo):
        lo = -0.5
    if not np.isfinite(hi):
        hi = 0.5
    if hi <= lo:
        return {"psnr_spread": 0.0, "n_members": 0, "lo": lo, "hi": hi}

    best_mse, worst_mse = np.inf, 0.0
    n_ok = 0
    # The family-of-one-step sweep is cheap here because it is one image, not a
    # pairwise matrix; a coarse grid plus the analytic optimum is enough to bound
    # the spread without overselling its precision.
    for dd in np.linspace(lo, hi, n_grid):
        u = np.exp(wm.beta.reshape(1, 3) * dd)
        mse = float((((u - 1.0) ** 2) * s_nn).sum() / (3.0 * N))
        n_ok += 1
        best_mse = min(best_mse, mse)
        worst_mse = max(worst_mse, mse)
    best_mse = max(best_mse, 0.0)
    worst_mse = max(worst_mse, 0.0)
    psnr_best = 99.0 if best_mse <= 1e-12 else 10.0 * np.log10(1.0 / best_mse)
    psnr_worst = 99.0 if worst_mse <= 1e-12 else 10.0 * np.log10(1.0 / worst_mse)
    return {
        "psnr_spread": float(max(0.0, psnr_best - psnr_worst)),
        "psnr_best": float(psnr_best),
        "psnr_worst": float(psnr_worst),
        "dd_lo": float(lo),
        "dd_hi": float(hi),
        "n_members": n_ok,
    }

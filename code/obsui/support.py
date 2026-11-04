"""Concentrated-likelihood water model: fit, direction-resolved support, ambiguity.

Forward model (declared, per pixel, 3 channels)::

    I_c = g_c ( J_c * T_c + b_c (1 - T_c) ) + n_c ,   T_c = exp(-tau_c)
    tau_c(p) = beta_c * d(p)                            (one depth, per-channel slope)

``n`` is independent Gaussian with ``sigma_c(p) = sqrt(read^2 + shot^2 I_c(p))``.

Two things a scalar "transmission threshold" cannot express
-----------------------------------------------------------
1. **A global colour offset is (almost) unidentifiable.**  Raising ``J_c`` by a
   constant and raising the veiling radiance ``b_c`` by ``const * T/(1-T)``
   leaves ``I`` unchanged; when ``T`` is spatially constant the two are *exactly*
   confounded, so the DC colour component carries no information at any
   transmission.  A global gain rescaling is likewise absorbed by ``g``.
2. **Local structure is identifiable.**  A pixel- or edge-scale change in ``J``
   cannot be reproduced by any global or smooth nuisance parameter, so it keeps
   nearly all of its raw information.

Both are quantified by the same object: for a unit-energy direction ``v`` in
``J``-space, ``q_v = ||Sigma_n^{-1/2} D_J G v||^2`` is the raw Fisher information
and ``q_v^eff = q_v (1 - h_v)`` with ``h_v = a_v^T (A^T A)^{-1} a_v`` the
Schur-complement (nuisance-profiled) information.  ``h_v`` is the *leverage* of
the nuisance parameters on that direction -- literally the fraction of the
change the imaging model can explain away.

Identifiability, measured not assumed
-------------------------------------
Rather than trusting the linearisation, the ambiguity of a component is measured
by **profiling**: move ``J`` along ``v`` by amplitude ``a``, re-optimise every
nuisance parameter, and find the largest ``a`` whose likelihood penalty stays
inside the 95% chi-square quantile.  That width is the reported ambiguity, so a
direction whose linearised standard deviation is meaningless (because the
constraint box, not the likelihood, bounds it) is reported as such instead of
emitting a fantasy number.

Gauge: only ``tau_c = beta_c d`` enters the observation, so ``beta`` and ``d``
are never separately reported.  ``beta`` is normalised to ``mean(beta) = 1`` and
the optical depth is what gets published.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple


def _pin_threads(n: int = 1) -> None:
    """Pin BLAS/OMP to one thread before numpy is imported.

    This machine exposes 96 cores (sharing a 20 GB GPU slice).  A ``(4096, 9)``
    product is far too small to amortise thread synchronisation, and the
    oversubscription made every solve ~18x slower:
    measured 5.386 s -> 0.289 s per 200 normal-equation solves.
    """
    for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(var, str(n))


_pin_threads()

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

try:
    from scipy.stats import chi2 as _chi2
except Exception:  # pragma: no cover
    _chi2 = None

TAU_MAX = 3.5
CONF = 0.95
Q_REF = 4.0
CHI2_1DOF = None  # filled lazily


def _chi2_ppf(p: float, df: int = 1) -> float:
    global CHI2_1DOF
    if _chi2 is not None:
        return float(_chi2.ppf(p, df))
    return 3.841458820694124 if df == 1 else float(df) * 2.0


# ==========================================================================
# parameters
# ==========================================================================
@dataclass
class WaterModel:
    """``tau_c = beta_c * d`` with ``mean(beta) = 1``."""

    beta: np.ndarray
    gain: np.ndarray
    b_inf: np.ndarray

    def as_vector(self) -> np.ndarray:
        return np.concatenate([self.beta, self.gain, self.b_inf])

    @staticmethod
    def from_vector(v: np.ndarray) -> "WaterModel":
        v = np.asarray(v, dtype=np.float64)
        return WaterModel(v[0:3].copy(), v[3:6].copy(), v[6:9].copy())

    def to_dict(self) -> dict:
        return {
            "beta": self.beta.tolist(),
            "gain": self.gain.tolist(),
            "b_inf": self.b_inf.tolist(),
        }


def default_water(b_inf: Optional[np.ndarray] = None) -> WaterModel:
    beta = np.array([0.62, 0.54, 0.45])
    beta = beta / beta.mean()
    b = np.array([0.10, 0.18, 0.25]) if b_inf is None else np.asarray(b_inf, float)
    return WaterModel(beta, np.ones(3), b)


def render(J: np.ndarray, tau: np.ndarray, wm: WaterModel) -> np.ndarray:
    J = np.asarray(J, dtype=np.float64).reshape(-1, 3)
    tau = np.asarray(tau, dtype=np.float64)
    if tau.ndim == 1:
        tau = np.repeat(tau.reshape(-1, 1), 3, axis=1)
    T = np.exp(-tau)
    return wm.gain.reshape(1, 3) * (J * T + wm.b_inf.reshape(1, 3) * (1.0 - T))


def invert(I: np.ndarray, tau: np.ndarray, wm: WaterModel) -> np.ndarray:
    I = np.asarray(I, dtype=np.float64).reshape(-1, 3)
    tau = np.asarray(tau, dtype=np.float64)
    if tau.ndim == 1:
        tau = np.repeat(tau.reshape(-1, 1), 3, axis=1)
    T = np.exp(-tau)
    return np.clip(
        (I / wm.gain.reshape(1, 3) - wm.b_inf.reshape(1, 3) * (1.0 - T)) / T, 0.0, 1.5
    )


def tau_from_ratio(I: np.ndarray, J: np.ndarray, wm: WaterModel, eps: float = 1e-3) -> np.ndarray:
    I = np.asarray(I, float).reshape(-1, 3) / wm.gain.reshape(1, 3)
    J = np.asarray(J, float).reshape(-1, 3)
    num = np.clip(I - wm.b_inf.reshape(1, 3), eps, None)
    den = np.clip(J - wm.b_inf.reshape(1, 3), eps, None)
    return np.clip(-np.log(num / den), 0.0, TAU_MAX)


# ==========================================================================
# smooth depth basis
# ==========================================================================
_PHI_CACHE: Dict[Tuple[int, int, int], np.ndarray] = {}


def depth_basis(h: int, w: int, k: int = 9) -> np.ndarray:
    """``(h*w, k)`` bilinear hat basis; ``k`` must be a perfect square."""
    m = int(round(np.sqrt(k)))
    if m * m != k:
        raise ValueError("k must be a perfect square")
    key = (h, w, k)
    if key in _PHI_CACHE:
        return _PHI_CACHE[key]
    ys = np.linspace(0.0, m - 1, h)
    xs = np.linspace(0.0, m - 1, w)
    gy, gx = np.meshgrid(ys, xs, indexing="ij")
    y0 = np.floor(gy).astype(int).clip(0, m - 1)
    x0 = np.floor(gx).astype(int).clip(0, m - 1)
    y1 = (y0 + 1).clip(0, m - 1)
    x1 = (x0 + 1).clip(0, m - 1)
    wy = (gy - y0).ravel()
    wx = (gx - x0).ravel()
    phi = np.zeros((h * w, k))
    idx = np.arange(h * w)
    for yy, xx, ww in (
        (y0.ravel(), x0.ravel(), (1 - wy) * (1 - wx)),
        (y1.ravel(), x0.ravel(), wy * (1 - wx)),
        (y0.ravel(), x1.ravel(), (1 - wy) * wx),
        (y1.ravel(), x1.ravel(), wy * wx),
    ):
        np.add.at(phi, (idx, yy * m + xx), ww)
    _PHI_CACHE[key] = phi
    return phi


def _resize_f(a: np.ndarray, h: int, w: int, out: int) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    chans = []
    for c in range(a.shape[2]):
        im = Image.fromarray(a[:, :, c], mode="F")
        chans.append(np.asarray(im.resize((out, out), Image.BILINEAR), dtype=np.float32))
    return np.stack(chans, axis=-1)


# ==========================================================================
# noise
# ==========================================================================
def noise_sigma(I: np.ndarray, read_sigma: float = 0.010, shot_alpha: float = 0.020) -> np.ndarray:
    """Per-pixel per-channel std ``(N,3)``."""
    I = np.asarray(I, dtype=np.float64).reshape(-1, 3)
    return np.sqrt(read_sigma ** 2 + shot_alpha ** 2 * np.clip(I, 0.0, 1.0))


def laplacian_sigma(I: np.ndarray, k: float = 1.4826) -> float:
    """Noise level from the median absolute Laplacian -- no ground truth needed."""
    a = np.asarray(I, dtype=np.float64)
    if a.ndim == 2:
        a = a[:, :, None]
    acc = None
    for c in range(a.shape[2]):
        ch = a[:, :, c]
        lap = (
            4 * ch[1:-1, 1:-1]
            - ch[:-2, 1:-1]
            - ch[2:, 1:-1]
            - ch[1:-1, :-2]
            - ch[1:-1, 2:]
        )
        acc = lap if acc is None else acc + lap
    acc = acc / a.shape[2]
    return float(k * np.median(np.abs(acc)) / np.sqrt(20.0))


def _lsq_normal(A: np.ndarray, b: np.ndarray, ridge: float = 1e-10) -> np.ndarray:
    """Solve ``min ||A c - b||`` through the small normal equations.

    ``np.linalg.lstsq`` runs an SVD of the full ``(N, k)`` matrix on every call.
    With ``k <= 16`` the normal equations are exact to rounding and cheaper --
    but only when BLAS is not thrashing across dozens of cores, so the thread
    count is pinned at import time (see ``_pin_threads``).
    """
    G = A.T @ A
    rhs = A.T @ b
    tr = float(np.trace(G)) / max(G.shape[0], 1)
    G = G + max(ridge * tr, 1e-14) * np.eye(G.shape[0])
    try:
        return np.linalg.solve(G, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(A, b, rcond=None)[0]


def _pin_threads(n: int = 1) -> None:
    for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(var, str(n))


# ==========================================================================
# concentrated likelihood
# ==========================================================================
def _implied_tau(I: np.ndarray, J: np.ndarray, wm: WaterModel) -> Tuple[np.ndarray, np.ndarray]:
    """Per-pixel optical depth implied by the observation for each channel.

    Solved in closed form: ``T = (I/g - b)/(J - b)``.  Returns
    ``(tau (N,3), valid (N,3))``.
    """
    Ig = I / wm.gain.reshape(1, 3)
    num = Ig - wm.b_inf.reshape(1, 3)
    den = J - wm.b_inf.reshape(1, 3)
    valid = (num > 1e-4) & (den > 1e-4)
    ratio = np.clip(num / np.where(np.abs(den) < 1e-4, 1e-4, den), 1e-4, 1.0)
    tau = np.clip(-np.log(ratio), 0.0, TAU_MAX)
    return tau, valid


def _sigma_tau(I: np.ndarray, wm: WaterModel, sigma: np.ndarray) -> np.ndarray:
    """Error propagation ``d tau / d I = 1/(I - g b)``."""
    denom = np.abs(I - wm.gain.reshape(1, 3) * wm.b_inf.reshape(1, 3))
    denom = np.clip(denom, 0.02, None)
    return sigma / denom


def _concentrated(
    I: np.ndarray,
    J: np.ndarray,
    wm: WaterModel,
    phi: np.ndarray,
    sigma_shape: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Chi-square of the best smooth depth field for a given ``wm``.

    Returns ``(chi2, coeff, d)``.  The depth field is concentrated out in closed
    form, so only the nine water/sensor parameters are searched.

    ``valid_mask`` pins the set of pixels the chi-square is evaluated on.  This
    matters when comparing chi-square values across candidate ``J``: if the mask
    moved with ``J``, the comparison would mix "the model fits worse" with "more
    pixels dropped out".
    """
    tau_c, valid = _implied_tau(I, J, wm)
    if valid_mask is not None:
        valid = valid & valid_mask
    sd = _sigma_tau(I, wm, sigma_shape)
    beta = wm.beta.reshape(1, 3)
    target = tau_c / beta
    w = valid.astype(np.float64) / np.clip(sd / beta, 1e-6, None) ** 2
    sw = w.sum(axis=1)
    tw = (target * w).sum(axis=1)
    rhs = np.where(sw > 0, tw / np.maximum(sw, 1e-12), 0.0)
    A = phi * np.sqrt(np.maximum(sw, 0.0)).reshape(-1, 1)
    bvec = rhs * np.sqrt(np.maximum(sw, 0.0))
    coeff = _lsq_normal(A, bvec)
    d = phi @ coeff
    r = np.where(valid, (target - d.reshape(-1, 1)) / np.clip(sd / beta, 1e-6, None), 0.0)
    return float((r ** 2).sum()), coeff, d


def fit_water(
    I: np.ndarray,
    J: np.ndarray,
    h: int,
    w: int,
    k_depth: int = 9,
    work: int = 64,
    iters: int = 60,
    read_sigma: float = 0.010,
    shot_alpha: float = 0.020,
    noise_mode: str = "image",
    verbose: bool = False,
    init: Optional[WaterModel] = None,
) -> dict:
    """Fit the nine water/sensor parameters; the depth field is concentrated out.

    ``noise_mode``
      ``"declared"`` -- trust ``read_sigma``/``shot_alpha`` (controlled runs where
                        we injected the noise ourselves);
      ``"image"``    -- set the overall level from a model-free Laplacian estimate
                        on the observation (a real deployment has no ground truth
                        and no trustworthy sensor spec);
      ``"residual"`` -- set it from the fit residual (inflates with model error,
                        so it is reported but not used by default);
      ``"misfit"``   -- like ``"image"``, then rescale so ``chi2/dof = 1``.

    Why ``"misfit"`` exists
    -----------------------
    Measured on real UIEB frames, ``chi2/dof`` under the sensor-noise model comes
    out at 5.8e5 to 7.4e7.  That is not a noise estimate, it is a statement that
    the water model explains a fraction of a percent of the residual variance.  If
    the Fisher information is computed against sensor noise while the residual is
    dominated by model error, every direction looks infinitely well determined and
    the support map collapses to a constant -- which is exactly what was observed
    (within-image sd 0.005 against a mean of 0.038, and four different spatial
    patterns producing identical ablation numbers).

    The standard remedy is to scale the covariance by the observed chi-square per
    degree of freedom, i.e. to treat unexplained model error as part of the noise.
    Then ``q_v`` answers "how much information per unit of *total* unexplained
    variance", which is the only version that is honest on real data.  Which mode
    produced a number is always recorded in the returned dict, because the two
    modes answer different questions and quoting one as the other would be the
    kind of substitution this project is supposed to catch.
    """
    I = np.asarray(I, dtype=np.float64).reshape(h, w, 3)
    J = np.asarray(J, dtype=np.float64).reshape(h, w, 3)
    Iw = _resize_f(I, h, w, work).reshape(-1, 3).astype(np.float64)
    Jw = _resize_f(J, h, w, work).reshape(-1, 3).astype(np.float64)
    phi_w = depth_basis(work, work, k_depth)
    sigma_w = noise_sigma(Iw, read_sigma, shot_alpha)
    if noise_mode == "image":
        est = laplacian_sigma(I)
        declared = float(np.sqrt(read_sigma ** 2 + shot_alpha ** 2 * 0.35))
        sigma_w = sigma_w * float(np.clip(est / max(declared, 1e-6), 0.25, 8.0))

    wm = default_water(np.clip(np.percentile(Iw, 2.0, axis=0), 0.0, 0.5)) if init is None else init
    theta = wm.as_vector()
    cost = _concentrated(Iw, Jw, wm, phi_w, sigma_w)[0]

    lam = 1e-2
    n_p = 9
    for it in range(iters):
        wm_c = WaterModel.from_vector(theta)
        _, _, d0 = _concentrated(Iw, Jw, wm_c, phi_w, sigma_w)
        tau_c, valid0 = _implied_tau(Iw, Jw, wm_c)
        sd = _sigma_tau(Iw, wm_c, sigma_w)
        beta = wm_c.beta.reshape(1, 3)
        sdv = np.clip(sd / beta, 1e-6, None)
        r = (np.where(valid0, tau_c / beta - d0.reshape(-1, 1), 0.0) / sdv).ravel()
        # numeric Jacobian of the whitened residual w.r.t. the nine parameters;
        # the valid mask is frozen so the comparison is like-for-like
        A = np.zeros((3 * Iw.shape[0], n_p))
        eps = 1e-5
        for j in range(n_p):
            th2 = theta.copy()
            th2[j] += eps
            wm2 = WaterModel.from_vector(th2)
            _, _, d2 = _concentrated(Iw, Jw, wm2, phi_w, sigma_w, valid_mask=valid0)
            tau2, _ = _implied_tau(Iw, Jw, wm2)
            sd2 = _sigma_tau(Iw, wm2, sigma_w)
            b2 = wm2.beta.reshape(1, 3)
            rr = (np.where(valid0, tau2 / b2 - d2.reshape(-1, 1), 0.0) / np.clip(sd2 / b2, 1e-6, None)).ravel()
            A[:, j] = (rr - r) / eps
        JtJ = A.T @ A
        Jtr = A.T @ r
        diag = np.diag(JtJ).copy()
        diag[diag <= 1e-9] = 1e-9
        try:
            delta = np.linalg.solve(JtJ + lam * np.diag(diag), -Jtr)
        except np.linalg.LinAlgError:
            lam *= 10.0
            continue
        cand = theta + delta
        cand[:3] = np.clip(cand[:3], 0.05, 6.0)
        cand[3:6] = np.clip(cand[3:6], 0.2, 5.0)
        cand[6:9] = np.clip(cand[6:9], 0.0, 0.85)
        wm_new = WaterModel.from_vector(cand)
        c_new = _concentrated(Iw, Jw, wm_new, phi_w, sigma_w)[0]
        if np.isfinite(c_new) and c_new < cost:
            step = cost - c_new
            theta, cost = cand, c_new
            lam = max(lam * 0.5, 1e-8)
            if verbose:
                print(f"  it{it:02d} chi2={cost:.4f} lam={lam:.1e}")
            if step < 1e-9 * max(cost, 1e-9):
                break
        else:
            lam *= 6.0
            if lam > 1e8:
                break

    wm_f = WaterModel.from_vector(theta)
    scale = float(wm_f.beta.mean())
    beta_n = wm_f.beta / scale
    wm_f = WaterModel(beta_n, wm_f.gain, wm_f.b_inf)

    sig = noise_sigma(I.reshape(-1, 3), read_sigma, shot_alpha)
    if noise_mode == "image":
        est = laplacian_sigma(I)
        declared = float(np.sqrt(read_sigma ** 2 + shot_alpha ** 2 * 0.35))
        sig = sig * float(np.clip(est / max(declared, 1e-6), 0.25, 8.0))
        sc = float(np.sqrt((sig ** 2).mean() / max((noise_sigma(I.reshape(-1, 3), read_sigma, shot_alpha) ** 2).mean(), 1e-12)))
    else:
        sc = 1.0

    phi_full = depth_basis(h, w, k_depth)
    # Polish at full resolution.  The parameter search above runs on the work
    # grid for speed, so its optimum is the work-grid optimum; the profiler and
    # every reported chi-square live at full resolution.  Leaving the two
    # different would mean the ambiguity is measured against a point that is not
    # even a local optimum of the objective it is evaluated on.
    cost_full, wm_f = _refit_params(I.reshape(-1, 3), J.reshape(-1, 3), wm_f, phi_full, sig, iters=8)
    beta_n = wm_f.beta / max(float(wm_f.beta.mean()), 1e-9)
    wm_f = WaterModel(beta_n, wm_f.gain, wm_f.b_inf)

    tau_c, valid = _implied_tau(I.reshape(-1, 3), J.reshape(-1, 3), wm_f)
    sd_full = _sigma_tau(I.reshape(-1, 3), wm_f, sig)
    wgt = valid / np.clip(sd_full / beta_n.reshape(1, 3), 1e-6, None) ** 2
    sw = wgt.sum(axis=1)
    tw = ((tau_c / beta_n.reshape(1, 3)) * wgt).sum(axis=1)
    sq = np.sqrt(np.maximum(sw, 0.0))
    coeff_f = _lsq_normal(phi_full * sq.reshape(-1, 1), (tw / np.maximum(sw, 1e-12)) * sq)
    d_full = np.clip(phi_full @ coeff_f, 0.0, TAU_MAX)
    tau_full = d_full.reshape(-1, 1) * beta_n.reshape(1, 3)

    resid = render(J.reshape(-1, 3), tau_full, wm_f) - I.reshape(-1, 3)
    if noise_mode == "residual":
        r2 = float((resid ** 2).mean())
        s2 = float((sig ** 2).mean())
        sc = float(np.clip(np.sqrt(max(r2, 1e-12) / max(s2, 1e-12)), 0.25, 8.0))
        sig = sig * sc

    n_dof = float(3 * I.shape[0] * I.shape[1] - 9 - k_depth)
    if noise_mode == "misfit":
        # Inflate the noise to absorb model error, so chi2/dof = 1 by construction
        # and the information is measured per unit of *total* unexplained variance.
        chi2_raw = float(((resid / sig) ** 2).sum())
        infl = float(np.sqrt(max(chi2_raw, 1e-12) / max(n_dof, 1.0)))
        sig = sig * infl
        sc = sc * infl
    chi2 = float(((resid / sig) ** 2).sum())

    # The concentrated chi-square used by the profiler must be on the same scale
    # as ``chi2`` above, otherwise the chi-square quantile is applied to a number
    # of a different magnitude.  Rescale it by their ratio at a = 0.
    conc0 = _concentrated(I.reshape(-1, 3), J.reshape(-1, 3), wm_f, phi_full, sig)[0]
    conc_scale = float(chi2 / conc0) if conc0 > 1e-12 else 1.0

    return {
        "water": wm_f,
        "coeff": coeff_f,
        "phi": phi_full,
        "k_depth": k_depth,
        "depth": d_full,
        "tau": tau_full,
        "rmse": float(np.sqrt((resid ** 2).mean())),
        "sigma": sig,
        "sigma_scale": sc,
        "chi2": chi2,
        "conc_scale": conc_scale,
        "n_obs": float(3 * I.shape[0] * I.shape[1] - 9 - k_depth),
        "dof": float(3 * I.shape[0] * I.shape[1] - 9 - k_depth),
        "chi2_over_dof": float(chi2 / max(n_dof, 1.0)),
        "cost": cost_full,
        "work": work,
        "noise_mode": noise_mode,
    }


def chi2_threshold(df: int = 1) -> float:
    return _chi2_ppf(CONF, df)


# ==========================================================================
# Object 1 -- direction-resolved support
# ==========================================================================
def jacobian(I: np.ndarray, J: np.ndarray, fit: dict) -> np.ndarray:
    """``A = Sigma_n^{-1/2} D_theta G`` at the fitted point, ``(3N, 9+k)``.

    Column order: ``beta(3), gain(3), b_inf(3), depth coeffs(k)``.
    """
    I = np.asarray(I, dtype=np.float64).reshape(-1, 3)
    J = np.asarray(J, dtype=np.float64).reshape(-1, 3)
    wm: WaterModel = fit["water"]
    phi = fit["phi"]
    d = fit["depth"]
    sig = fit["sigma"].ravel()
    T = np.exp(-fit["tau"])
    n = I.shape[0]
    A = np.zeros((3 * n, 9 + phi.shape[1]))
    for c in range(3):
        Jc = J[:, c]
        A[c::3, c] = wm.gain[c] * (-(Jc - wm.b_inf[c])) * d * T[:, c]
        A[c::3, 3 + c] = Jc * T[:, c] + wm.b_inf[c] * (1 - T[:, c])
        A[c::3, 6 + c] = wm.gain[c] * (1 - T[:, c])
        base = (-wm.gain[c] * (Jc - wm.b_inf[c]) * wm.beta[c] * T[:, c]).reshape(-1, 1)
        A[c::3, 9:] = base * phi
    return A / sig.reshape(-1, 1)


def covariance(A: np.ndarray, ridge: float = 1e-10) -> dict:
    G = A.T @ A
    tr = float(np.trace(G)) / max(G.shape[0], 1)
    G = G + max(ridge * tr, 1e-14) * np.eye(G.shape[0])
    try:
        invG = np.linalg.inv(G)
    except np.linalg.LinAlgError:
        invG = np.linalg.pinv(G)
    dof = max(A.shape[0] - G.shape[0], 1)
    return {"invG": invG, "G": G, "dof": dof, "radius": _chi2_ppf(CONF, dof) / dof}


def analyse(I: np.ndarray, J: np.ndarray, fit: dict, cov: Optional[dict] = None) -> dict:
    """Per-pixel raw and effective (nuisance-profiled) information."""
    I3 = np.asarray(I, dtype=np.float64).reshape(-1, 3)
    J3 = np.asarray(J, dtype=np.float64).reshape(-1, 3)
    wm: WaterModel = fit["water"]
    T = np.exp(-fit["tau"])
    sig = fit["sigma"]
    q = ((wm.gain.reshape(1, 3) * T) / sig) ** 2

    A = jacobian(I3, J3, fit)
    if cov is None:
        cov = covariance(A)
    AW = A @ cov["invG"]
    h = np.clip(((AW * A).sum(axis=1)).reshape(-1, 3), 0.0, 1.0)
    q_eff = q * (1.0 - h)

    return {
        "q": q,
        "leverage": h,
        "q_eff": q_eff,
        "log10q": np.log10(np.maximum(q, 1e-30)),
        "log10q_eff": np.log10(np.maximum(q_eff, 1e-30)),
        "T": T,
        "tau": fit["tau"],
        "sigma": sig,
        "cov": cov,
        "A": A,
    }


def direction_info(an: dict, v: np.ndarray) -> dict:
    """``q_v`` and ``q_v^eff`` for a unit-energy direction ``(N,3)``.

    ``D_J G v = g_c T_c v_c``, so the whitened row is ``(g T v)/sigma``.
    """
    n = an["T"].shape[0]
    v = np.asarray(v, dtype=np.float64).reshape(n, 3)
    r = ((an["T"] * v) / an["sigma"]).ravel()
    q = float((r ** 2).sum())
    A = an["A"]
    absorbed = float(r @ A @ an["cov"]["invG"] @ (A.T @ r))
    return {"q": q, "q_eff": max(q - absorbed, 0.0), "leverage": min(absorbed / max(q, 1e-30), 1.0)}


def support_score(log10q_eff: np.ndarray) -> np.ndarray:
    """Bounded support in ``(0,1)``; 0.5 at ``q^eff = Q_REF``."""
    return 1.0 / (1.0 + np.exp(-(np.asarray(log10q_eff) - np.log10(Q_REF))))


# ==========================================================================
# Object 2 -- ambiguity measured by profiling, not by the delta method
# ==========================================================================
def chi2_full(
    I: np.ndarray,
    J: np.ndarray,
    wm: WaterModel,
    phi: np.ndarray,
    sigma: np.ndarray,
    coeff0: Optional[np.ndarray] = None,
    iters: int = 4,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Chi-square on **every** pixel, with the depth field re-optimised.

    Why the closed-form objective cannot be the profiler's target
    -----------------------------------------------------------
    ``_concentrated`` scores a water model by how well its implied optical depths
    ``tau = -log((I/g - b)/(J - b))`` fit a smooth basis.  That inversion exists
    only where ``J_c != b_c``: at ``J_c = b_c`` every transmission produces the
    same observation, so the channel genuinely carries no depth information
    there.  Excluding those pixels is correct.

    Using the residual *on the survivors* as the objective is not, for two
    compounding reasons:

    1. **Pixels leave the objective silently.**  When much of the scene sits near
       the fitted veiling radiance the score is computed on a small residue, and
       nothing in the report shows it.
    2. **The basis can interpolate that residue exactly.**  With ``k`` depth
       coefficients and fewer than ``k`` surviving pixels the fit is
       underdetermined and the residual goes to zero whatever the water model is.

    Measured on the first scene tried: chi-square collapsed to ``0.0000`` while
    the same fit's full-image RMSE was ``0.078``, and the resulting ambiguity
    width came out 14x *smaller* at 4x the noise -- a bound that tightened as
    information decreased.  That is an instrument failure, not a finding, and it
    is why this function exists.

    Here the closed-form depths seed the coefficients and are then refined
    against ``render(J, tau, wm)`` over all pixels.  Every pixel contributes, no
    pixel can be dropped for free, and values are comparable across candidate
    scenes -- which is the only condition under which a likelihood-ratio
    threshold means anything.

    Returns ``(chi2, coeff, d)``.
    """
    I3 = np.asarray(I, dtype=np.float64).reshape(-1, 3)
    J3 = np.asarray(J, dtype=np.float64).reshape(-1, 3)
    N = I3.shape[0]
    beta = wm.beta.reshape(1, 3)

    if coeff0 is None:
        tau_c, valid = _implied_tau(I3, J3, wm)
        sd = _sigma_tau(I3, wm, sigma)
        wt = valid.astype(np.float64) / np.clip(sd / beta, 1e-6, None) ** 2
        sw = wt.sum(axis=1)
        tw = (tau_c / beta * wt).sum(axis=1)
        rhs = np.where(sw > 0, tw / np.maximum(sw, 1e-12), 0.0)
        sq = np.sqrt(np.maximum(sw, 0.0))
        coeff = _lsq_normal(phi * sq.reshape(-1, 1), rhs * sq)
    else:
        coeff = np.asarray(coeff0, dtype=np.float64).copy()

    for _ in range(iters):
        d = np.clip(phi @ coeff, 0.0, TAU_MAX)
        T = np.exp(-d.reshape(-1, 1) * beta)
        resid = (wm.gain.reshape(1, 3) * (J3 * T + wm.b_inf.reshape(1, 3) * (1.0 - T))
                 - I3)
        # d render / d coeff = -g (J - b) T beta phi
        jac = (-wm.gain.reshape(1, 3) * (J3 - wm.b_inf.reshape(1, 3)) * T
               * beta).reshape(N, 3, 1) * phi.reshape(N, 1, -1)
        A = (jac / sigma.reshape(-1, 3)[:, :, None]).reshape(3 * N, -1)
        step = _lsq_normal(A, -(resid / sigma).reshape(-1))
        if not np.all(np.isfinite(step)):
            break
        coeff = coeff + step
        if np.abs(step).max() < 1e-13:
            break

    if not np.all(np.isfinite(coeff)):
        coeff = np.zeros(phi.shape[1])
    d = np.clip(phi @ coeff, 0.0, TAU_MAX)
    T = np.exp(-d.reshape(-1, 1) * beta)
    resid = (wm.gain.reshape(1, 3) * (J3 * T + wm.b_inf.reshape(1, 3) * (1.0 - T))
             - I3)
    return float(((resid / sigma) ** 2).sum()), coeff, d


def _full_residual(
    I: np.ndarray,
    J: np.ndarray,
    wm: WaterModel,
    phi: np.ndarray,
    sigma: np.ndarray,
    coeff: np.ndarray,
) -> np.ndarray:
    """Whitened residual of the all-pixel model at fixed depth coefficients."""
    I3 = np.asarray(I, dtype=np.float64).reshape(-1, 3)
    J3 = np.asarray(J, dtype=np.float64).reshape(-1, 3)
    d = np.clip(phi @ coeff, 0.0, TAU_MAX)
    T = np.exp(-d.reshape(-1, 1) * wm.beta.reshape(1, 3))
    pred = wm.gain.reshape(1, 3) * (J3 * T + wm.b_inf.reshape(1, 3) * (1.0 - T))
    return ((pred - I3) / sigma).reshape(-1)


def _refit_params(
    I: np.ndarray,
    J: np.ndarray,
    wm0: WaterModel,
    phi: np.ndarray,
    sigma: np.ndarray,
    iters: int = 6,
) -> Tuple[float, WaterModel]:
    """Re-optimise the nine water/sensor parameters for a perturbed ``J``.

    Warm-started from ``wm0`` and deliberately short: this runs inside a profiler,
    where what matters is that the nuisance parameters are *free*, not that they
    are converged to machine precision.  ``sigma`` is held fixed so chi-square
    values from different amplitudes remain comparable.
    """
    theta = wm0.as_vector()
    cost, coeff0, _ = chi2_full(I, J, wm0, phi, sigma)
    n_p = 9
    lam = 1e-3
    best = (cost, theta.copy(), coeff0)
    for _ in range(iters):
        r = _full_residual(I, J, WaterModel.from_vector(theta), phi, sigma, coeff0)
        A = np.zeros((r.size, n_p))
        eps = 1e-5
        for j in range(n_p):
            th2 = theta.copy()
            th2[j] += eps
            r2 = _full_residual(I, J, WaterModel.from_vector(th2), phi, sigma, coeff0)
            A[:, j] = (r2 - r) / eps
        JtJ = A.T @ A
        diag = np.diag(JtJ).copy()
        diag[diag <= 1e-9] = 1e-9
        try:
            delta = np.linalg.solve(JtJ + lam * np.diag(diag), -(A.T @ r))
        except np.linalg.LinAlgError:
            break
        cand = theta + delta
        cand[:3] = np.clip(cand[:3], 0.05, 6.0)
        cand[3:6] = np.clip(cand[3:6], 0.2, 5.0)
        cand[6:9] = np.clip(cand[6:9], 0.0, 0.85)
        if not np.all(np.isfinite(cand)):
            break
        c_new, coeff_new, _ = chi2_full(I, J, WaterModel.from_vector(cand),
                                        phi, sigma, coeff0=coeff0)
        if np.isfinite(c_new) and c_new < cost:
            theta, cost, coeff0 = cand, c_new, coeff_new
            best = (cost, theta.copy(), coeff0)
            lam = max(lam * 0.4, 1e-8)
        else:
            lam *= 6.0
            if lam > 1e6:
                break
    return best[0], WaterModel.from_vector(best[1])


def profile_chi2(
    I: np.ndarray,
    J: np.ndarray,
    fit: dict,
    v: np.ndarray,
    a: float,
    h: int,
    w: int,
    valid_mask: Optional[np.ndarray] = None,
    refit_iters: int = 6,
) -> float:
    """Chi-square after moving ``J`` by ``a*v`` with every nuisance re-optimised.

    Evaluated at **full resolution**, deliberately.  An earlier version profiled
    on a 48x48 work grid for speed and the single-pixel directions were destroyed
    by the resampling -- a one-pixel offset averaged away to nothing, so the
    profiler reported "unbounded" for the direction that is in fact the most
    tightly constrained.  A direction is only meaningful at the resolution it is
    defined on.

    All nine water/sensor parameters are free, not just the depth field.  That
    matters: a global colour offset is absorbed by moving ``b_inf`` in the
    opposite direction, and an earlier version that held ``b_inf`` fixed
    reported such offsets as *bounded*, contradicting the leverage computed in
    :func:`analyse`.  The profile and the linearised leverage must agree, and
    the only way to make them agree honestly is to let the nuisances move.
    ``sigma`` stays at the base value so chi-square values are comparable.
    """
    J3 = np.asarray(J, dtype=np.float64).reshape(-1, 3)
    v = np.asarray(v, dtype=np.float64).reshape(-1, 3)
    Ja = np.clip(J3 + a * v, 0.0, 1.5)
    I3 = np.asarray(I, dtype=np.float64).reshape(-1, 3)
    wm: WaterModel = fit["water"]
    phi = fit["phi"]
    if valid_mask is None:
        valid_mask = _implied_tau(I3, J3, wm)[1]
    cost, _ = _refit_params(I3, Ja, wm, phi, fit["sigma"], iters=refit_iters)
    # ``_refit_params`` now optimises ``chi2_full``, the same all-pixel objective
    # that produced ``fit["chi2"]``, so no rescaling is needed or wanted.  The
    # earlier ``conc_scale`` existed only because the profiler used the
    # concentrated residual while the fit reported the full one; matching the
    # objectives removed the discrepancy instead of papering over it.
    return cost


def profile_width(
    I: np.ndarray,
    J: np.ndarray,
    fit: dict,
    v: np.ndarray,
    h: int,
    w: int,
    a_max: float = 0.5,
    n_grid: int = 8,
    df: int = 1,
) -> dict:
    """95% ambiguity radius of the component of ``J`` along ``v``.

    ``a_max`` is a physical ceiling; hitting it means the data do not bound the
    component at all (``censored=True``), which is the honest answer for an
    unidentifiable direction -- not a large finite number.
    """
    thr = chi2_threshold(df)
    J3 = np.asarray(J, dtype=np.float64).reshape(-1, 3)
    I3 = np.asarray(I, dtype=np.float64).reshape(-1, 3)
    mask = _implied_tau(I3, J3, fit["water"])[1]

    # The baseline has to be the *same* objective evaluated at a = 0.  Using
    # ``fit["chi2"]`` looked equivalent but is not: ``fit_water`` reports the
    # chi-square of the full-resolution polish, whose depth field is obtained by a
    # weighted least-squares step, while ``_concentrated`` solves the same profile
    # in closed form.  The two agree to well under one chi-square unit but not
    # exactly, and differencing across two different objectives puts a constant
    # offset into every profile -- which at thresholds of ~3.84 is the whole
    # answer.
    base = profile_chi2(I, J, fit, v, 0.0, h, w, mask)
    budget = base + thr

    def fails(a: float) -> bool:
        return profile_chi2(I, J, fit, v, a, h, w, mask) > budget

    # locate a bracketing pair [lo, hi] with lo inside the interval and hi outside
    lo, hi = 0.0, a_max
    if not fails(hi):
        return {"width": float(a_max), "censored": True, "base_chi2": float(base)}
    grid = np.linspace(0.0, a_max, n_grid + 1)[1:]
    for a in grid:
        if fails(float(a)):
            hi = float(a)
            break
        lo = float(a)
    # geometric bisection: the interesting widths span 1e-3 .. 1e0, so halving
    # linearly wastes the first few steps when the answer is small
    for _ in range(14):
        mid = 0.5 * (lo + hi)
        if mid <= lo or mid >= hi or (hi - lo) < 1e-4:
            break
        if fails(mid):
            hi = mid
        else:
            lo = mid
    return {"width": lo, "censored": False, "base_chi2": float(base)}


def profile_table(
    I: np.ndarray,
    J: np.ndarray,
    fit: dict,
    h: int,
    w: int,
    k_low: int = 9,
    k_px: int = 8,
    a_max: float = 0.5,
) -> List[dict]:
    """Profile the ambiguity of one representative of each direction family."""
    rows: List[dict] = []
    comps = components(h, w, k_low)
    picks = [("dc", f"dc_{c}", comps[f"dc_{c}"]) for c in range(3)]
    picks += [("low", f"low_0_{j}", comps[f"low_0_{j}"]) for j in (0, 4, 8)]
    px = pixel_dirs(h * w, k_px)
    picks += [("pixel", f"px_{i}", px[i]) for i in range(k_px)]
    for kind, name, v in picks:
        r = profile_width(I, J, fit, v, h, w, a_max=a_max)
        amp = float(np.abs(v).max())
        rows.append(
            {
                "kind": kind,
                "name": name,
                "width": r["width"],
                "censored": r["censored"],
                "amp_scale": amp,
                "per_pixel_offset": float(r["width"] * amp),
                "base_chi2": r["base_chi2"],
            }
        )
    return rows


def profile_summary(rows: List[dict]) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for kind in ("dc", "low", "pixel"):
        sub = [r for r in rows if r["kind"] == kind]
        if not sub:
            continue
        w = np.array([r["per_pixel_offset"] for r in sub])
        out[kind] = {
            "n": len(sub),
            "median_width": float(np.median(w)),
            "min_width": float(w.min()),
            "max_width": float(w.max()),
            "n_censored": int(sum(r["censored"] for r in sub)),
        }
    return out


# ==========================================================================
# components
# ==========================================================================
def components(h: int, w: int, k_low: int = 9) -> Dict[str, np.ndarray]:
    """Orthonormal unit-energy ``(N,3)`` directions: DC and low-frequency."""
    n = h * w
    out: Dict[str, np.ndarray] = {}
    for c in range(3):
        v = np.zeros((n, 3))
        v[:, c] = 1.0 / np.sqrt(n)
        out[f"dc_{c}"] = v
    phi = depth_basis(h, w, k_low)
    phi = phi - phi.mean(axis=0, keepdims=True)
    q_mat, _ = np.linalg.qr(phi)
    for c in range(3):
        for j in range(k_low):
            v = np.zeros((n, 3))
            v[:, c] = q_mat[:, j]
            out[f"low_{c}_{j}"] = v
    return out


def blob_dirs(h: int, w: int, radii: Sequence[float], channel: Optional[int] = None) -> List[Tuple[str, np.ndarray]]:
    """Peak-normalised Gaussian-blob directions of different spatial extent.

    Footprint-matched comparison.  An "equal-energy" DC direction spreads its
    amplitude over every pixel, so its per-pixel number cannot be compared with a
    one-pixel direction: the two differ by ``sqrt(N)`` in normalisation alone.
    Here every direction has **peak 1**, so the profiled amplitude ``a`` is
    directly the peak per-pixel offset -- the quantity a restoration decision
    acts on when it decides how much colour to add back.
    """
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    out: List[Tuple[str, np.ndarray]] = []
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    for r in radii:
        if r <= 0:
            g = np.zeros((h, w))
            g[int(round(cy)), int(round(cx))] = 1.0
        else:
            g = np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * r * r)))
        v = np.zeros((h * w, 3))
        chans = list(range(3)) if channel is None else [channel]
        for c in chans:
            v[:, c] = g.ravel() / len(chans)
        out.append((f"r{r:g}", v))
    return out


def pixel_dirs(n: int, k: int) -> np.ndarray:
    out = np.zeros((k, n, 3))
    idx = np.linspace(0, n - 1, k).astype(int)
    for i, p in enumerate(idx):
        out[i, p, 0] = 1.0
    return out


def direction_table(an: dict, h: int, w: int, k_low: int = 9, k_px: int = 24) -> List[dict]:
    rows: List[dict] = []
    for name, v in components(h, w, k_low).items():
        d = direction_info(an, v)
        rows.append({"kind": "dc" if name.startswith("dc") else "low", "name": name, **d})
    for i, v in enumerate(pixel_dirs(h * w, k_px)):
        rows.append({"kind": "pixel", "name": f"px_{i}", **direction_info(an, v)})
    return rows


def summarise(rows: List[dict], n: Optional[int] = None) -> Dict[str, dict]:
    """Per-family summary in **amplitude** units (not unit-energy units).

    ``sd_amplitude = 1/sqrt(q_eff)`` for a unit-energy direction is the standard
    deviation of the *coefficient* of that direction; dividing by the direction's
    per-pixel amplitude norm converts it to "how well is one pixel's value known".
    """
    out: Dict[str, dict] = {}
    for kind in ("dc", "low", "pixel"):
        sub = [r for r in rows if r["kind"] == kind]
        if not sub:
            continue
        out[kind] = {
            "n": len(sub),
            "mean_leverage": float(np.mean([r["leverage"] for r in sub])),
            "mean_log10q": float(np.mean([np.log10(max(r["q"], 1e-30)) for r in sub])),
            "median_q_eff": float(np.median([r["q_eff"] for r in sub])),
            "median_sd_unit": float(np.median([1.0 / np.sqrt(max(r["q_eff"], 1e-30)) for r in sub])),
        }
    return out

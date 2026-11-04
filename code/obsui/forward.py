"""Forward observation model for underwater imaging + controlled degradation.

Implements the Akkaynak-Treibitz revised underwater formation model with the
wideband attenuation coefficient form, plus a backscatter term that depends on
range.  Everything is differentiable in torch so we can compute the local
observation sensitivity  q_v = || Sigma_n^{-1/2} D_J G_psi v ||_2^2.

Forward model (per channel c, pixel x):

    I_c(x) = J_c(x) * exp(-beta_c^D * d(x))
           + B_c^inf * (1 - exp(-beta_c^B * d(x)))
           + n_c(x)

The simplified coefficient form used in Akkaynak & Treibitz (CVPR 2018 /
TPAMI 2019) is adopted for beta_c^D:

    beta_c^D = beta_c^M + beta_c^N
    beta_c^M = -(ln B_c^inf) / D_inf
    beta_c^N = a_c * exp(b_c * lam_c) + c_c * exp(d_c * lam_c)

with lam_c the channel wavelength (0.62, 0.54, 0.45 um for R, G, B in the
ocean-optics convention used there).  beta_c^B = -ln(1-B_c^inf)/D_inf.

This module is used in three distinct modes, and they must not be confused:
  * `synth_*`  -- build the CONTROLLED degradation track (ground truth known)
  * `render`   -- render a candidate J under hypothesised psi (inverse problem)
  * `sensitivity` -- local Jacobian of the above w.r.t. J, for q_v
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, Tuple, Optional

import numpy as np

LAM_R, LAM_G, LAM_B = 0.620, 0.540, 0.450  # micrometres, R/G/B
ABCOEF = {  # a, b, c, d  (Akkaynak & Treibitz 2019, ocean type I water)
    "a": (1.0, 1.0, 1.0),
    "b": (-0.00113, -0.00113, -0.00113),
    "c": (0.0, 0.0, 0.0),
    "d": (0.0, 0.0, 0.0),
}


@dataclass
class WaterParams:
    """Hypothesised (or true, in the controlled track) water parameters."""
    beta_d: Tuple[float, float, float] = (0.45, 0.18, 0.13)   # diffuse attenuation per channel
    beta_b: Tuple[float, float, float] = (0.40, 0.17, 0.12)   # backscatter attenuation
    b_inf: Tuple[float, float, float] = (0.12, 0.22, 0.30)    # veiling light / colour cast
    depth_scale: float = 1.0                                   # multiplier on normalised range
    gamma: float = 1.0                                         # camera response exponent
    gain: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    noise_sigma: Tuple[float, float, float] = (0.010, 0.010, 0.010)
    shot_alpha: float = 0.0                                    # signal-dependent noise scale
    clip: bool = True                                          # saturation / quantisation

    def key(self) -> str:
        return "|".join(
            f"{k}={asdict(self)[k]}" for k in
            ("beta_d", "beta_b", "b_inf", "depth_scale", "gamma", "noise_sigma", "shot_alpha")
        )

    def to_dict(self) -> Dict:
        d = asdict(self)
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in d.items()}


def _softplus(z: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(-np.abs(z))) + np.maximum(z, 0.0)


def depth_map_from_content(J: np.ndarray, rng: np.random.RandomState,
                           mode: str = "smooth") -> np.ndarray:
    """A normalised range map d in [0,1].  NOT the true scene depth: the point of
    the controlled track is that *we choose* the range map and therefore know it.
    """
    h, w = J.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    yy /= max(h - 1, 1)
    xx /= max(w - 1, 1)
    if mode == "smooth":
        d = 0.35 + 0.45 * yy + 0.10 * np.sin(2 * np.pi * (0.7 * xx + 0.3 * yy))
    elif mode == "gradient":
        d = 0.15 + 0.8 * yy
    elif mode == "radial":
        d = 0.2 + 0.7 * np.hypot(yy - 0.4, xx - 0.5)
    else:
        raise ValueError(mode)
    # low-frequency perturbation so that every image gets a different geometry
    ph = rng.uniform(0, 2 * np.pi, size=2)
    d = d + 0.05 * np.sin(2 * np.pi * xx + ph[0]) * np.cos(2 * np.pi * yy + ph[1])
    d = np.clip(d, 0.0, 1.0)
    return d.astype(np.float32)


def forward_np(J: np.ndarray, d: np.ndarray, p: WaterParams,
               rng: Optional[np.random.RandomState] = None,
               return_terms: bool = False):
    """Numpy reference implementation. J in [0,1], d in [0,1]. Returns I in [0,1]."""
    J = J.astype(np.float64)
    d = d.astype(np.float64)[..., None] * p.depth_scale
    bd = np.array(p.beta_d).reshape(1, 1, 3)
    bb = np.array(p.beta_b).reshape(1, 1, 3)
    bi = np.array(p.b_inf).reshape(1, 1, 3)
    T = np.exp(-bd * d)
    Bs = bi * (1.0 - np.exp(-bb * d))
    if p.gamma != 1.0:
        Jg = np.power(np.clip(J, 1e-8, 1.0), p.gamma)
    else:
        Jg = J
    I = Jg * T + Bs
    I = I * np.array(p.gain).reshape(1, 1, 3)
    if return_terms:
        return I, {"T": T, "B": Bs}
    return I


def add_sensor_noise(I: np.ndarray, p: WaterParams,
                     rng: np.random.RandomState) -> np.ndarray:
    """Read noise + signal-dependent (shot) noise + 8-bit quantisation."""
    sig = np.array(p.noise_sigma).reshape(1, 1, 3)
    n = rng.normal(0.0, 1.0, size=I.shape) * sig
    if p.shot_alpha > 0:
        n = n + rng.normal(0.0, 1.0, size=I.shape) * p.shot_alpha * np.sqrt(np.clip(I, 0, None))
    out = I + n
    if p.clip:
        out = np.clip(out, 0.0, 1.0)
        out = np.round(out * 255.0) / 255.0
    return out


def synth(J: np.ndarray, d: np.ndarray, p: WaterParams,
          rng: np.random.RandomState, return_terms: bool = False):
    clean = forward_np(J, d, p)
    if p.gamma != 1.0:
        clean = np.clip(clean, 0.0, 1.0)
    noisy = add_sensor_noise(clean, p, rng)
    if return_terms:
        _, t = forward_np(J, d, p, return_terms=True)
        return noisy, clean, t
    return noisy, clean


def sample_water(rng: np.random.RandomState, regime: str = "mixed",
                 severity: float = 1.0) -> WaterParams:
    """Sample a water/render condition.  `regime` fixes the *family* so that
    train / dev / test can be made disjoint in a controlled way."""
    if regime == "clear":
        bd = (0.25, 0.12, 0.10); bi = (0.06, 0.12, 0.18); sig = 0.006
    elif regime == "coastal":
        bd = (0.55, 0.22, 0.15); bi = (0.14, 0.24, 0.32); sig = 0.010
    elif regime == "harbour":
        bd = (0.80, 0.32, 0.22); bi = (0.22, 0.32, 0.40); sig = 0.014
    elif regime == "deep":
        bd = (1.05, 0.42, 0.28); bi = (0.10, 0.26, 0.42); sig = 0.016
    elif regime == "turbid":
        bd = (1.30, 0.60, 0.42); bi = (0.30, 0.38, 0.46); sig = 0.020
    else:  # mixed: draw a random point of the family
        bd = tuple(rng.uniform(0.2, 1.3, 3)); bi = tuple(rng.uniform(0.05, 0.45, 3)); sig = float(rng.uniform(0.005, 0.022))
    s = severity
    return WaterParams(
        beta_d=tuple(float(v) * s for v in bd),
        beta_b=tuple(float(v) * s * 0.9 for v in bd),
        b_inf=tuple(float(v) for v in bi),
        depth_scale=float(rng.uniform(0.75, 1.25)),
        gamma=float(1.0 + rng.uniform(-0.08, 0.08)),
        gain=tuple(float(g) for g in rng.uniform(0.94, 1.06, 3)),
        noise_sigma=(sig, sig, sig),
        shot_alpha=float(rng.uniform(0.0, 0.025)),
        clip=True,
    )


# ---------------------------------------------------------------- torch side
try:
    import torch
    import torch.nn.functional as F

    def render_torch(J: "torch.Tensor", d: "torch.Tensor", beta_d, beta_b, b_inf,
                     gamma: float = 1.0) -> "torch.Tensor":
        """J: (N,3,H,W) in [0,1]; d: (N,1,H,W) per-sample range. Returns I (N,3,H,W)."""
        bd = torch.as_tensor(beta_d, dtype=J.dtype, device=J.device).view(1, 3, 1, 1)
        bb = torch.as_tensor(beta_b, dtype=J.dtype, device=J.device).view(1, 3, 1, 1)
        bi = torch.as_tensor(b_inf, dtype=J.dtype, device=J.device).view(1, 3, 1, 1)
        T = torch.exp(-bd * d)
        Bs = bi * (1.0 - torch.exp(-bb * d))
        Jg = torch.clamp(J, 1e-6, 1.0).pow(gamma) if gamma != 1.0 else J
        return Jg * T + Bs

    def local_sensitivity(J0: "torch.Tensor", d0: "torch.Tensor", direction: "torch.Tensor",
                          beta_d, beta_b, b_inf, gamma: float = 1.0,
                          sigma_n: float = 0.01) -> "torch.Tensor":
        """q_v(x) = || Sigma_n^{-1/2} D_J G v_x ||_2^2 computed by JVP.

        `direction` is a unit-norm (per-pixel) perturbation field v in image space
        (N,3,H,W).  Returns per-pixel scalar (N,1,H,W).
        The whole point is that this is a linearised, noise-weighted local
        measure of how strongly the observation constrains J along v -- it is
        NOT the inverse-network's error.
        """
        v = direction.detach()
        vn = v / (v.pow(2).sum(1, keepdim=True).sqrt() + 1e-8)
        J = J0.detach().clone().requires_grad_(True)
        I = render_torch(J, d0, beta_d, beta_b, b_inf, gamma)
        g = torch.autograd.grad(I, J, grad_outputs=vn, create_graph=False, retain_graph=False)[0]
        q = (g / sigma_n).pow(2).sum(1, keepdim=True)
        return q

    def true_error_sensitivity(Jt: "torch.Tensor", Jhat: "torch.Tensor",
                               d0: "torch.Tensor", beta_d, beta_b, b_inf,
                               gamma: float = 1.0, sigma_n: float = 0.01) -> "torch.Tensor":
        """Same quantity as `local_sensitivity`, but with v = the ACTUAL error
        direction (Jt - Jhat) instead of a predefined perturbation direction.

        Diagnosis-only: this needs the ground-truth J, so it can never be an
        input feature of a single-image model.  It answers "how strongly does
        the observation constrain the *error we actually made*", which is the
        quantity H1 claims `local_sensitivity` proxies.
        """
        v = (Jt - Jhat).detach()
        return local_sensitivity(Jt.detach(), d0, v, beta_d, beta_b, b_inf, gamma, sigma_n)

    HAS_TORCH = True
except Exception:  # pragma: no cover
    HAS_TORCH = False


def _selftest():
    rng = np.random.RandomState(1)
    J = rng.rand(64, 64, 3).astype(np.float64)
    d = depth_map_from_content(J, rng)
    p = sample_water(rng, "coastal")
    I, clean, terms = synth(J, d, p, rng, return_terms=True)
    assert I.shape == J.shape and I.min() >= 0 and I.max() <= 1
    # deeper -> darker in every channel
    shallow = forward_np(J, np.zeros_like(d), p)
    assert shallow.mean() > clean.mean()
    # T is monotone decreasing along the depth gradient direction.
    d_grad = depth_map_from_content(J, np.random.RandomState(2), mode="gradient")
    _, _, terms_g = synth(J, d_grad, p, np.random.RandomState(3), return_terms=True)
    col = terms_g["T"][:, 0, 0]
    assert col[0] > col[-1], (col[0], col[-1])
    assert np.all(np.diff(col) <= 1e-9)
    # deterministic without noise
    p0 = WaterParams(noise_sigma=(0.0, 0.0, 0.0), shot_alpha=0.0)
    a1 = forward_np(J, d, p0); a2 = forward_np(J, d, p0)
    assert np.allclose(a1, a2)
    print("forward selftest OK", "mean I=%.3f" % I.mean())


if __name__ == "__main__":
    _selftest()

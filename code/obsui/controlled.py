"""Controlled forward-degradation track.

Why this exists
---------------
Real datasets cannot answer the question this project asks.  UIEB/LSUI/UFO give
``(I, J*)`` pairs produced by human selection and unknown processing, so a
"recoverability" claim read off them is confounded with whatever the reference
producer did.  Here the forward model is *ours*: we choose the water, the range
map, the noise and the camera response, so the observation is generated from a
known ``J*`` by a known ``G_psi``, and the question "does the observation
constrain this component of ``J``" has a true answer we can compare against.

Design decisions that matter
----------------------------
**One factor at a time.**  Every condition starts from a common base and changes
exactly one physical axis.  Without that, an "attenuation effect" is really an
attenuation-and-noise-and-depth effect, and the controlled track would inherit
the same confound it was built to escape.

**Attenuation is changed by scaling the range, not by scaling beta.**  The
observation depends on ``tau = beta * d`` only; ``beta`` and ``d`` are separately
unidentifiable (see ``support.py``).  Rescaling ``beta`` at fixed ``d`` changes
nothing observable, so it cannot be used as an attenuation axis.  Scaling ``d``
does.

**Everything is seeded and hashed.**  Each sample's parameters, the RNG seed and
the SHA-256 of the output are recorded, so the grid can be regenerated bit for
bit and the report can state a manifest hash rather than a claim.

**Indistinguishability is measured, not asserted.**  Two settings that "look
indistinguishable" in a figure prove nothing; figures are chosen after the fact.
:func:`detectability` draws independent noise realisations under both hypotheses
and reports the separation of the log-likelihood ratio.  If that separation is
below the noise floor, the pair is genuinely ambiguous, and a single-image method
claiming to tell them apart is inventing information.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import forward as F

# ---------------------------------------------------------------------------
# the controlled axis definition
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Condition:
    """One point of the controlled grid.  All fields are physical and known."""
    name: str = "base"
    attenuation: float = 1.0     # multiplies the range map (not beta: see module doc)
    backscatter: float = 1.0     # multiplies the veiling radiance
    read_sigma: float = 0.004
    shot_alpha: float = 0.008
    gamma: float = 1.0           # camera response exponent
    exposure: float = 1.0        # gain before quantisation; >1 pushes into clipping
    illum_strength: float = 0.0  # amplitude of the non-uniform illumination field
    depth_var: float = 1.0       # spatial variation of the range map
    seed: int = 0

    def key(self) -> str:
        return (f"{self.name}|a{self.attenuation:g}|b{self.backscatter:g}"
                f"|r{self.read_sigma:g}|s{self.shot_alpha:g}|g{self.gamma:g}"
                f"|e{self.exposure:g}|i{self.illum_strength:g}|dv{self.depth_var:g}"
                f"|seed{self.seed}")

    def to_dict(self) -> Dict:
        return asdict(self)


BASE = Condition(name="base")

# One-factor-at-a-time axes.  The values are chosen to span the range that the
# real datasets occupy, not to make any particular result look good.
AXES: Dict[str, List[Tuple[str, Condition]]] = {
    # band-dependent attenuation / red loss.  Scaling d is the physical knob.
    "attenuation": [
        ("attn_x0.4", replace(BASE, name="attn_x0.4", attenuation=0.4)),
        ("attn_x0.7", replace(BASE, name="attn_x0.7", attenuation=0.7)),
        ("attn_x1.0", replace(BASE, name="attn_x1.0", attenuation=1.0)),
        ("attn_x1.4", replace(BASE, name="attn_x1.4", attenuation=1.4)),
        ("attn_x1.9", replace(BASE, name="attn_x1.9", attenuation=1.9)),
    ],
    "backscatter": [
        ("bs_x0.2", replace(BASE, name="bs_x0.2", backscatter=0.2)),
        ("bs_x0.6", replace(BASE, name="bs_x0.6", backscatter=0.6)),
        ("bs_x1.0", replace(BASE, name="bs_x1.0", backscatter=1.0)),
        ("bs_x1.5", replace(BASE, name="bs_x1.5", backscatter=1.5)),
        ("bs_x2.2", replace(BASE, name="bs_x2.2", backscatter=2.2)),
    ],
    "noise": [
        ("nz_x0.25", replace(BASE, name="nz_x0.25", read_sigma=0.001, shot_alpha=0.002)),
        ("nz_x0.5", replace(BASE, name="nz_x0.5", read_sigma=0.002, shot_alpha=0.004)),
        ("nz_x1.0", replace(BASE, name="nz_x1.0")),
        ("nz_x2.0", replace(BASE, name="nz_x2.0", read_sigma=0.008, shot_alpha=0.016)),
        ("nz_x4.0", replace(BASE, name="nz_x4.0", read_sigma=0.016, shot_alpha=0.032)),
    ],
    # camera response deviation: a gamma != 1 is what a mis-specified forward
    # model looks like when the calibrating method assumes gamma == 1
    "gamma": [
        ("gm_0.70", replace(BASE, name="gm_0.70", gamma=0.70)),
        ("gm_0.85", replace(BASE, name="gm_0.85", gamma=0.85)),
        ("gm_1.00", replace(BASE, name="gm_1.00", gamma=1.00)),
        ("gm_1.20", replace(BASE, name="gm_1.20", gamma=1.20)),
        ("gm_1.45", replace(BASE, name="gm_1.45", gamma=1.45)),
    ],
    # saturation / clipping: exposure before quantisation
    "saturation": [
        ("ex_0.60", replace(BASE, name="ex_0.60", exposure=0.60)),
        ("ex_1.00", replace(BASE, name="ex_1.00", exposure=1.00)),
        ("ex_1.80", replace(BASE, name="ex_1.80", exposure=1.80)),
        ("ex_2.80", replace(BASE, name="ex_2.80", exposure=2.80)),
        ("ex_4.20", replace(BASE, name="ex_4.20", exposure=4.20)),
    ],
    # non-uniform illumination (a spotlight / vignette the forward model does not
    # contain -- this is the simulator-mismatch axis)
    "illumination": [
        ("il_0.00", replace(BASE, name="il_0.00", illum_strength=0.0)),
        ("il_0.25", replace(BASE, name="il_0.25", illum_strength=0.25)),
        ("il_0.50", replace(BASE, name="il_0.50", illum_strength=0.50)),
        ("il_0.75", replace(BASE, name="il_0.75", illum_strength=0.75)),
    ],
    # range-map spatial variation: for the identifiability question, not a
    # degradation axis in the quality sense
    "depth_var": [
        ("dv_0.00", replace(BASE, name="dv_0.00", depth_var=0.0)),
        ("dv_0.35", replace(BASE, name="dv_0.35", depth_var=0.35)),
        ("dv_0.70", replace(BASE, name="dv_0.70", depth_var=0.70)),
        ("dv_1.00", replace(BASE, name="dv_1.00", depth_var=1.00)),
    ],
}

BASE_WATER = F.WaterParams(
    beta_d=(0.62, 0.54, 0.45), beta_b=(0.55, 0.48, 0.40),
    b_inf=(0.10, 0.18, 0.25), depth_scale=1.0, gamma=1.0,
    gain=(1.0, 1.0, 1.0), noise_sigma=(0.010, 0.010, 0.010), shot_alpha=0.0,
)


def _range_map(J: np.ndarray, rng: np.random.RandomState, depth_var: float,
               mode: str = "smooth") -> np.ndarray:
    """Normalised range map with a controlled amount of spatial variation.

    ``depth_var=0`` gives a *spatially constant* optical depth -- the case where a
    global colour offset is exactly absorbable by the veiling term.  That
    degenerate limit has to be reachable, or the identifiability claim cannot be
    tested at all.
    """
    d = F.depth_map_from_content(J, rng, mode=mode)
    return (0.5 + depth_var * (d - 0.5)).astype(np.float32)


def render(J: np.ndarray, cond: Condition, rng: np.random.RandomState,
           quantise: bool = True) -> Dict[str, np.ndarray]:
    """Render ``J`` (H, W, 3) in [0, 1] under ``cond``.

    Returns a dict with the noisy observation, the noise-free observation (so a
    caller can separate "the information the model has" from "the noise added on
    top"), the range map and the water parameters actually used.
    """
    J = np.clip(np.asarray(J, np.float64), 0.0, 1.0)
    h, w = J.shape[:2]
    d = _range_map(J, rng, cond.depth_var)

    p = F.WaterParams(
        beta_d=BASE_WATER.beta_d,
        beta_b=BASE_WATER.beta_b,
        b_inf=tuple(float(x) * cond.backscatter for x in BASE_WATER.b_inf),
        depth_scale=float(cond.attenuation),
        gamma=float(cond.gamma),
        gain=(float(cond.exposure),) * 3,
        noise_sigma=(float(cond.read_sigma),) * 3,
        shot_alpha=float(cond.shot_alpha),
        clip=bool(quantise),
    )
    I_clean = F.forward_np(J, d, p)

    # non-uniform illumination: a smooth multiplicative field, applied to the
    # *signal* before the water, i.e. the thing the forward model does not know
    # about.  This is the mismatch axis.
    if cond.illum_strength > 0:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
        yy /= max(h - 1, 1)
        xx /= max(w - 1, 1)
        ph = rng.uniform(0, 2 * np.pi, size=2)
        field = 1.0 + cond.illum_strength * (
            0.7 * np.sin(2 * np.pi * (0.6 * xx + ph[0]) + ph[1])
            + 0.3 * np.cos(2 * np.pi * (0.35 * yy - ph[1]))
        )
        field = np.clip(field, 0.05, None)[..., None]
        I_clean = I_clean * field

    I = F.add_sensor_noise(I_clean, p, rng) if quantise else I_clean + rng.normal(
        0.0, float(cond.read_sigma), size=I_clean.shape)
    I = np.clip(I, 0.0, 1.0).astype(np.float32)
    return {"I": I, "I_clean": I_clean.astype(np.float32), "d": d,
            "water": p, "cond": cond}


def grid(axes: Optional[Sequence[str]] = None) -> List[Condition]:
    """All conditions in the requested axes, in a stable order."""
    names = list(axes) if axes is not None else list(AXES)
    out: List[Condition] = []
    for ax in names:
        out.extend(c for _, c in AXES[ax])
    return out


# ---------------------------------------------------------------------------
# indistinguishability: the measurement that makes "ambiguity" falsifiable
# ---------------------------------------------------------------------------


def detectability(J_a: np.ndarray, J_b: np.ndarray, cond: Condition,
                  n_draws: int = 64, seed: int = 0) -> Dict[str, float]:
    """Can the observation distinguish ``J_a`` from ``J_b`` at this noise level?

    Both hypotheses are rendered with the *same* range map (the range map is a
    property of the scene, not of the candidate restoration) and the same water,
    so the only difference between them is the scene content.  For each of
    ``n_draws`` independent noise realisations we compute the log-likelihood
    ratio of the two hypotheses and report its separation.

    ``d_prime`` is the standardised separation of the LLR distribution under the
    two hypotheses.  Under a correct model it equals the Mahalanobis distance
    between the two observation means, scaled by noise.  ``auc`` is the
    probability that a draw from one is ranked above a draw from the other;
    ``0.5`` means the two are not distinguishable at all, and a method that
    claims to tell them apart is fabricating information.

    Reported instead of "these two images look the same", because that claim is
    unfalsifiable: any pair can be made to look similar by printing them small.
    """
    Ja = np.clip(np.asarray(J_a, np.float64), 0.0, 1.0)
    Jb = np.clip(np.asarray(J_b, np.float64), 0.0, 1.0)
    if Ja.shape != Jb.shape:
        raise ValueError(f"shape mismatch {Ja.shape} vs {Jb.shape}")
    h, w = Ja.shape[:2]

    # same range map for both hypotheses
    rng_map = np.random.RandomState(seed + 991)
    d = _range_map(Ja, rng_map, cond.depth_var)

    p = F.WaterParams(
        beta_d=BASE_WATER.beta_d, beta_b=BASE_WATER.beta_b,
        b_inf=tuple(float(x) * cond.backscatter for x in BASE_WATER.b_inf),
        depth_scale=float(cond.attenuation), gamma=float(cond.gamma),
        gain=(float(cond.exposure),) * 3,
        noise_sigma=(float(cond.read_sigma),) * 3,
        shot_alpha=float(cond.shot_alpha), clip=False,
    )
    mu_a = F.forward_np(Ja, d, p)
    mu_b = F.forward_np(Jb, d, p)

    var = float(cond.read_sigma) ** 2 + float(cond.shot_alpha) ** 2 * np.clip(
        0.5 * (mu_a + mu_b), 1e-6, None)
    sd = np.sqrt(var)
    # LLR for equal-variance Gaussians reduces to the whitened inner product
    w_a = (mu_a - mu_b) / np.clip(sd, 1e-9, None)

    rng = np.random.RandomState(seed)
    llr = {"a": np.empty(n_draws), "b": np.empty(n_draws)}
    for i in range(n_draws):
        noise = rng.normal(0.0, 1.0, size=mu_a.shape) * sd
        llr["a"][i] = float(((mu_a + noise - mu_b) * w_a).sum())
        llr["b"][i] = float(((mu_b + noise - mu_b) * w_a).sum())

    # the first term is a constant shift; what matters is the fluctuation, so
    # subtract each hypothesis' own mean before comparing shapes
    mean_gap = float(llr["a"].mean() - llr["b"].mean())
    pooled = float(np.sqrt(0.5 * (llr["a"].var() + llr["b"].var())))
    d_prime = mean_gap / max(pooled, 1e-12)

    # ROC AUC by rank
    allv = np.concatenate([llr["a"], llr["b"]])
    order = np.argsort(allv)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(allv), dtype=float)
    ra = ranks[:n_draws]
    auc = float((ra.sum() - n_draws * (n_draws - 1) / 2.0) / (n_draws * n_draws))

    return {
        "d_prime": float(d_prime), "auc": float(auc),
        "mean_llr_a": float(llr["a"].mean()), "mean_llr_b": float(llr["b"].mean()),
        "sd_llr": pooled, "n_draws": int(n_draws),
        "pixel_rms_diff": float(np.sqrt(((mu_a - mu_b) ** 2).mean())),
        "noise_rms": float(np.sqrt((sd ** 2).mean())),
    }


def colour_offset_pair(J: np.ndarray, delta: float, channel: int = 0) -> np.ndarray:
    """The reference scene with one channel globally offset by ``delta``.

    This is the pair that matters: it is exactly the "unsupported colour" a
    restoration can add.  At ``depth_var=0`` it should be undetectable; at large
    ``depth_var`` the range map makes it visible.
    """
    out = np.array(J, dtype=np.float64, copy=True)
    out[..., channel] = np.clip(out[..., channel] + delta, 0.0, 1.0)
    return out

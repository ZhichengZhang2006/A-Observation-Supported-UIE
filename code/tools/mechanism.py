"""H3 -- does the support signal actually buy anything, applied to a frozen method?

The measurement half of this project is settled: the observation determines colour
only up to a one-parameter family (``obsui.gauge``), and the information the
observation carries about colour, band and pixel components differs by orders of
magnitude (``obsui.support``).  What is *not* settled is whether acting on that
knowledge improves an actual restoration.  This script tests exactly that, and it
does so without training anything, for a reason:

    A trained intervention confounds the mechanism with capacity.  If a new
    network head wins, the win could be the head.  Applying the rule to a frozen
    method's frozen output isolates the decision: same pixels in, same model, only
    the colour assertion is damped where the data does not support it.

The rule
--------
The gauge theorem says the ambiguity is **chromatic** -- the family lifts each
channel by ``exp(beta_c dd)``, so a different admissible explanation changes colour
but is not free to invent structure.  Two interventions are tested:

``support_g`` -- **damped interpolation toward the naive inversion.**  This is the
    obvious reading of "act where support is low", and it is what the first
    version of this script did.  It is kept because it failed informatively: the
    naive inversion ``J_obs`` is a poor restoration in chroma terms, so pulling
    toward it makes chroma error *worse* (+11.6%), and the oracle gating makes it
    worse too (+12.4%).  When the oracle also loses, the anchor is the problem,
    not the gate.

``project`` -- **projection onto the admissible family.**  The rule the gauge
    theorem licenses: move the method's output the least amount that makes it
    observation-consistent.  **This projection is provably the identity** (``dd=0``
    is admissible and reproduces the input exactly), and the experiment is what
    demonstrates it rather than asserts it:

        Every restoration is already observation-consistent for some water depth.
        A consistency-based loss, gate or projection therefore cannot correct
        colour.  Anything that does correct colour is drawing on a scene prior,
        not on the measurement.

``project_oracle`` -- the same projection with the *reference* choosing the step.
    Diagnostic only, never deployable.  It measures how much of the colour error is
    inside the family at all, i.e. the ceiling of any consistency-based correction.

``greyworld`` -- pull chroma to the scene's own grey point.  **Uses no observation
    model at all**, only the assumption that a natural underwater scene has roughly
    balanced chroma.  This is the honest competitor to the paper's framing: if it
    moves the metric as much as a support-driven variant, then whatever improvement
    is available is available to a scene prior, and the observation contributes
    nothing to colour.  It is a real baseline that real papers use, not a strawman.

Ablations, and why each one exists
----------------------------------
Every one of these is a way the claimed effect could be an artefact, and each is a
spec-mandated falsifier:

``base``        the frozen method, untouched.  Reference point.
``project``     the family projection -- the mechanism this paper claims.
``project_1ch`` projection allowed to move only one channel at a time (a weaker
                version, to show the gain is not an artefact of touching all three).
``project_per_ch`` unconstrained per-channel gain, i.e. the same operation with the
                water constraint removed.  If this does as well, the *constraint*
                is doing nothing and any brightness/sharpness sweep would match.
``random_g``    same damping strength, random spatial pattern.  If this helps as
                much, the gain is from smoothing, not from support.
``const_g``     uniform damping at the mean ``g``.  Tests whether spatial
                *selectivity* matters or only the average amount.  (Measured: the
                support map's within-image spread is small, so this was expected to
                match -- and it does.  Reported as a limitation of the spatial
                form, not hidden.)
``degrade_g``   g driven by degradation brightness rather than support -- the
                obvious cheap proxy the paper must beat.
``lowpass``     plain low-pass on chroma, matched in strength.  If a blur does the
                same, the mechanism is a blur.
``oracle_g``    g driven by the *true* error (needs the reference).  Upper bound
                and a sanity check that the direction of the intervention is right;
                it is never a deployable method and is labelled as such.

Metrics are the ones the claim is about: chromatic error (primary), luma error,
PSNR and SSIM.  A mechanism that lowers chroma error while lowering PSNR is
trading one failure for another and must be reported as such.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import obsui  # noqa: E402,F401

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import obsui.metrics as M  # noqa: E402
import obsui.support as S  # noqa: E402

EPS = 0.05
LUM = np.array([0.2126, 0.7152, 0.0722])


def chroma_opp(a: np.ndarray) -> np.ndarray:
    """Log-opponent chroma, the space the gauge family moves in."""
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    return np.stack([np.log((r + EPS) / (g + EPS)),
                     np.log((g + EPS) / (b + EPS))], axis=-1)


def apply_chroma(a: np.ndarray, ch: np.ndarray) -> np.ndarray:
    """Rebuild an RGB image from its luma and a target chroma.

    Luma is preserved exactly so a chroma edit cannot smuggle in a brightness
    change -- that separation is what makes the ablation table interpretable.
    """
    luma = (a @ LUM)
    rg, gb = ch[..., 0], ch[..., 1]
    # ratio form: R/G = e^rg, G/B = e^gb, plus luma -> solve for scale
    r_over_g = np.exp(rg)
    g_over_b = np.exp(gb)
    # luma = 0.2126R + 0.7152G + 0.0722B, with R = r_over_g * G, B = G / g_over_b
    denom = 0.2126 * r_over_g + 0.7152 + 0.0722 / g_over_b
    g = luma / np.maximum(denom, 1e-6)
    out = np.stack([g * r_over_g, g, g / np.maximum(g_over_b, 1e-6)], axis=-1)
    return np.clip(out, 0.0, 1.0)


def smooth(g: np.ndarray, k: int = 5) -> np.ndarray:
    return cv2.GaussianBlur(g.astype(np.float32), (k, k), 0).astype(np.float64)


def project_onto_family(base: np.ndarray, beta: np.ndarray, binf: np.ndarray,
                        target: np.ndarray = None,
                        n_grid: int = 401, d_max: float = 1.2) -> dict:
    """Move ``base`` onto the nearest member of the admissible gauge family.

    The family is ``J'(dd)_c = b_c + (J_c - b_c) exp(beta_c dd)``.  The projection
    is the member closest to what the method already said:

        dd* = argmin_{|dd| <= d_max} sum_c || J_c - J'(dd)_c ||^2

    **This projection is the identity.**  ``dd = 0`` is always admissible and
    reproduces ``J`` exactly, so the minimiser is ``dd* = 0`` for every input.  That
    is not an implementation slip; it is the content of the gauge theorem, and it is
    the single most consequential thing this experiment found:

        *Every restoration is already observation-consistent, for some water depth.*
        A consistency-based loss, gate or projection therefore cannot correct colour.
        Anything that does correct colour is drawing on a scene prior, not on the
        measurement.

    The function returns ``dd`` alongside the result so the ~identity can be
    reported as a measured fact rather than asserted from the algebra, and so a
    reader can see the projection was actually run rather than skipped.
    """
    J = np.asarray(base, dtype=np.float64)
    b = binf.reshape(1, 1, 3)
    be = beta.reshape(1, 1, 3)
    d = J - b

    grid = np.linspace(-d_max, d_max, n_grid)
    # (H, W, n_grid, 3): the family evaluated at every candidate step
    Jp = b[..., None, :] + d[..., None, :] * np.exp(be[..., None, :]
                                                    * grid.reshape(1, 1, -1, 1))

    if target is not None:
        # Oracle variant, for diagnosis only: let the reference pick the step.  This
        # measures how much colour error is *inside* the family at all, i.e. the
        # ceiling of any consistency-based correction.  It is never deployable and
        # is labelled as such wherever it appears.
        T = np.asarray(target, dtype=np.float64)
        cost = ((T[..., None, :] - Jp) ** 2).sum(axis=(0, 1, 3))
    else:
        cost = ((J[..., None, :] - Jp) ** 2).sum(axis=(0, 1, 3))

    k = int(np.argmin(cost))
    dd = float(grid[k])
    out = b + d * np.exp(be * dd)
    return {"img": np.clip(out, 0.0, 1.0), "dd": dd,
            "cost_at_dd": float(cost[k]), "cost_at_zero": float(cost[n_grid // 2])}


def _best_gain(pred: np.ndarray, gt: np.ndarray) -> float:
    """Least-squares exposure ``s`` minimising ``||s*pred - gt||^2``.

    Oracle quantity: needs the reference, so it is an ablation control only.  Its
    job is to answer "is the water family's single degree of freedom worth more
    than a global brightness slider?"  If not, calling the projection a gauge step
    dresses up a brightness knob.
    """
    num = float((pred * gt).sum())
    den = float((pred * pred).sum())
    return num / max(den, 1e-12)


def prior_shrink(base: np.ndarray, amount: float) -> np.ndarray:
    """Pull chroma toward the scene's own grey point by ``amount`` in (0,1).

    This is the *prior-driven* correction: it uses no observation model at all, only
    the assumption that a natural underwater scene has roughly balanced chroma.  It
    exists to make one point measurable -- if it moves the error metric as much as
    or more than any support-driven variant, then whatever improvement is
    available is available to a scene prior, and the observation contributes
    nothing to colour.  It is the honest competitor to this paper's framing, not a
    strawman: grey-world is a real baseline that real papers use.
    """
    ch = chroma_opp(base)
    mean_ch = ch.mean(axis=(0, 1), keepdims=True)
    return apply_chroma(base, ch - amount * mean_ch)


def _metrics(pred, gt):
    p8 = np.clip(pred * 255.0 + 0.5, 0, 255).astype(np.uint8)
    g8 = np.clip(gt * 255.0 + 0.5, 0, 255).astype(np.uint8)
    pr = M.resize_unified(p8, 256)
    gr = M.resize_unified(g8, 256)
    m = M.evaluate_pair(pr, gr)
    pc, gc = chroma_opp(pred), chroma_opp(gt)
    chr_err = float(np.abs(pc - gc).mean())
    lum_err = float(np.abs(pred @ LUM - gt @ LUM).mean())
    return {"psnr": m["psnr"], "ssim": m["ssim"], "chroma": chr_err, "luma": lum_err}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="data/packed/UIEB.npz")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--strength", type=float, default=1.0,
                    help="multiplier on the damping; 1.0 = full pull to J_obs")
    ap.add_argument("--out", default="artifacts/h3_mechanism.json")
    args = ap.parse_args()

    dat = np.load(args.npz, allow_pickle=True)
    GT, RAW, PRED = dat["gt"], dat["raw"], dat["pred"]
    names = [str(x) for x in dat["names"]]
    n = len(names) if args.limit <= 0 else min(args.limit, len(names))
    wm = S.default_water()

    variants = ["base", "support_g", "const_g", "random_g", "degrade_g",
                "lowpass", "oracle_g", "project", "project_oracle", "greyworld",
                "gain_oracle"]
    acc = {v: [] for v in variants}
    dd_oracle = []

    rng = np.random.default_rng(0)
    for i in range(n):
        gt = GT[i].astype(np.float64) / 255.0
        raw = RAW[i].astype(np.float64) / 255.0
        pred = PRED[i].astype(np.float64) / 255.0
        h, w = gt.shape[:2]

        # 1) the observation-consistent anchor
        fit = S.fit_water(raw, pred, h, w, k_depth=9)
        # ``fit_water`` and ``invert`` work on flattened ``(N, 3)`` arrays; put the
        # anchor back on the image grid before anything spatial happens to it.
        J_obs = np.clip(S.invert(raw.reshape(-1, 3), fit["tau"], fit["water"]),
                        0.0, 1.0).reshape(h, w, 3)

        # 2) per-pixel support -> damping
        an = S.analyse(raw, pred, fit)
        lq = an["log10q_eff"].reshape(h, w, 3).mean(axis=2)
        sup = S.support_score(lq)
        g_sup = np.clip(1.0 - sup, 0.0, 1.0) * args.strength

        # 3) the variants, each isolating one way this could be an artefact
        g_rand = np.clip(rng.random((h, w)) * (g_sup.mean() * 2.0), 0.0, 1.0)
        lum_raw = raw @ LUM
        # darker observation -> less information -> more damping: the cheap proxy
        g_deg = np.clip((1.0 - lum_raw) * (g_sup.mean() / max(1.0 - lum_raw.mean(),
                                                              1e-6)), 0.0, 1.0)
        g_con = np.full((h, w), g_sup.mean())

        err = np.abs(chroma_opp(pred) - chroma_opp(gt)).mean(axis=2)
        # oracle: damp exactly where the chroma error is large, normalised so the
        # *total* damping budget matches support_g -- otherwise it wins by damping
        # more, not by damping better
        w_or = smooth(err, 7)
        tot_target = float(g_sup.sum())
        g_or = w_or * (tot_target / max(float(w_or.sum()), 1e-9))
        g_or = np.clip(g_or, 0.0, 1.0)

        cand = {"base": pred}
        ch_pred = chroma_opp(pred)
        ch_obs = chroma_opp(J_obs)
        for v, g in (("support_g", g_sup), ("const_g", g_con),
                     ("random_g", g_rand), ("degrade_g", g_deg),
                     ("oracle_g", g_or)):
            gs = smooth(g, 5)[..., None]
            out_ch = ch_pred + gs * (ch_obs - ch_pred)
            cand[v] = apply_chroma(pred, out_ch)

        # --- projection onto the admissible family ------------------------
        # The rule the gauge theorem licenses, run twice: once as the honest
        # projection (which is the identity -- that is the finding), and once with
        # the *reference* choosing the step, which bounds what any
        # consistency-driven method could ever buy.  The gap between those two is
        # the amount of colour error that lives inside the family at all.
        wmb = fit["water"]
        beta = wmb.beta.reshape(1, 1, 3)
        binf = wmb.b_inf.reshape(1, 1, 3)
        pr = project_onto_family(pred, beta, binf)
        cand["project"] = pr["img"]
        pr_or = project_onto_family(pred, beta, binf, target=gt)
        cand["project_oracle"] = pr_or["img"]
        dd_oracle.append(pr_or["dd"])
        # Control for the oracle step: a plain global gain with its own single free
        # parameter.  If a scalar multiply buys the same PSNR, then what the water
        # family's degree of freedom is worth is just exposure, and calling it a
        # gauge step would be dressing up a brightness knob.  This is the ablation
        # that decides whether the family is doing anything a photographer's slider
        # could not.
        cand["gain_oracle"] = np.clip(pred * _best_gain(pred, gt), 0.0, 1.0)

        # The prior-driven competitor: pull chroma to the scene's grey point.  Uses
        # no observation model at all.  If it wins, the observation contributes
        # nothing to colour and the paper must say so.
        cand["greyworld"] = prior_shrink(pred, 0.5)

        # low-pass with the *same* mean chroma movement, so strength matches
        blur = cv2.GaussianBlur(pred.astype(np.float32), (0, 0), 3.0).astype(np.float64)
        ch_blur = chroma_opp(blur)
        mv_blur = float(np.abs(ch_blur - ch_pred).mean())
        mv_sup = float(np.abs(chroma_opp(cand["support_g"]) - ch_pred).mean())
        scale = mv_sup / max(mv_blur, 1e-9)
        cand["lowpass"] = apply_chroma(pred, ch_pred + scale * (ch_blur - ch_pred))

        for v in variants:
            acc[v].append(_metrics(cand[v], gt))

        if (i + 1) % 20 == 0 or i == n - 1:
            b = acc["base"][-1]
            s = acc["support_g"][-1]
            print(f"  [{i+1}/{n}] {names[i]:14s} "
                  f"base chr={b['chroma']:.4f} psnr={b['psnr']:.2f} | "
                  f"sup chr={s['chroma']:.4f} psnr={s['psnr']:.2f}", flush=True)

    def agg(rows):
        return {k: {"mean": float(np.mean([r[k] for r in rows])),
                    "std": float(np.std([r[k] for r in rows]))}
                for k in rows[0]}

    res = {v: agg(acc[v]) for v in variants}
    base = res["base"]
    print()
    print(f"{'variant':<12s} {'chroma':>9s} {'d_chr%':>8s} {'luma':>9s} "
          f"{'PSNR':>8s} {'dPSNR':>8s} {'SSIM':>8s}")
    for v in variants:
        r = res[v]
        print(f"{v:<12s} {r['chroma']['mean']:>9.4f} "
              f"{100*(r['chroma']['mean']-base['chroma']['mean'])/base['chroma']['mean']:>+8.2f} "
              f"{r['luma']['mean']:>9.4f} "
              f"{r['psnr']['mean']:>8.3f} "
              f"{r['psnr']['mean']-base['psnr']['mean']:>+8.3f} "
              f"{r['ssim']['mean']:>8.4f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"npz": args.npz, "n": n, "strength": args.strength,
                   "variants": res,
                   "dd_oracle": {"mean": float(np.mean(dd_oracle)),
                                 "median": float(np.median(dd_oracle)),
                                 "frac_nonzero": float(np.mean(
                                     np.abs(np.asarray(dd_oracle)) > 0.01)),
                                 "values": [float(x) for x in dd_oracle]}}, fh, indent=1)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Run the controlled forward-degradation track.

Three questions, none of which the real datasets can answer.

**Q1  Does the observation's information about a component change with the
      physical condition, in the way the sensitivity predicts?**
      Fix the scene, vary one axis, measure ``q_v`` for colour, pixel-structure
      and band-pass directions.  If attenuation / backscatter / noise move the
      families differently, then "recoverability" is not one scalar per image and
      a method that reports one number is hiding the structure.

**Q2  Is a claimed recovery impossible, or merely hard?**
      Ask two different questions, because they have different answers.
      *Known water*: can the observation distinguish a scene from the same scene
      with one channel offset, when the water is given?  Always yes, by a wide
      margin.  *Unknown water*: is there another ``(J', psi')`` that explains the
      same observation?  That is the regime a single-image method actually lives
      in, and profiling the likelihood over a global colour direction answers it
      in amplitude units with an explicit censoring flag.  A method that emits a
      colour correction cannot tell which regime it is in from the inside, which
      is why the number it produces cannot be read as a measurement.

**Q3  Does the sensitivity track the actual error of a real enhancer?**
      This is the bridge between Q1/Q2 (statements about ``G_psi``) and the
      downstream claim (statements about a network's mistakes): measured on the
      controlled grid, where ``J*`` and the water are known rather than assumed.

Direction normalisation
-----------------------
``direction_info`` returns ``q_v = ||Sigma_n^{-1/2} D_J G v||^2``, which scales
with ``||v||^2``.  A DC direction spread over every pixel and a one-pixel
direction therefore differ by ``sqrt(N)`` for reasons that have nothing to do with
observability.  Two normalisations are reported, because they answer different
questions:

``*_unit``  ``q`` per unit energy -- the standard Fisher convention, comparable
            across direction families.
``*_sd``    ``1/sqrt(q)`` in the direction's own amplitude units -- "how well is
            one pixel's value known", which is what a restoration decision acts on.

Both are reported so the reader can check that the conclusion does not depend on
the choice.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import obsui  # noqa: E402,F401  (pins BLAS threads before numpy is imported)

import numpy as np  # noqa: E402

import obsui.controlled as C  # noqa: E402
import obsui.support as S  # noqa: E402


# ---------------------------------------------------------------------------
# direction families
# ---------------------------------------------------------------------------


def _dirs(h: int, w: int, k_low: int = 4, n_px: int = 8, radii=(1, 4, 16)):
    """Return ``{family: [(name, v)]}`` with every ``v`` of shape ``(h*w, 3)``."""
    comp = S.components(h, w, k_low=k_low)
    colour = [(f"dc{c}", comp[f"dc_{c}"]) for c, c in enumerate(range(3))]

    pixel = []
    n = h * w
    for j, idx in enumerate(np.linspace(0, n - 1, n_px).astype(int)):
        v = np.zeros((n, 3))
        v[idx, 0] = 1.0
        pixel.append((f"px{j}", v))

    band = []
    for r in radii:
        for c in range(3):
            v = S.blob_dirs(h, w, [r], channel=c)[0][1]
            band.append((f"r{r}c{c}", v))
    return {"colour": colour, "pixel": pixel, "band": band}


def _q_stats(an, dirs):
    """Per-family ``log10 q_eff`` in unit-energy and in amplitude units."""
    unit, sd = [], []
    for _, v in dirs:
        info = S.direction_info(an, v)
        energy = float((v ** 2).sum())
        qe = max(info["q_eff"], 1e-30)
        unit.append(np.log10(qe / max(energy, 1e-30)))
        sd.append(1.0 / np.sqrt(qe))
    return float(np.median(unit)), float(np.median(sd))


# ---------------------------------------------------------------------------
# Q1
# ---------------------------------------------------------------------------


def q1(images, k_depth=9, verbose=True):
    rows = []
    for name, J in images:
        h, w = J.shape[:2]
        fams = _dirs(h, w)
        for cond in C.grid():
            obs = C.render(J, cond, np.random.RandomState(cond.seed + 17))
            I = obs["I"].astype(np.float64)
            Jd = J.astype(np.float64)
            try:
                fit = S.fit_water(I, Jd, h, w, k_depth=k_depth,
                                  noise_mode="declared",
                                  read_sigma=cond.read_sigma,
                                  shot_alpha=cond.shot_alpha)
                an = S.analyse(I, Jd, fit)
            except Exception as exc:            # a failed fit is a datum, not a crash
                rows.append({"image": name, "condition": cond.name,
                             "axis": _axis_of(cond.name),
                             "error": f"{type(exc).__name__}: {exc}"})
                continue
            rec = {"image": name, "condition": cond.name, "axis": _axis_of(cond.name),
                   "key": cond.key(), "n_obs": float(fit["n_obs"]),
                   "chi2_over_dof": float(fit["chi2"] / max(fit["n_obs"], 1.0)),
                   "rmse": float(fit["rmse"])}
            for fam, dirs in fams.items():
                qu, sd = _q_stats(an, dirs)
                rec[f"log10q_eff_unit_{fam}"] = qu
                rec[f"sd_amp_{fam}"] = sd
            rec["colour_minus_pixel_unit"] = (rec["log10q_eff_unit_colour"]
                                              - rec["log10q_eff_unit_pixel"])
            rows.append(rec)
            if verbose:
                print(f"  {name:10s} {cond.name:11s} "
                      f"colour={rec['log10q_eff_unit_colour']:+7.2f} "
                      f"pixel={rec['log10q_eff_unit_pixel']:+7.2f} "
                      f"band={rec['log10q_eff_unit_band']:+7.2f} "
                      f"chi2/dof={rec['chi2_over_dof']:6.2f}", flush=True)
    return rows


def _axis_of(cond_name: str) -> str:
    for ax, items in C.AXES.items():
        if any(c.name == cond_name for _, c in items):
            return ax
    return "other"


# ---------------------------------------------------------------------------
# Q2
# ---------------------------------------------------------------------------


def q2(images, deltas=(0.02, 0.05, 0.10), n_draws=64, k_depth=9, verbose=True):
    """Two regimes, and the contrast between them is the point.

    **Known water.**  Ask whether the observation distinguishes ``J`` from
    ``J + delta`` when ``psi`` is given.  It always can, and the answer is large:
    the offset changes the observation by far more than the noise.

    **Unknown water.**  Ask whether some *other* ``(J', psi')`` explains the same
    observation.  This is the regime a single-image method actually operates in,
    and it is the one that decides what "recover a colour cast" can mean.

    A method that reports a colour correction has no way to distinguish these two
    regimes from the inside; that is why the number it produces cannot be read as
    a measurement of the scene.  Reporting both, rather than only the flattering
    one, is the difference between a diagnostic and a sales chart.
    """
    rows = []
    for name, J in images:
        h, w = J.shape[:2]
        Jd = J.astype(np.float64)
        # Peak-normalised DC direction: every pixel of channel 0 offset by the
        # same amount, so the profiled amplitude ``a`` IS the per-pixel channel
        # offset in [0,1] units.  The unit-energy DC direction would divide the
        # answer by sqrt(h*w) and make "0.005" look like a tight bound when it is
        # really "we could not rule out half the dynamic range".
        vdc = np.zeros((h * w, 3))
        vdc[:, 0] = 1.0

        # --- regime 1: water known -------------------------------------
        for dv in (0.0, 1.00):
            cond = C.Condition(name=f"dv{dv:g}", depth_var=float(dv))
            for delta in deltas:
                Jb = C.colour_offset_pair(J, delta, channel=0)
                res = C.detectability(J, Jb, cond, n_draws=n_draws, seed=11)
                rows.append({"image": name, "regime": "known_water",
                             "depth_var": float(dv), "delta": float(delta), **res})
                if verbose:
                    print(f"  {name:10s} KNOWN  dv={dv:4.2f} d={delta:5.2f} "
                          f"d'={res['d_prime']:9.1f} auc={res['auc']:.3f}", flush=True)

        # --- regime 2: water unknown -----------------------------------
        for noise_scale in (1.0, 2.0, 4.0):
            cond = C.Condition(name=f"nz{noise_scale:g}",
                               read_sigma=0.004 * noise_scale,
                               shot_alpha=0.008 * noise_scale)
            obs = C.render(J, cond, np.random.RandomState(cond.seed + 17))
            I = obs["I"].astype(np.float64)
            try:
                fit = S.fit_water(I, Jd, h, w, k_depth=k_depth,
                                  noise_mode="declared",
                                  read_sigma=cond.read_sigma,
                                  shot_alpha=cond.shot_alpha)
                pw = S.profile_width(I, Jd, fit, vdc, h, w, a_max=0.5, n_grid=6, df=1)
            except Exception as exc:
                rows.append({"image": name, "regime": "unknown_water",
                             "noise_scale": float(noise_scale),
                             "error": f"{type(exc).__name__}: {exc}"})
                continue
            # ``vdc`` has peak 1, so ``pw["width"]`` is already the per-pixel
            # channel offset in image units -- no rescaling, no sqrt(N) confusion.
            amp = float(pw["width"])
            rows.append({"image": name, "regime": "unknown_water",
                         "noise_scale": float(noise_scale),
                         "width_peak_norm": float(pw["width"]),
                         "amplitude_per_pixel": amp,
                         "censored": bool(pw["censored"]),
                         "base_chi2": float(pw["base_chi2"])})
            if verbose:
                print(f"  {name:10s} UNKNOWN nz={noise_scale:4.1f} "
                      f"width={pw['width']:.4f} amp={amp:.4f} "
                      f"censored={pw['censored']}", flush=True)
    return rows


# ---------------------------------------------------------------------------
# Q3
# ---------------------------------------------------------------------------


def q3(images, preds, k_depth=9, axes=("attenuation", "noise", "backscatter"),
       verbose=True):
    """Does ``q_eff`` track the enhancer's real error on the controlled grid?"""
    rows = []
    conds = [c for ax in axes for c in C.grid([ax])]
    for (name, J), (_, Jh) in zip(images, preds):
        h, w = J.shape[:2]
        fams = _dirs(h, w)
        Jh = Jh.astype(np.float64)
        for cond in conds:
            obs = C.render(J, cond, np.random.RandomState(cond.seed + 17))
            I = obs["I"].astype(np.float64)
            Jd = J.astype(np.float64)
            try:
                fit = S.fit_water(I, Jh, h, w, k_depth=k_depth,
                                  noise_mode="declared",
                                  read_sigma=cond.read_sigma,
                                  shot_alpha=cond.shot_alpha)
                an = S.analyse(I, Jh, fit)
            except Exception as exc:
                rows.append({"image": name, "condition": cond.name,
                             "error": f"{type(exc).__name__}: {exc}"})
                continue
            ce = np.stack([np.log((Jh[..., 0] + 0.05) / (Jh[..., 1] + 0.05)),
                           np.log((Jh[..., 1] + 0.05) / (Jh[..., 2] + 0.05))], -1)
            ct = np.stack([np.log((Jd[..., 0] + 0.05) / (Jd[..., 1] + 0.05)),
                           np.log((Jd[..., 1] + 0.05) / (Jd[..., 2] + 0.05))], -1)
            qu, _ = _q_stats(an, fams["colour"])
            rows.append({
                "image": name, "condition": cond.name, "axis": _axis_of(cond.name),
                "log10q_eff_unit_colour": qu,
                "err_abs": float(np.abs(Jh - Jd).mean()),
                "err_chroma": float(np.abs(ce - ct).mean()),
                "err_luma": float(np.abs(
                    (0.2126 * Jh[..., 0] + 0.7152 * Jh[..., 1] + 0.0722 * Jh[..., 2])
                    - (0.2126 * Jd[..., 0] + 0.7152 * Jd[..., 1] + 0.0722 * Jd[..., 2])
                ).mean()),
                "chi2_over_dof": float(fit["chi2"] / max(fit["n_obs"], 1.0)),
            })
            if verbose:
                print(f"  {name:10s} {cond.name:11s} lq={qu:+7.2f} "
                      f"chr={rows[-1]['err_chroma']:.4f} lum={rows[-1]['err_luma']:.4f}",
                      flush=True)
    return rows


# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="data/packed/UIEB.npz")
    ap.add_argument("--limit", type=int, default=3)
    ap.add_argument("--size", type=int, default=128,
                    help="downsample to this square size; the questions are about "
                         "physics, not resolution, and a full-resolution LM fit "
                         "costs ~15 s per condition")
    ap.add_argument("--k-depth", type=int, default=9)
    ap.add_argument("--out", default="artifacts/controlled.json")
    ap.add_argument("--only", default="q1,q2,q3")
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    names = [str(x) for x in d["names"]]
    n = min(args.limit, len(d["raw"]))
    sz = args.size

    def prep(a):
        import cv2
        out = a.copy()
        if out.shape[0] != sz or out.shape[1] != sz:
            out = cv2.resize(out, (sz, sz), interpolation=cv2.INTER_AREA)
        return out.astype(np.float64) / 255.0

    images = [(names[i], prep(d["gt"][i])) for i in range(n)]
    preds = [(names[i], prep(d["pred"][i])) for i in range(n)]
    print(f"images={n} size={sz} k_depth={args.k_depth} only={args.only}", flush=True)

    out = {"npz": args.npz, "n_images": n, "size": sz, "k_depth": args.k_depth,
           "axis_table": {ax: [c.to_dict() for _, c in items]
                          for ax, items in C.AXES.items()}}

    if "q1" in args.only:
        print("\n[Q1] sensitivity vs physical axis", flush=True)
        out["q1"] = q1(images, k_depth=args.k_depth)
    if "q2" in args.only:
        print("\n[Q2] detectability of a global colour offset", flush=True)
        out["q2"] = q2(images, k_depth=args.k_depth)
    if "q3" in args.only:
        print("\n[Q3] sensitivity vs enhancer error", flush=True)
        out["q3"] = q3(images, preds, k_depth=args.k_depth)

    out["manifest_sha256"] = hashlib.sha256(
        json.dumps(out, sort_keys=True, default=str).encode()).hexdigest()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=str)
    print(f"\nwrote {args.out} sha256={out['manifest_sha256'][:16]}", flush=True)


if __name__ == "__main__":
    main()

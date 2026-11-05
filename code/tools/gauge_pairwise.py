"""How much of the gap between two enhancement methods is not a measurement?

:mod:`obsui.gauge` establishes that the observation cannot separate

    J  <->  J'(dd) = b + (J - b) * exp(beta * dd),   d  <->  d + dd

for any ``dd`` that keeps the pair physically admissible.  Every member of that
family reproduces the input pixel-for-pixel.  So if method B's output sits inside
method A's family, the two methods **did not disagree about anything the data
constrains** -- and the PSNR/SSIM gap between them measures the methods' priors,
not the scene.

This script answers that for every ordered pair of methods:

    for each image,
        fit dd that best maps A's output onto B's output, in closed form
        record the fraction of the A-B difference that this single degree of
        freedom removes, and whether the required dd is physically admissible

The fraction is a **lower bound**: only the constant-``dd`` member of the family
is searched, so a spatially varying ``dd`` could explain strictly more.  Reporting
a lower bound is the honest direction -- it cannot manufacture the conclusion that
gaps are meaningless.

Read the output as: "for this image, ``frac`` of the squared difference between
the two methods is reachable without changing the observation".  A high ``frac``
with an admissible ``dd`` means the ranking between those two methods on that
image is decided by an unfalsifiable choice.

The script never uses the reference to build the family -- the family comes from
the observation plus the water model.  The reference appears only in the separate
``family_spread`` block, which reports what the gauge freedom is worth in dB.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import obsui  # noqa: E402,F401

import numpy as np  # noqa: E402

import obsui.gauge as G  # noqa: E402
import obsui.support as S  # noqa: E402


def _q(a, p):
    return float(np.percentile(a, p)) if len(a) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="data/packed/UIEB.npz")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="artifacts/gauge_pairwise.json")
    args = ap.parse_args()

    dat = np.load(args.npz, allow_pickle=True)
    # NpzFile is lazy -- read once, not once per image.
    GT = dat["gt"]
    RAW = dat["raw"]
    pred = dat["pred"]
    names = [str(x) for x in dat["names"]]
    wm = S.default_water()
    n = len(names) if args.limit <= 0 else min(args.limit, len(names))

    pred = dat["pred"]
    if pred.ndim == 5:
        methods = [str(x) for x in dat["methods"]]
    else:
        methods = ["pred"]
        pred = pred[:, None]
    print(f"npz={args.npz} n={n} methods={methods}", flush=True)

    # 1) How many dB could a method win by moving ONLY inside the gauge family?
    #
    # The anchor is the method's own output, not the reference.  The family is a set
    # of scenes that all render the SAME observation, so sliding a prediction along
    # it costs nothing in consistency: it is a choice of prior, not a use of
    # information.  ``psnr_gain_max`` is therefore an upper bound on PSNR that is
    # available for free -- a method gaining less than this has not shown that its
    # colour came from measurement.
    fam = []
    for i in range(n):
        gt = GT[i].astype(np.float64) / 255.0
        raw = RAW[i].astype(np.float64) / 255.0
        fit = S.fit_water(raw, gt, gt.shape[0], gt.shape[1], k_depth=9,
                          noise_mode="misfit")
        for mi, mname in enumerate(methods):
            jhat = pred[i, mi].astype(np.float64) / 255.0
            sp = G.family_spread(jhat, gt, wm, d_ref=fit["depth"])
            sp["name"] = names[i]
            sp["method"] = mname
            fam.append(sp)
        if (i + 1) % 20 == 0 or i == n - 1:
            print("  family [%d/%d] " % (i + 1, n) + " | ".join(
                "%s free=%+.2f dB band=%.2f" %
                (f["method"], f["psnr_gain_max"], f["psnr_spread"])
                for f in fam[-len(methods):]), flush=True)

    gain_arr = np.array([f["psnr_gain_max"] for f in fam])
    band_arr = np.array([f["psnr_spread"] for f in fam])
    print("")
    print("[gauge family per method output]  n=%d" % len(gain_arr))
    print("  PSNR available free inside the family (dB): mean=%.2f median=%.2f "
          "p90=%.2f max=%.2f" % (gain_arr.mean(), _q(gain_arr, 50),
                                 _q(gain_arr, 90), gain_arr.max()))
    print("  band where the observation cannot rank colour (dB): mean=%.2f "
          "median=%.2f p90=%.2f" % (band_arr.mean(), _q(band_arr, 50),
                                    _q(band_arr, 90)))

    sp_arr = np.array([f["psnr_spread"] for f in fam])
    print("")
    print("[gauge band, method outputs]  n=%d" % len(sp_arr))
    print("  spread (dB): mean=%.2f median=%.2f p90=%.2f max=%.2f"
          % (sp_arr.mean(), _q(sp_arr, 50), _q(sp_arr, 90), sp_arr.max()))

    # 2) Pairwise unfalsifiability, closed form
    rows = []
    for a, b in itertools.permutations(range(len(methods)), 2):
        fr, ddv, adm = [], [], 0
        for i in range(n):
            Ja = pred[i, a].astype(np.float64) / 255.0
            Jb = pred[i, b].astype(np.float64) / 255.0
            r = G.best_step(Ja, Jb, wm)
            if r["admissible"]:
                adm += 1
                fr.append(r["frac"])
                ddv.append(r["dd"])
        if not fr:
            continue
        fr = np.array(fr)
        rows.append({"a": methods[a], "b": methods[b], "n": len(fr),
                     "frac_mean": float(fr.mean()),
                     "frac_median": _q(fr, 50),
                     "frac_gt50": float((fr > 0.5).mean()),
                     "dd_median": _q(ddv, 50)})
        print(f"  {methods[a]:>12s} -> {methods[b]:<12s} n={len(fr)} "
              f"frac mean={fr.mean():.3f} med={_q(fr,50):.3f} "
              f">0.5: {(fr>0.5).mean()*100:.0f}%  dd_med={_q(ddv,50):+.3f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"npz": args.npz, "n": n, "methods": methods,
                   "family": fam, "pairs": rows,
                   "family_summary": {
                       "psnr_gain_max_mean": float(gain_arr.mean()),
                       "psnr_gain_max_median": _q(gain_arr, 50),
                       "psnr_gain_max_p90": _q(gain_arr, 90),
                       "psnr_gain_max_max": float(gain_arr.max()),
                       "band_mean": float(band_arr.mean()),
                       "band_median": _q(band_arr, 50),
                       "band_p90": _q(band_arr, 90),
                   }},
                  fh, indent=1)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

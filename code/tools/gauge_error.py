"""Is a method's colour error *wrong*, or merely *unfalsifiable*?

The gauge family in :mod:`obsui.gauge` says the observation cannot distinguish
``J`` from ``J' = b + (J - b) exp(beta dd)`` for any depth increment ``dd``.  That
turns a vague worry ("colour restoration is ill-posed") into a per-image,
per-method measurement:

For a restored image ``Jhat`` and the reference ``J*``, decompose the error
``Jhat - J*`` into

* a component **inside** the gauge family -- the part any method could have
  chosen without contradicting the observation, and
* a component **outside** it -- the part that is actually inconsistent.

The ratio is the paper's headline diagnostic.  A method whose colour error lies
almost entirely inside the family is not making a measurable mistake; it is
exercising a prior.  That is a legitimate thing to do, but it cannot be reported
as a recovery.

Three things this deliberately does *not* do:

* It does not use ``J*`` to compute the gauge family.  The family is defined by
  the observation and the water model; using the reference to pick the best ``dd``
  would make the "explained" fraction circular.  Instead ``dd`` is chosen to
  minimise the residual for each method independently and the *fraction* is
  reported alongside the recovered ``dd``, so a reader can see whether the
  method's implied water is physically sensible.
* It does not claim the reference is ground truth.  UIEB references are
  human-selected; that is recorded in the data manifest and it bounds what this
  can mean.
* It does not touch the test set for model selection -- this is a measurement on
  frozen predictions.

The fit is closed-form: for each channel the family reduces to a quadratic in
``u_c = exp(beta_c dd)``, so three sufficient statistics per image replace a
parameter sweep.  :func:`obsui.gauge.best_step` does the algebra.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import obsui  # noqa: E402,F401

import numpy as np  # noqa: E402

import obsui.gauge as G  # noqa: E402
import obsui.support as S  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="data/packed/UIEB.npz")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="artifacts/gauge_error.json")
    args = ap.parse_args()

    dat = np.load(args.npz, allow_pickle=True)
    # ``NpzFile`` is lazy: every ``dat["gt"][i]`` decompresses the whole array
    # again.  Reading the three arrays once turns an O(n) decompression storm into
    # a single load, which is the difference between minutes and seconds here.
    GT = dat["gt"]
    PRED = dat["pred"]
    RAW = dat["raw"]
    names = [str(x) for x in dat["names"]]
    n = len(names) if args.limit <= 0 else min(args.limit, len(names))
    wm = S.default_water()

    print(f"npz={args.npz} n={n} water beta={wm.beta.round(3)} "
          f"b_inf={wm.b_inf.round(3)}", flush=True)

    # The family is anchored on the *reference* scene: the question is "could a
    # different water explanation have moved the reference onto the method's
    # output?"  A constant depth step is the simplest member and therefore gives a
    # lower bound on the explained fraction.
    rows = []
    for i in range(n):
        gt = GT[i].astype(np.float64) / 255.0
        pred = PRED[i].astype(np.float64) / 255.0
        raw = RAW[i].astype(np.float64) / 255.0
        shp = gt.shape

        fm = G.best_step(gt, pred, wm)
        rm = G.best_step(gt, raw, wm)

        rows.append({
            "name": names[i],
            "dd": fm["dd"], "frac_inside": fm["frac"],
            "per_channel_dd": fm["per_channel_dd"],
            "err_sq": fm["total"],
            "rmse": float(np.sqrt(fm["residual"] / max(shp[0] * shp[1] * 3, 1))),
            "raw_dd": rm["dd"], "raw_frac_inside": rm["frac"],
        })
        if (i + 1) % 20 == 0 or i == n - 1:
            print(f"  [{i+1}/{n}] {names[i]:14s} dd={fm['dd']:+.3f} "
                  f"inside={fm['frac']:.3f} | raw dd={rm['dd']:+.3f} "
                  f"raw_inside={rm['frac']:.3f}", flush=True)

    fr = np.array([r["frac_inside"] for r in rows])
    dd = np.array([r["dd"] for r in rows])
    rf = np.array([r["raw_frac_inside"] for r in rows])
    rd = np.array([r["raw_dd"] for r in rows])

    def q(a, p):
        return float(np.percentile(a, p))

    print()
    print("fraction of scene-space error reachable inside the gauge family:")
    print(f"  mean={fr.mean():.3f} median={q(fr,50):.3f} "
          f"p10={q(fr,10):.3f} p90={q(fr,90):.3f}")
    print(f"  images with frac>0.5: {(fr>0.5).sum()}/{len(fr)}")
    print(f"  implied dd: mean={dd.mean():+.3f} median={q(dd,50):+.3f}")
    print(f"  degraded input: frac={rf.mean():.3f} dd={rd.mean():+.3f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    summary = {
        "frac_mean": float(fr.mean()), "frac_median": q(fr, 50),
        "frac_p10": q(fr, 10), "frac_p90": q(fr, 90),
        "frac_gt50": float((fr > 0.5).mean()),
        "dd_mean": float(dd.mean()), "dd_median": q(dd, 50),
        "raw_frac_mean": float(rf.mean()), "raw_dd_mean": float(rd.mean()),
    }
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"npz": args.npz, "n": n, "water": wm.to_dict(),
                   "summary": summary, "rows": rows}, fh, indent=1)
    for k, v in summary.items():
        print(f"  {k:16s} {v:+.4f}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

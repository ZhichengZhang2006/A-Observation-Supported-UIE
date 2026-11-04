"""Is the colour ambiguity a theorem or a numerical accident?

Two things must hold before the gauge family in :mod:`obsui.gauge` can carry a
claim:

1. The transform must be **exact** -- the rendered observation must not change at
   all, to round-off.  If it drifted, the family would be an approximation and a
   strong enough prior could break it.
2. The resulting ambiguity must be **large enough to matter** in the units an
   enhancement method works in, and it must shrink where the physics says it
   should (bright scenes, little headroom).

Both are checked here against the same forward model the fitter uses, with no
network in the loop, so a positive result cannot be an artefact of training.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import obsui  # noqa: E402,F401

import numpy as np  # noqa: E402

import obsui.controlled as C  # noqa: E402
import obsui.gauge as G  # noqa: E402
import obsui.support as S  # noqa: E402

OK = []
BAD = []


def check(name, cond, detail=""):
    (OK if cond else BAD).append(name)
    print(f"  {'PASS' if cond else 'FAIL'} {name} {detail}", flush=True)


def main():
    import cv2

    h = w = 96
    rng = np.random.RandomState(3)
    wm = S.default_water()
    beta = wm.beta.reshape(1, 3)
    d = 0.35 + 0.45 * rng.rand(h, w)

    print("=" * 70)
    print("G1  the transform is exact")
    print("=" * 70)
    J = rng.uniform(0.15, 0.85, size=(h, w, 3))
    for dd in (-0.35, -0.1, 0.0, 0.1, 0.35):
        r = G.verify_exact(J, d, wm, np.full(h * w, float(dd)))
        check(f"G1.exact_dd{dd:+.2f}", r["max_abs_diff"] < 1e-10,
              f"max|dI|={r['max_abs_diff']:.2e} (scale {r['scale']:.2f})")

    print()
    print("=" * 70)
    print("G2  the ambiguity is chromatic, and tracks signal not scene level")
    print("=" * 70)
    J = rng.uniform(0.15, 0.75, size=(h, w, 3))
    rows = G.basis_budget(J, wm, None, d)
    for r in rows:
        print(f"  dd={r['dd_step']:+.2f} used={r['dd_used']:+.3f} "
              f"adm={r['admissible']} chroma_p50={r['chroma_p50']:.3f} "
              f"p95={r['chroma_p95']:.3f} lift_err={r['lift_err_max']:.2e} "
              f"lift=({r['lift_pred_r']:+.3f},{r['lift_pred_g']:+.3f},"
              f"{r['lift_pred_b']:+.3f})", flush=True)
    pos = [r for r in rows if r["dd_step"] > 0]
    neg = [r for r in rows if r["dd_step"] < 0]
    # The lift must be exactly exp(beta_c dd) - 1 applied to the signal above
    # veiling.  A plain brightness gain would divide by J instead of (J - b) and
    # would fail this to many digits.
    check("G2.lift_is_exp_beta_dd",
          all(r["lift_err_max"] < 1e-12 for r in rows),
          f"max lift_err={max(r['lift_err_max'] for r in rows):.2e}")
    # beta differs per channel, so the required lift differs per channel: that is
    # what makes this a colour statement rather than a brightness statement.
    check("G2.lift_is_chromatic",
          abs(pos[0]["lift_pred_r"] - pos[0]["lift_pred_b"]) > 0.01,
          f"lift_R={pos[0]['lift_pred_r']:+.4f} vs "
          f"lift_B={pos[0]['lift_pred_b']:+.4f}")
    check("G2.positive_side_bounded",
          pos[-1]["chroma_p95"] <= pos[0]["chroma_p95"] * 8 + 1e-6,
          "positive side saturates at dd_pos_max")
    check("G2.negative_side_available",
          neg[0]["admissible"] and neg[0]["chroma_p50"] > 0,
          f"dd={neg[0]['dd_step']:+.2f} chroma_p50={neg[0]['chroma_p50']:.4f}")
    check("G2.chromatic_not_pure_luma",
          max(r["chroma_p95"] for r in rows) > 0.02,
          f"max chroma_p95={max(r['chroma_p95'] for r in rows):.4f}")

    print()
    print("=" * 70)
    print("G3  dark scenes admit more colour ambiguity than bright ones")
    print("=" * 70)
    # Same water, same depth statistics, different scene level.  If "unsupported
    # colour" is really about headroom then the dark scene must admit more.
    levels = []
    for base in (0.10, 0.25, 0.45, 0.65, 0.85):
        Jb = np.clip(rng.uniform(0.0, 1.0, size=(h, w, 3)) * (2 * base), 0.0, 1.0)
        r = G.colour_budget(Jb, d, wm, 0.8)
        levels.append((base, r["dd_pos_max"], r["chroma_p95"]))
        print(f"  level={base:.2f} dd_pos_max={r['dd_pos_max']:.3f} "
              f"dd_neg_min={r['dd_neg_min']:.3f} "
              f"chroma_p95={r['chroma_p95']:.3f}", flush=True)
    check("G3.headroom_monotone",
          all(levels[i][1] >= levels[i + 1][1] - 1e-9 for i in range(len(levels) - 1)),
          "dd_pos_max non-increasing in scene level")

    print()
    print("=" * 70)
    print("G4  a real scene, and the budget it implies")
    print("=" * 70)
    npz = "data/packed/UIEB.npz"
    if os.path.exists(npz):
        dat = np.load(npz, allow_pickle=True)
        budget = []
        for idx in (0, 1, 2):
            J = cv2.resize(dat["gt"][idx], (h, w), interpolation=cv2.INTER_AREA)
            J = J.astype(np.float64) / 255.0
            dd = 0.5 + 0.4 * rng.rand(h, w)
            # Request the deepest admissible step in each direction separately: for a
            # real photograph the `dd > 0` side is empty, so asking for +0.8 would
            # return the zero-width clip and say nothing.
            rp = G.colour_budget(J, dd, wm, 0.0)
            rn = G.colour_budget(J, dd, wm, -0.25)
            budget.append((rp, rn))
            print(f"  {str(dat['names'][idx]):16s} level={J.mean():.3f} "
                  f"minJ={J.min():.3f} maxJ={J.max():.3f} "
                  f"dd_pos_max={rp['dd_pos_max']:+.3f} "
                  f"dd_neg_min={rn['dd_neg_min']:+.3f} "
                  f"chroma@-0.25 p95={rn['chroma_p95']:.3f} "
                  f"luma@-0.25 p95={rn['luma_p95']:.3f}", flush=True)
        # Real photographs touch the top of the range somewhere, which empties the
        # `dd > 0` side of the family.  The `dd < 0` side stays open, and that is
        # the side that matters: a washed-out restoration is exactly as consistent
        # with the observation as a vivid one.
        check("G4.positive_side_closed_on_real_images",
              all(rp["dd_pos_max"] <= 1e-6 for rp, _ in budget),
              "saturated pixels close the brightening side")
        check("G4.negative_side_carries_colour",
              all(rn["admissible"] and rn["chroma_p95"] > 0.01 for _, rn in budget),
              f"min chroma_p95={min(rn['chroma_p95'] for _, rn in budget):.4f}")
        check("G4.lift_exact_on_real_images",
              all(max(rp["lift_err_max"], rn["lift_err_max"]) < 1e-12
                  for rp, rn in budget),
              "gauge algebra holds on real pixels too")
    else:
        check("G4.real_scene_has_budget", False, f"{npz} missing")

    print()
    print("=" * 70)
    print("G5  best_step fits ONE parameter, not three")
    print("=" * 70)
    # ``u_c = exp(beta_c dd)`` links the channels.  Fitting each ``u_c`` freely is
    # a three-parameter model and absorbs far more of a difference than a different
    # water explanation could -- measured at a factor of ~5 on real images before
    # this was caught (0.95 vs 0.20).  The regression is: when the target really is
    # a gauge image, one parameter must recover it exactly; when the target is an
    # unrelated scene, one parameter must do clearly worse than three.
    J0 = rng.rand(h, w, 3) * 0.8 + 0.1
    Jg, _ = G.gauge_pair(J0, np.zeros((h * w,)), wm, np.full(h * w, 0.25))
    r_exact = G.best_step(J0, Jg, wm)
    check("G5.recovers_a_pure_gauge_image",
          r_exact["frac"] > 0.999 and abs(r_exact["dd"] - 0.25) < 5e-3,
          f"frac={r_exact['frac']:.5f} dd={r_exact['dd']:+.4f} (true +0.25)")

    Jx = np.zeros_like(J0)
    b = wm.b_inf.reshape(1, 1, 3)
    # An independent per-channel gain is EXACTLY the three-parameter model.  Two
    # independent uniform scenes would not test this: both are centred on ``b_inf``,
    # so they correlate by construction and either model absorbs a similar share.
    # Giving the free fit a target it can represent perfectly exposes whether the
    # constrained fit is genuinely restricted to one degree of freedom.
    #
    # The scene is kept low-contrast so the test target stays inside [0,1] *without*
    # clipping: clipping would break the very linearity the test is checking, which
    # is how the first version of this test managed to fail for the wrong reason.
    b = wm.b_inf.reshape(1, 1, 3)
    Jlow = b + rng.rand(h, w, 3) * 0.15
    gains = np.array([4.0, 1.0, 2.0]).reshape(1, 1, 3)
    Jx = b + (Jlow - b) * gains
    check("G5.test_target_is_unclipped", Jx.max() <= 1.0 and Jx.min() >= 0.0,
          f"range [{Jx.min():.3f}, {Jx.max():.3f}]")
    r_open = G.best_step(Jlow, Jx, wm)
    check("G5.free_fit_absorbs_a_per_channel_gain",
          r_open["frac_free"] > 0.99,
          f"frac_free={r_open['frac_free']:.4f} on a target the 3-gain model spans")
    check("G5.one_param_cannot_match_three",
          r_open["frac"] < r_open["frac_free"] - 0.10,
          f"constrained={r_open['frac']:.3f} vs free={r_open['frac_free']:.3f}")

    print()
    print("=" * 70)
    print(f"RESULT {'ok' if not BAD else 'bad'}  ok={len(OK)} bad={len(BAD)}")
    if BAD:
        print("  failed: " + ", ".join(BAD))
    return 0 if not BAD else 1


if __name__ == "__main__":
    sys.exit(main())

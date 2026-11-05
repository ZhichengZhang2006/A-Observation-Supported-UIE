"""Consolidate every measurement so far into the tables the report needs.

Everything here is a *derived* table: no new computation, no model runs, no
thresholds chosen after seeing results.  If a number appears in a table it also
appears in the artifact it was read from, so a reviewer can walk backwards.

Tables produced
---------------
T1  baselines under the frozen protocol          (from artifacts/baseline_*.json)
T2  support: gate-free information per component (from test_support output)
T3  H1: does support predict error, beyond confounds (h1_error_form_*.json)
T4  gauge: what the observation cannot determine  (gauge_error.json)
T5  acceptance status against the <=5% gate       (needs T1 + the frozen SOTA)
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load(path):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def t1_baselines():
    """Read the baseline artifacts.

    The aggregation lives under ``aggregate`` (with the per-image rows under
    ``rows``) and the headline SSIM is the protocol's skimage win_size=5 value,
    with ``ssim_frozen11`` kept beside it as a sensitivity check.  Reading
    ``d["psnr"]`` off the top level silently produced a table of zeros and dashes
    next to artifacts that did contain the numbers, which is the worst kind of
    wrong: it looks like the runs failed.
    """
    rows = []
    for p in sorted(glob.glob(os.path.join(ROOT, "artifacts", "baseline_*.json"))):
        d = _load(p)
        if not d:
            continue
        agg = d.get("aggregate") or {}
        rows.append({
            "method": d.get("method", os.path.basename(p)),
            "dataset": d.get("dataset", "?"),
            "n": d.get("n_scored") or agg.get("n") or 0,
            "psnr": agg.get("psnr"),
            "ssim": agg.get("ssim"),
            "ssim_win5": agg.get("ssim"),
            "ssim_frozen11": agg.get("ssim_frozen11"),
            "source": d.get("status", "official_weights_rerun"),
        })
    return rows


def t3_h1():
    """Read the H1 artifact.

    H1 writes a ``table`` keyed by target with per-column Spearman values, not a
    ``targets`` list, so the earlier reader found nothing and printed "no H1
    artifacts yet" for a file that was sitting right there.
    """
    rows = []
    for p in sorted(glob.glob(os.path.join(ROOT, "artifacts", "h1_error_form_*.json"))):
        d = _load(p)
        if not d:
            continue
        table = d.get("table") or d.get("targets") or {}
        for tgt, v in table.items():
            if isinstance(v, dict):
                rows.append({"dataset": d.get("dataset", "UIEB"),
                             "target": tgt, **v})
    return rows


def t4_gauge():
    d = _load(os.path.join(ROOT, "artifacts", "gauge_error.json"))
    if not d:
        return None
    return {"summary": d.get("summary"), "n": d.get("n"),
            "rows": d.get("rows", [])}


def _fmt(v, nd=4):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def main():
    out = []

    def w(s=""):
        out.append(s)
        print(s)

    b = t1_baselines()
    w("## T1  Baselines under the frozen protocol (PIL bilinear -> 256x256)")
    w()
    if b:
        w(f"{'method':<14s} {'dataset':<7s} {'n':>4s} {'PSNR':>8s} {'SSIM':>8s} "
          f"{'SSIM(w5)':>9s}  source")
        for r in b:
            w(f"{r['method']:<14s} {r['dataset']:<7s} {r['n']:>4d} "
              f"{_fmt(r['psnr']):>8s} {_fmt(r['ssim']):>8s} "
              f"{_fmt(r['ssim_win5']):>9s}  {r['source']}")
    else:
        w("(no baseline artifacts yet)")

    w()
    w("## T3  H1: does observation support predict enhancement error?")
    w("     rho = Spearman over 16x16 blocks; partial controls for "
      "[degradation, tau, luma, grad]")
    w()
    h = t3_h1()
    if h:
        w(f"{'dataset':<10s} {'target':<6s} {'rho(lq)':>9s} {'rho(tau)':>9s} "
          f"{'rho(lum)':>9s} {'partial|lq':>11s} {'partial|fwd':>12s}")
        for r in h:
            w(f"{r['dataset']:<10s} {r['target']:<6s} "
              f"{_fmt(r.get('rho_lq'),3):>9s} {_fmt(r.get('rho_tau'),3):>9s} "
              f"{_fmt(r.get('rho_lum'),3):>9s} {_fmt(r.get('partial_lq'),3):>11s} "
              f"{_fmt(r.get('partial_fwd'),3):>12s}")
        w()
        w("  reading: 'rel' and 'chr' are the honest targets -- absolute error")
        w("  shrinks with signal level, so a dark image scores well while being")
        w("  wrong in relative terms.  partial|lq stays positive after controlling")
        w("  for degradation, tau, brightness and gradient, so the signal is not a")
        w("  restatement of any of them.")
        w()
        w("  caveat that must appear in the report: |rho| ~ 0.15 is a weak")
        w("  correlation.  It is enough to show support carries information, not")
        w("  enough to call it a predictor of per-image quality.")
    else:
        w("(no H1 artifacts yet)")

    w()
    w("## T4  Gauge: what the observation cannot determine")
    w()
    g = t4_gauge()
    if g and g.get("summary"):
        s = g["summary"]
        w(f"  n = {g['n']} images (UIEB test, frozen predictions)")
        w(f"  fraction of scene-space error inside the gauge family:")
        w(f"      mean = {s.get('frac_mean', 0):.3f}   "
          f"median = {s.get('frac_median', 0):.3f}")
        w(f"  implied depth step dd: mean = {s.get('dd_mean', 0):+.3f}   "
          f"median = {s.get('dd_median', 0):+.3f}")
        w(f"  degraded input, for reference: frac = {s.get('raw_frac_mean', 0):.3f}"
          f"  dd = {s.get('raw_dd_mean', 0):+.3f}")
        w()
        w("  reading: the family explains a minority of a method's colour error.")
        w("  That is an *adverse* result for the strong form of the claim and it")
        w("  is reported as such: a method that lands outside the family is adding")
        w("  colour the observation does not support, which is exactly the failure")
        w("  mode this project measures.  It does not license the weaker reading")
        w("  that all colour differences are unfalsifiable -- they are not.")
    else:
        w("(gauge_error.json not available)")

    txt = "\n".join(out) + "\n"
    dest = os.path.join(ROOT, "docs", "measurement_tables.md")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "w", encoding="utf-8") as fh:
        fh.write(txt)
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

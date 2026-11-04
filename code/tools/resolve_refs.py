"""Resolve which reference (GT) directory belongs to a directory of predictions.

The frozen SOTA registry must be built from predictions whose references are
*proven*, not assumed.  This tool tries several normalisations of the stem
(`${stem}`, `${stem}_`, `${stem}_img_`, numeric core) against candidate
reference directories and reports, per candidate, how many prediction stems it
resolves.  A candidate that does not resolve ~all stems is rejected.

Usage:
  python tools/resolve_refs.py --out refs.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

IMG = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
D = r"C:\Users\Zhingcheng Zhang\Desktop"


def is_img(p: str) -> bool:
    return p.lower().endswith(IMG) and not os.path.basename(p).startswith(".")


def stems_of(d: str) -> set:
    if not os.path.isdir(d):
        return set()
    return {os.path.splitext(f)[0] for f in os.listdir(d) if is_img(f)}


def norm_variants(s: str) -> list:
    """Every plausible way a prediction stem can encode its reference stem."""
    out = [s]
    for suf in ("_img_", "_img", "_pred", "_enhanced", "_out", "_result"):
        if s.endswith(suf):
            out.append(s[: -len(suf)])
        if suf in s:
            out.append(s.replace(suf, ""))
    m = re.match(r"^(\d+)", s)
    if m:
        out.append(m.group(1))
    m = re.search(r"(\d+)", s)
    if m:
        out.append(m.group(1))
    # dedupe, keep order
    seen, res = set(), []
    for v in out:
        if v and v not in seen:
            seen.add(v)
            res.append(v)
    return res


def resolve(pred_dir: str, ref_dirs: list) -> dict:
    ps = stems_of(pred_dir)
    rep = {"pred_dir": pred_dir, "n_pred": len(ps), "candidates": {}}
    for rd in ref_dirs:
        rs = stems_of(rd)
        if not rs:
            rep["candidates"][rd] = {"n_ref": 0}
            continue
        exact = len(ps & rs)
        loose = 0
        for s in ps:
            if any(v in rs for v in norm_variants(s)):
                loose += 1
        rep["candidates"][rd] = {"n_ref": len(rs), "exact": exact, "loose": loose,
                                 "coverage_exact": round(exact / max(len(ps), 1), 3),
                                 "coverage_loose": round(loose / max(len(ps), 1), 3)}
    return rep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    sfg = os.path.join(D, "fsdownload", "SFGNet-main", "data")
    mc = os.path.join(D, "MambaCue-Result", "ALL", "results")

    ref_candidates = {
        "UIEB": [os.path.join(sfg, "UIEB", "train", "GT"),
                 os.path.join(sfg, "UIEB", "train", "input"),
                 os.path.join(sfg, "UIEB", "test", "GT"),
                 os.path.join(D, "fsdownload", "CDF-UIE-main", "data", "UIEB", "reference"),
                 os.path.join(D, "Physician-for-UIE", "data", "UIEB", "reference")],
        "LSUI": [os.path.join(sfg, "LSUI", "test", "GT"),
                 os.path.join(sfg, "LSUI", "val", "GT"),
                 os.path.join(sfg, "LSUI", "train", "GT")],
        "UFO": [os.path.join(sfg, "UFO", "val", "GT"),
                os.path.join(sfg, "UFO", "val", "input"),
                os.path.join(sfg, "UFO", "train", "GT")],
    }

    # also try to discover extra reference roots: any dir named GT/reference under Desktop
    extra = []
    for root in (os.path.join(D, "fsdownload"), os.path.join(D, "Physician-for-UIE"),
                 os.path.join(D, "New-UIE")):
        if not os.path.isdir(root):
            continue
        for dp, dns, fns in os.walk(root):
            dns[:] = [x for x in dns if not x.startswith(".")
                      and x not in ("__pycache__", ".git", "node_modules")]
            if os.path.basename(dp).lower() in ("gt", "reference", "ref", "ground_truth", "clean"):
                if any(is_img(f) for f in fns):
                    extra.append(dp)
            if dp.count(os.sep) - root.count(os.sep) > 3:
                dns[:] = []

    rep = {"pred_sets": {}, "extra_ref_roots": sorted(set(extra))}
    for ds in ("UIEB", "LSUI", "UFO"):
        pdir = os.path.join(mc, ds)
        if not os.path.isdir(pdir):
            continue
        # pick the largest method dir as the probe
        methods = [m for m in sorted(os.listdir(pdir)) if os.path.isdir(os.path.join(pdir, m))]
        if not methods:
            continue
        probe = max(methods, key=lambda m: len(stems_of(os.path.join(pdir, m))))
        cands = ref_candidates[ds] + sorted(set(extra))
        rep["pred_sets"][ds] = resolve(os.path.join(pdir, probe), cands)
        rep["pred_sets"][ds]["probe_method"] = probe

    txt = json.dumps(rep, indent=1, ensure_ascii=False)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(txt)
        print("wrote", args.out, len(txt), "bytes")
    else:
        print(txt)
    return 0


if __name__ == "__main__":
    sys.exit(main())

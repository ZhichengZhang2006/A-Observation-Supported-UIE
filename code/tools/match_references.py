"""Locate the reference (GT) images that correspond to a set of prediction
stems produced by third-party methods.

Motivation: the audit showed that the UIEB-R90 and UFO-120 prediction sets do
NOT have matching reference folders inside the SFGNet data tree.  Before any
number is reported we must prove, per stem, that a reference image exists, and
from which dataset release it came.  Missing references are reported, not
silently skipped.

Usage:
  python tools/match_references.py --pred <dir-or-dir-of-dirs> --search <root> [--search <root>] --out refs.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

IMG = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


def is_img(p: str) -> bool:
    return p.lower().endswith(IMG) and not os.path.basename(p).startswith(".")


def index_by_stem(roots, max_files=400_000):
    """Map stem -> [full paths] over every image below the given roots."""
    idx = defaultdict(list)
    n = 0
    for root in roots:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames
                           if not d.startswith(".") and d not in
                           ("__pycache__", "node_modules", ".git")]
            for f in filenames:
                if not is_img(f):
                    continue
                idx[os.path.splitext(f)[0]].append(os.path.join(dirpath, f))
                n += 1
                if n >= max_files:
                    return idx, n, True
    return idx, n, False


def pred_stems(pred):
    if os.path.isdir(pred):
        return {os.path.splitext(f)[0]: os.path.join(pred, f)
                for f in sorted(os.listdir(pred)) if is_img(f)}
    raise SystemExit("not a dir: " + pred)


def score_candidate(pred_path, cand_path):
    """Prefer same basename; then prefer paths whose parents look like GT/ref."""
    s = 0
    if os.path.basename(pred_path) == os.path.basename(cand_path):
        s += 2
    low = cand_path.lower()
    for tag in ("\\gt\\", "/gt/", "reference", "ref\\", "/ref/", "ground"):
        if tag in low:
            s += 3
    for tag in ("input", "raw", "degraded", "pred", "result", "output"):
        if tag in low:
            s -= 2
    return s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--search", action="append", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--exclude", action="append", default=[])
    args = ap.parse_args()

    stems = pred_stems(args.pred)
    print("pred stems:", len(stems))
    idx, n_scanned, truncated = index_by_stem(args.search)
    print("indexed images:", n_scanned, "stems:", len(idx), "truncated:", truncated)

    rep = {"pred": args.pred, "n_pred": len(stems),
           "search": args.search, "indexed_images": n_scanned,
           "index_truncated": truncated, "matches": {}, "missing": []}

    groups = defaultdict(int)
    for st, pp in sorted(stems.items()):
        cands = idx.get(st, [])
        if args.exclude:
            cands = [c for c in cands
                     if not any(c.lower().startswith(e.lower()) for e in args.exclude)]
        if not cands:
            rep["missing"].append(st)
            continue
        cands = sorted(cands, key=lambda c: -score_candidate(pp, c))
        best = cands[0]
        rep["matches"][st] = {"ref": best, "score": score_candidate(pp, best),
                              "n_candidates": len(cands)}
        # group key = grandparent + parent, so we can see which folder the refs live in
        key = os.path.dirname(best)
        groups[key] += 1

    rep["match_groups"] = dict(sorted(groups.items(), key=lambda kv: -kv[1])[:40])
    rep["n_matched"] = len(rep["matches"])
    rep["n_missing"] = len(rep["missing"])
    txt = json.dumps(rep, indent=1, ensure_ascii=False)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(txt)
        print("wrote", args.out, len(txt), "bytes")
    else:
        print(txt)
    print("matched=%d missing=%d" % (rep["n_matched"], rep["n_missing"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())

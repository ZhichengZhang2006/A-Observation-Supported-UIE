"""Audit local underwater-UIE assets: splits, pairing, duplicates, and the
result directories produced by third-party methods.

This is a *measurement* script: it reports what is on disk and refuses to infer
a split from a directory name alone.  Output is JSON so it can be diffed and
hashed into the data manifest.

Run:  python tools/audit_assets.py --root <Desktop> --out <json>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict

IMG = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


def is_img(p: str) -> bool:
    return p.lower().endswith(IMG) and not p.startswith(".")


def walk(root: str):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        imgs = [f for f in filenames if is_img(f)]
        if imgs:
            yield dirpath, imgs


def sha1_of(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dhash(path: str, size: int = 8):
    """Perceptual (difference) hash. Returns None when PIL/cv2 is unavailable."""
    try:
        import cv2
        import numpy as np
    except Exception:
        return None
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, (size + 1, size), interpolation=cv2.INTER_AREA)
    diff = img[:, 1:] > img[:, :-1]
    bits = 0
    for b in diff.flatten():
        bits = (bits << 1) | int(b)
    return format(bits, "016x")


def hamming(a: str, b: str) -> int:
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def describe_tree(root: str, max_files: int = 6) -> dict:
    out = {"root": root, "dirs": {}}
    total = 0
    for d, files in walk(root):
        rel = os.path.relpath(d, root)
        total += len(files)
        out["dirs"][rel] = {"n": len(files), "sample": sorted(files)[:max_files]}
    out["total_images"] = total
    return out


def list_names(root: str) -> dict:
    """All image basenames under root, grouped by parent dir."""
    g = defaultdict(list)
    for d, files in walk(root):
        for f in files:
            g[os.path.relpath(d, root)].append(f)
    return {k: sorted(v) for k, v in sorted(g.items())}


def stem_set(root: str) -> set:
    s = set()
    for _, files in walk(root):
        for f in files:
            s.add(os.path.splitext(f)[0])
    return s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"C:\Users\Zhingcheng Zhang\Desktop")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    D = args.root
    rep = {"root": D}

    targets = {
        "sfg_data": os.path.join(D, "fsdownload", "SFGNet-main", "data"),
        "mc_results": os.path.join(D, "MambaCue-Result", "ALL", "results"),
        "cdf_results": os.path.join(D, "fsdownload", "CDF-UIE-main", "results"),
        "phase_results": os.path.join(D, "fsdownload", "Phaseformer-main", "results"),
    }

    # ---------- 1. dataset trees ----------
    rep["trees"] = {}
    for k in ("sfg_data",):
        p = targets[k]
        if os.path.isdir(p):
            rep["trees"][k] = {ds: describe_tree(os.path.join(p, ds))
                               for ds in sorted(os.listdir(p))
                               if os.path.isdir(os.path.join(p, ds))}

    # ---------- 2. result dirs: which methods, how many, which stems ----------
    rep["results"] = {}
    for k in ("mc_results", "cdf_results", "phase_results"):
        p = targets[k]
        if not os.path.isdir(p):
            continue
        node = {}
        for ds in sorted(os.listdir(p)):
            dsp = os.path.join(p, ds)
            if not os.path.isdir(dsp):
                continue
            node[ds] = {}
            for m in sorted(os.listdir(dsp)):
                mp = os.path.join(dsp, m)
                if not os.path.isdir(mp):
                    continue
                names = list_names(mp)
                allf = [f for v in names.values() for f in v]
                node[ds][m] = {"n": len(allf), "subdirs": {kk: len(vv) for kk, vv in names.items()},
                               "sample": sorted(allf)[:4]}
        rep["results"][k] = node

    # ---------- 3. which dataset images do the result stems refer to? ----------
    sfg = targets["sfg_data"]
    rep["split_resolution"] = {}
    if os.path.isdir(sfg):
        for ds in ("UIEB", "LSUI", "UFO"):
            dsp = os.path.join(sfg, ds)
            if not os.path.isdir(dsp):
                continue
            ref = {"all_stems": stem_set(dsp)}
            for sub in ("train", "val", "test"):
                sp = os.path.join(dsp, sub)
                if os.path.isdir(sp):
                    ref[sub] = stem_set(sp)
            rep["split_resolution"][ds] = {k: len(v) for k, v in ref.items()}

            # cross-check against MambaCue-Result stems
            res_root = targets["mc_results"]
            if os.path.isdir(res_root):
                rp = None
                for cand in ("UIEB", "LSUI", "UFO"):
                    c = os.path.join(res_root, cand)
                    if cand == ds and os.path.isdir(c):
                        rp = c
                        break
                if rp:
                    m = sorted(os.listdir(rp))[0]
                    mpath = os.path.join(rp, m)
                    rstem = stem_set(mpath)
                    rep["split_resolution"][ds]["result_stems"] = len(rstem)
                    for sub in ("train", "val", "test"):
                        if sub in ref:
                            inter = rstem & ref[sub]
                            rep["split_resolution"][ds][f"result_in_{sub}"] = len(inter)
                    rep["split_resolution"][ds]["result_not_in_any_sub"] = len(
                        rstem - ref.get("all_stems", set()))

    # ---------- 4. duplicate / leakage screening ----------
    rep["duplicates"] = {}
    leak_targets = [
        ("UIEB", os.path.join(sfg, "UIEB", "train", "GT"), os.path.join(sfg, "UIEB", "val", "GT")),
        ("LSUI", os.path.join(sfg, "LSUI", "train", "GT"), os.path.join(sfg, "LSUI", "val", "GT")),
        ("LSUI", os.path.join(sfg, "LSUI", "train", "GT"), os.path.join(sfg, "LSUI", "test", "GT")),
        ("UFO", os.path.join(sfg, "UFO", "train", "GT"), os.path.join(sfg, "UFO", "val", "GT")),
    ]
    for ds, a, b in leak_targets:
        if not (os.path.isdir(a) and os.path.isdir(b)):
            continue
        af = [os.path.join(a, f) for f in sorted(os.listdir(a)) if is_img(f)]
        bf = [os.path.join(b, f) for f in sorted(os.listdir(b)) if is_img(f)]
        key = f"{ds}:{os.path.basename(os.path.dirname(a))}_vs_{os.path.basename(os.path.dirname(b))}"
        # exact-content duplicates via sha1 over a sample-limited scan
        ha = {}
        for f in af:
            ha.setdefault(sha1_of(f), []).append(f)
        dup = []
        for f in bf:
            h = sha1_of(f)
            if h in ha:
                dup.append({"test_file": os.path.basename(f),
                            "train_file": [os.path.basename(x) for x in ha[h]]})
        rep["duplicates"][key] = {"n_a": len(af), "n_b": len(bf), "exact_dup": len(dup),
                                  "examples": dup[:5]}

        # perceptual near-dup on a bounded sample
        if af and bf:
            sample_a = [dhash(f) for f in af[:400]]
            sample_a = [x for x in sample_a if x]
            near = 0
            ex = []
            for f in bf[:200]:
                hb = dhash(f)
                if not hb:
                    continue
                for i, x in enumerate(sample_a):
                    if hamming(x, hb) <= 4:
                        near += 1
                        if len(ex) < 5:
                            ex.append({"test_file": os.path.basename(f),
                                       "train_file": os.path.basename(af[i])})
                        break
            rep["duplicates"][key]["phash_scanned"] = [len(sample_a), min(len(bf), 200)]
            rep["duplicates"][key]["phash_near_dup"] = near
            rep["duplicates"][key]["phash_examples"] = ex

    txt = json.dumps(rep, indent=1, ensure_ascii=False)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(txt)
        print("wrote", args.out, len(txt), "bytes")
    else:
        print(txt)
    return 0


if __name__ == "__main__":
    sys.exit(main())

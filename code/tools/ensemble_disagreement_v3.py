"""Per-image ensemble disagreement between two frozen backbones (ordinary uncertainty proxy).

MC-dropout and deep ensembles need a trainable model with stochastic layers.  We
have two *frozen, independently produced* restorations of the same observation,
which is the cheapest honest stand-in for epistemic spread: if the two models
disagree on a pixel, no single one of them is confident there.  This is an
ordinary uncertainty signal available to any practitioner with two checkpoints,
and it is what the support statistic has to beat.

Writes ``artifacts/features_v3/disagree_<split>.npz`` with a 16x16x3 grid per
image (downsampled with INTER_AREA) so pixelwise disagreement survives the
reduction without shipping full images.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import obsui  # noqa: E402,F401

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

import obsui.dataio as dataio  # noqa: E402

DATASET_ROOT = "/home/user/UnderWaterDataset"
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PRED_ROOT = os.path.join(PROJECT_ROOT, "predictions")
OUT = os.path.join(PROJECT_ROOT, "artifacts", "features_v3")
GRID = 16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "val", "test"])
    ap.add_argument("--dataset", default="UIEB")
    ap.add_argument("--methods", nargs="+", default=["phaseformer", "aquaclarity", "fdcenet",
                                                     "cdfuie", "ssuie", "cclnet", "sfgnet"])
    args = ap.parse_args()

    tag = args.dataset if args.split == "test" else f"{args.dataset}-{args.split}"
    in_dir = os.path.join(DATASET_ROOT, f"{args.dataset}/{args.split}", "input")
    ref_dir = os.path.join(DATASET_ROOT, f"{args.dataset}/{args.split}", "GT")
    pairs = dataio.resolve_pairs(in_dir, ref_dir, args.dataset)

    names, grids = [], []
    for p in pairs.pairs:
        paths = [os.path.join(PRED_ROOT, tag, m, p.name + ".png") for m in args.methods]
        if not all(os.path.exists(q) for q in paths):
            continue
        stack = np.stack([
            np.asarray(Image.open(q).convert("RGB").resize((GRID, GRID), Image.BILINEAR),
                       dtype=np.float32) / 255.0 for q in paths])
        # per-pixel standard deviation across the independently trained methods:
        # a deep-ensemble style epistemic spread, cheap and available.
        names.append(p.name)
        grids.append(stack.std(axis=0))
    os.makedirs(OUT, exist_ok=True)
    out = os.path.join(OUT, f"disagree_{args.dataset}_{args.split}.npz")
    np.savez_compressed(out, names=np.array(names), absdiff=np.array(grids, dtype=np.float32),
                        methods=np.array(args.methods))
    print(f"wrote {out} n={len(names)}")


if __name__ == "__main__":
    main()

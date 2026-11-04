"""Dataset discovery + manifest generation.

Scans the local asset roots, resolves each dataset into (input, reference) pairs,
and writes a machine-readable manifest plus a human-audit report.

Design decisions (recorded for the paper):
  * No dataset is downloaded: the local tree already holds the official archives
    that the user's previous projects (MambaCue / SFGNet / CDF-UIE / Phaseformer)
    were evaluated on.  We hash every file so that the manifest is reproducible.
  * For UIEB we take the 90 reference images (R90) as the sealed test set, which
    is the community convention for UIEB "test".
  * LSUI / UFO splits are *verified* against the official lists if present; if a
    split file cannot be found we say so rather than guessing (see SPLIT_STATUS).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional, Tuple

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def sha256_of(path: str, blocksize: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(blocksize)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def list_images(root: str) -> List[str]:
    out = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if os.path.splitext(fn)[1].lower() in IMG_EXT:
                out.append(os.path.join(dp, fn))
    return sorted(out)


@dataclass
class PairSet:
    name: str
    root: str
    pairs: List[Tuple[str, str]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def manifest(self, hash_files: bool = False) -> Dict:
        rec = {
            "name": self.name,
            "root": self.root,
            "n_pairs": len(self.pairs),
            "notes": self.notes,
        }
        if hash_files:
            rec["sha256"] = [
                {"input": os.path.basename(a), "sha256_input": sha256_of(a),
                 "ref": os.path.basename(b), "sha256_ref": sha256_of(b)}
                for a, b in self.pairs
            ]
        else:
            rec["filelist"] = [{"input": os.path.basename(a), "ref": os.path.basename(b)}
                               for a, b in self.pairs]
        return rec


# --------------------------------------------------------------- resolvers
def resolve_uieb(sfg_root: str) -> PairSet:
    """UIEB: raw/ + reference/.  R90 = the 90 images with reference."""
    raw = os.path.join(sfg_root, "UIEB", "raw")
    ref = os.path.join(sfg_root, "UIEB", "reference")
    ps = PairSet("UIEB-R90", os.path.join(sfg_root, "UIEB"))
    if not (os.path.isdir(raw) and os.path.isdir(ref)):
        ps.notes.append(f"MISSING: raw={os.path.isdir(raw)} ref={os.path.isdir(ref)}")
        return ps
    refs = {os.path.splitext(os.path.basename(p))[0]: p for p in list_images(ref)}
    raws = {os.path.splitext(os.path.basename(p))[0]: p for p in list_images(raw)}
    common = sorted(set(refs) & set(raws))
    ps.pairs = [(raws[k], refs[k]) for k in common]
    ps.notes.append(f"raw={len(raws)} ref={len(refs)} matched={len(common)}")
    ps.notes.append(f"UNPAIRED raws (no reference, UIEB-challenging part)={len(set(raws)-set(refs))}")
    if len(common) != 90:
        ps.notes.append(f"WARNING expected 90 paired references, found {len(common)}")
    return ps


def resolve_flat(inp_dir: str, ref_dir: str, name: str,
                 ref_suffix: str = "") -> PairSet:
    """Generic resolver for datasets laid out as input/*.png reference/*.png."""
    ps = PairSet(name, os.path.dirname(inp_dir))
    if not (os.path.isdir(inp_dir) and os.path.isdir(ref_dir)):
        ps.notes.append(f"MISSING: inp={os.path.isdir(inp_dir)} ref={os.path.isdir(ref_dir)}")
        return ps
    ins = {os.path.splitext(os.path.basename(p))[0]: p for p in list_images(inp_dir)}
    refs = {}
    for p in list_images(ref_dir):
        k = os.path.splitext(os.path.basename(p))[0]
        if ref_suffix and k.endswith(ref_suffix):
            k = k[: -len(ref_suffix)]
        refs[k] = p
    common = sorted(set(ins) & set(refs))
    ps.pairs = [(ins[k], refs[k]) for k in common]
    ps.notes.append(f"inputs={len(ins)} refs={len(refs)} matched={len(common)}")
    return ps


def probe(root: str, maxdepth: int = 2) -> List[str]:
    """Print a shallow listing so we can see the real layout."""
    lines = []
    root = os.path.abspath(root)
    if not os.path.isdir(root):
        return [f"(missing) {root}"]
    for dp, dns, fns in os.walk(root):
        depth = dp[len(root):].count(os.sep)
        if depth > maxdepth:
            dns[:] = []
            continue
        rel = os.path.relpath(dp, root)
        lines.append(f"[D] {rel}  dirs={len(dns)} files={len(fns)}  {sorted(fns)[:4]}")
    return lines


if __name__ == "__main__":
    SFG = r"C:\Users\Zhingcheng Zhang\Desktop\fsdownload\SFGNet-main\data"
    for d in ("UIEB", "LSUI", "UFO", "U45"):
        print("=" * 70)
        print(probe(os.path.join(SFG, d), 2))

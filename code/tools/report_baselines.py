"""Print the unified-protocol baseline table from ``artifacts/baseline_*.json``.

Kept as a file rather than an inline ``python -c`` because nested quoting through
ssh silently mangled the expression and returned an empty exit code 1 -- an error
that looks like "no results" but is actually "your shell ate the program".
"""

from __future__ import annotations

import argparse
import glob
import json
import os


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="artifacts/baseline_*.json")
    ap.add_argument("--datasets", default="")
    args = ap.parse_args()

    rows = []
    for path in sorted(glob.glob(args.glob)):
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)
        ds = d.get("dataset") or os.path.basename(path).split("_")[-1].replace(".json", "")
        if args.datasets and ds not in args.datasets.split(","):
            continue
        s = d.get("summary") or d.get("aggregate") or {}
        rows.append({
            "method": d.get("method", "?"),
            "dataset": ds,
            "n": d.get("n_scored", 0),
            "psnr": s.get("psnr"),
            "ssim": s.get("ssim"),
            "ssim11": s.get("ssim_frozen11"),
            "ckpt": (d.get("checkpoint") or d.get("ckpt") or "")[-58:],
            "err": d.get("errors") or d.get("error") or "",
        })

    for ds in sorted({r["dataset"] for r in rows}):
        sub = [r for r in rows if r["dataset"] == ds and r["n"]]
        if not sub:
            print(f"\n== {ds}: no completed runs ==")
            for r in rows:
                if r["dataset"] == ds:
                    print(f"  {r['method']:<14s} FAILED  {r['err'][:90]}")
            continue
        sub.sort(key=lambda r: -(r["psnr"] or 0))
        print(f"\n== {ds} (n={sub[0]['n']}) ==")
        print(f"{'method':<14s} {'PSNR':>8s} {'SSIM5':>8s} {'SSIM11':>8s}  ckpt")
        for r in sub:
            print(f"{r['method']:<14s} {r['psnr']:>8.3f} {r['ssim']:>8.4f} "
                  f"{(r['ssim11'] or 0):>8.4f}  {r['ckpt']}")
        failed = [r for r in rows if r["dataset"] == ds and not r["n"]]
        for r in failed:
            print(f"{r['method']:<14s} FAILED  {r['err'][:80]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

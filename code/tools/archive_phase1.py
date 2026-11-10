"""Freeze every raw number of the E0-E3 sweep into the repository.

The point of this script is that the repository keeps the *original* records,
not a curated summary: for every (arm, dataset) cell it copies the run's
``result.json`` byte-for-byte, keeps the training log, and records SHA-256 of
both.  Anything derived (means, deltas, CIs) lives in the report and can be
recomputed from these files alone.

Run it on the GPU host, then commit the output:

    PYTHONPATH=/home/user/obsui/code python code/tools/archive_phase1.py \\
        --runs /home/user/obsui/runs/ablation_phase1 \\
        --dest artifacts/phase1_raw

Layout written
--------------
artifacts/phase1_raw/
    <tag>/result.json     unmodified copy of the run's own record
    <tag>/train.log       unmodified training log
    manifest.json         tag -> {paths, sha256, bytes, gpu_seconds, env}
    INDEX.md              human-readable inventory with the headline numbers
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from typing import Dict, List, Optional


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_run(run_dir: str) -> Optional[dict]:
    p = os.path.join(run_dir, "result.json")
    if not os.path.isfile(p):
        return None
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def main() -> int:
    ap = argparse.ArgumentParser(description="Archive raw E0-E3 run records")
    ap.add_argument("--runs", default="/home/user/obsui/runs/ablation_phase1")
    ap.add_argument("--dest", required=True)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.dest, exist_ok=True)
    manifest: Dict[str, dict] = {}
    skipped: List[str] = []

    for tag in sorted(os.listdir(args.runs)):
        run_dir = os.path.join(args.runs, tag)
        if not os.path.isdir(run_dir):
            continue
        r = load_run(run_dir)
        if r is None:
            skipped.append(f"{tag} (no result.json)")
            continue

        out_dir = os.path.join(args.dest, tag)
        os.makedirs(out_dir, exist_ok=True)
        entry: Dict[str, object] = {
            "arm": r["arm"], "dataset": r["dataset"], "updates": r["updates"],
            "seed": r.get("seed"), "files": {},
        }

        for name in ("result.json", "train.log"):
            src = os.path.join(run_dir, name)
            if not os.path.isfile(src):
                continue
            dst = os.path.join(out_dir, name)
            if os.path.isfile(dst) and not args.force:
                pass  # keep the earliest copy; raw records are never overwritten
            else:
                shutil.copy2(src, dst)
            entry["files"][name] = {
                "sha256": sha256_of(dst),
                "bytes": os.path.getsize(dst),
            }

        entry.update({
            "n_backbone_params": r["n_backbone_params"],
            "n_new_params": r["n_new_params"],
            "e1_blocks": r.get("e1_blocks"),
            "sec_per_update": r["sec_per_update"],
            "sec_total": r["sec_total"],
            "peak_mem_gb": r.get("peak_mem_gb"),
            "start": r.get("start"),
            "start_test": r.get("start_test"),
            "endpoints": {
                k: {"update": v["update"], "val": v["val"], "test": v["test"]}
                for k, v in (r.get("endpoints") or {}).items()
            },
            "curve": r["curve"],
            "env": r.get("env"),
            "start_weights": r.get("start_weights"),
            "hparams": {
                k: r.get(k) for k in
                ("batch", "accum", "effective_batch", "lr_backbone", "lr_new",
                 "lr_min", "warmup_updates", "grad_clip", "eval_every")
            },
        })
        manifest[tag] = entry

    with open(os.path.join(args.dest, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1, sort_keys=True)

    # ---- INDEX.md ------------------------------------------------------- #
    lines = [
        "# Phase-1 raw run records (E0-E3)",
        "",
        "Every number below is copied unmodified from the run that produced it.",
        "`result.json` holds the full per-image val/test rows, the whole validation",
        "curve, and the branch/gate diagnostics; `train.log` is the run's own stdout.",
        "Derived statistics live in the report and can be recomputed from these files.",
        "",
        "| tag | arm | dataset | updates | val PSNR | val SSIM | test PSNR | test SSIM | GPU h | new params |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for tag, e in sorted(manifest.items()):
        eps = e["endpoints"] or {}
        end = eps.get("final") or eps.get("best") or {}
        v, t = end.get("val", {}), end.get("test", {})
        lines.append(
            f"| {tag} | {e['arm'].upper()} | {e['dataset']} | {e['updates']} | "
            f"{v.get('psnr', float('nan')):.4f} | {v.get('ssim', float('nan')):.6f} | "
            f"{t.get('psnr', float('nan')):.4f} | {t.get('ssim', float('nan')):.6f} | "
            f"{e['sec_total'] / 3600:.2f} | {e['n_new_params']} |")
    lines += ["", "## Untouched start point (published weights)", "",
              "| tag | start val PSNR | start val SSIM | start test PSNR | start test SSIM |",
              "|---|---:|---:|---:|---:|"]
    for tag, e in sorted(manifest.items()):
        s, st = e.get("start") or {}, e.get("start_test") or {}
        lines.append(
            f"| {tag} | {s.get('psnr', float('nan')):.4f} | {s.get('ssim', float('nan')):.6f} | "
            f"{st.get('psnr', float('nan')):.4f} | {st.get('ssim', float('nan')):.6f} |")
    with open(os.path.join(args.dest, "INDEX.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    print(f"archived {len(manifest)} runs to {args.dest}")
    if skipped:
        print("skipped:", *skipped, sep="\n  ")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

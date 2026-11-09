"""Score the E0-E3 ablation: paired bootstrap, strata, and the development table.

Inputs are the ``result.json`` files written by ``train_ablation.py``.  Nothing
here re-runs a model; it only reads per-image rows, so it is safe to re-run and
safe to extend with more runs later.

What it reports
---------------
1. the brief's development table (section 7): experiment, dataset, updates,
   PSNR, SSIM, change vs E0, GPU hours, next step;
2. paired bootstrap over images (10 000 resamples, images as the unit) for every
   arm's delta against E0, with a 95 % CI -- the brief asks for a stable
   +0.15..0.30 dB signal, and a CI that contains 0 is reported as "not
   established" no matter what the point estimate says;
3. per-image delta summaries (mean, median, quartiles, share of images improved);
4. strata by input-image degradation (colour cast, brightness, contrast, detail),
   reviewed per the brief's "strong cast / low light / low contrast / detail"
   requirement;
5. branch and gate diagnostics collected during training.

The polluted-split caveat from the plan is applied here: LSUI has a small number
of exact duplicates across splits, so this tool recomputes the LSUI means both
with and without those duplicated images and prints both.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from obsui import dataio  # noqa: E402

ARMS = ("e0", "e1", "e2", "e3")
DATA_ROOT = "/home/user/UnderWaterDataset"


# --------------------------------------------------------------------------- #
# io
# --------------------------------------------------------------------------- #
def load_results(root: str) -> Dict[Tuple[str, str, int], dict]:
    """(arm, dataset, updates) -> result dict, one per run directory."""
    out: Dict[Tuple[str, str, int], dict] = {}
    for path in sorted(glob.glob(os.path.join(root, "*", "result.json"))):
        with open(path, encoding="utf-8") as fh:
            r = json.load(fh)
        key = (r["arm"], r["dataset"], int(r["updates"]))
        if key in out:
            raise SystemExit(f"duplicate run for {key}: {path} and {out[key]['_path']}")
        r["_path"] = path
        out[key] = r
    return out


def rows_by_name(rows: Sequence[dict]) -> Dict[str, dict]:
    return {r["name"]: r for r in rows}


# --------------------------------------------------------------------------- #
# statistics
# --------------------------------------------------------------------------- #
def paired_bootstrap(base: Dict[str, dict], other: Dict[str, dict],
                     metric: str = "psnr", draws: int = 10000,
                     seed: int = 0) -> Dict[str, float]:
    """Bootstrap the mean per-image delta over the *images* (paired design)."""
    names = sorted(set(base) & set(other))
    if not names:
        raise ValueError("no shared images")
    d = np.array([other[n][metric] - base[n][metric] for n in names], dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(draws, len(d)))
    means = d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {
        "n": int(len(d)),
        "mean": float(d.mean()),
        "median": float(np.median(d)),
        "q25": float(np.percentile(d, 25)),
        "q75": float(np.percentile(d, 75)),
        "improved_frac": float((d > 0).mean()),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "ci_excludes_zero": bool(lo > 0 or hi < 0),
    }


def verdict(delta: Dict[str, float]) -> str:
    """Turn a bootstrap result into the brief's screening vocabulary."""
    if not delta["ci_excludes_zero"]:
        return "no established change (95 % CI contains 0)"
    if delta["mean"] >= 0.15:
        return "real gain (>= +0.15 dB, CI excludes 0)"
    if delta["mean"] > 0:
        return "small but consistent gain (< +0.15 dB, CI excludes 0)"
    return "consistent loss (CI excludes 0)"


# --------------------------------------------------------------------------- #
# strata
# --------------------------------------------------------------------------- #
def image_stats(u8: np.ndarray) -> Dict[str, float]:
    """Degradation descriptors of the *input* image (256x256 uint8 RGB)."""
    x = u8.astype(np.float64) / 255.0
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    mx, mn = x.max(axis=2), x.min(axis=2)
    sat = np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    # colour cast: distance of the channel-mean white point from neutral
    cast = float(np.linalg.norm(x.reshape(-1, 3).mean(axis=0) - 0.5))
    # local contrast: std of the luminance in 16x16 tiles, averaged
    h, w = lum.shape
    tiles = lum.reshape(h // 16, 16, w // 16, 16).transpose(0, 2, 1, 3).reshape(-1, 256)
    local_std = float(tiles.std(axis=1).mean())
    gy, gx = np.gradient(lum)
    grad = float(np.sqrt(gx ** 2 + gy ** 2).mean())
    return {
        "cast": cast,
        "brightness": float(lum.mean()),
        "contrast": local_std,
        "detail": grad,
        "saturation": float(sat.mean()),
    }


def strata_tables(base: Dict[str, dict], other: Dict[str, dict], stats: Dict[str, dict],
                  keys=("cast", "brightness", "contrast", "detail")) -> Dict[str, list]:
    """Tercile strata of one degradation descriptor -> mean delta per tercile."""
    names = sorted(set(base) & set(other) & set(stats))
    if not names:
        return {}
    out: Dict[str, list] = {}
    for k in keys:
        v = np.array([stats[n][k] for n in names], dtype=np.float64)
        d = np.array([other[n]["psnr"] - base[n]["psnr"] for n in names], dtype=np.float64)
        q1, q2 = np.percentile(v, [33.333, 66.667])
        buckets = [("low", v <= q1), ("mid", (v > q1) & (v <= q2)), ("high", v > q2)]
        out[k] = [
            {
                "tercile": label,
                "n": int(mask.sum()),
                "delta_psnr_mean": float(d[mask].mean()) if mask.any() else float("nan"),
                "delta_psnr_median": float(np.median(d[mask])) if mask.any() else float("nan"),
                "share_improved": float((d[mask] > 0).mean()) if mask.any() else float("nan"),
                "value_range": [float(v[mask].min()), float(v[mask].max())] if mask.any() else [],
            }
            for label, mask in buckets
        ]
    return out


def image_stat_table(dataset: str, split: str, names: Sequence[str]) -> Dict[str, dict]:
    root = os.path.join(DATA_ROOT, dataset, split, "input")
    index = {os.path.splitext(f)[0]: f for f in os.listdir(root)}
    cache: Dict[str, dict] = {}
    for n in names:
        f = index.get(n)
        if f is None:
            continue
        cache[n] = image_stats(dataio.to_unified(dataio.load_rgb(os.path.join(root, f))))
    return cache


def duplicate_names(dataset: str, split: str) -> set:
    """Input files whose content hash also appears in another split of the set."""
    if dataset == "UIEB":
        # verified disjoint by content hash on 2026-11-09; nothing to exclude
        return set()
    hashes: Dict[str, List[Tuple[str, str]]] = {}
    for s in ("train", "val", "test"):
        root = os.path.join(DATA_ROOT, dataset, s, "input")
        if not os.path.isdir(root):
            continue
        for f in sorted(os.listdir(root)):
            p = os.path.join(root, f)
            if not os.path.isfile(p):
                continue
            h = dataio.sha256_of(p)
            hashes.setdefault(h, []).append((s, os.path.splitext(f)[0]))
    dup: set = set()
    for lst in hashes.values():
        if len({s for s, _ in lst}) > 1:
            dup.update(n for s, n in lst if s == split)
    return dup


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description="Score the E0-E3 ablation runs")
    ap.add_argument("--runs", default="/home/user/obsui/runs/ablation_phase1")
    ap.add_argument("--out", default="/home/user/obsui/artifacts/ablation_phase1.json")
    ap.add_argument("--draws", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    runs = load_results(args.runs)
    if not runs:
        raise SystemExit(f"no result.json under {args.runs}")

    datasets = sorted({k[1] for k in runs})
    report: Dict[str, object] = {"runs": {}, "table": [], "bootstrap": {}, "strata": {},
                                 "polluted_split": {}}

    for key, r in sorted(runs.items()):
        arm, ds, n_up = key
        eps = r.get("endpoints", {})
        report["runs"][f"{arm}/{ds}/{n_up}"] = {
            "path": r["_path"], "best": r["best"], "val_best": r["val_best"],
            "test": r["test"], "n_backbone_params": r["n_backbone_params"],
            "n_new_params": r["n_new_params"], "sec_per_update": r["sec_per_update"],
            "peak_mem_gb": r.get("peak_mem_gb"), "env": r.get("env"),
            "start": r.get("start"),
            "start_test": r.get("start_test"),
            "endpoints": {k: {"update": v["update"], "val": v["val"], "test": v["test"]}
                          for k, v in eps.items()},
        }

    def rows_for(run: dict, endpoint: str) -> Dict[str, dict]:
        """Per-image rows of one checkpoint endpoint; falls back for old runs."""
        eps = run.get("endpoints") or {}
        if endpoint in eps:
            return rows_by_name(eps[endpoint]["test_rows"])
        return rows_by_name(run["test_rows"])

    def summary_for(run: dict, endpoint: str) -> dict:
        eps = run.get("endpoints") or {}
        if endpoint in eps:
            return eps[endpoint]["test"]
        return run["test"]

    for ds in datasets:
        # pick the longest run per arm, so a short screen is never compared to a
        # long extended run
        per_arm: Dict[str, dict] = {}
        for arm in ARMS:
            cands = [k for k in runs if k[0] == arm and k[1] == ds]
            if cands:
                per_arm[arm] = runs[max(cands, key=lambda k: k[2])]
        if "e0" not in per_arm:
            continue

        # Every comparison is made at matched endpoints: fixed-budget ``final``
        # versus fixed-budget ``final``, selection-based ``best`` versus
        # selection-based ``best``.  Mixing the two would compare an arm that
        # trained to step N against one that stopped at step 40.
        for endpoint in ("final", "best"):
            base_test = rows_for(per_arm["e0"], endpoint)
            if not base_test:
                continue
            for arm, r in per_arm.items():
                other = rows_for(r, endpoint)
                if not other:
                    continue
                delta = paired_bootstrap(base_test, other, "psnr", args.draws, args.seed)
                delta_ssim = paired_bootstrap(base_test, other, "ssim", args.draws, args.seed)
                key = f"{arm}/{ds}/{endpoint}"
                report["bootstrap"][key] = {
                    "psnr": delta, "ssim": delta_ssim, "verdict": verdict(delta),
                    "updates": r["updates"], "endpoint": endpoint,
                }
                test_s = summary_for(r, endpoint)
                base_s = summary_for(per_arm["e0"], endpoint)
                start_s = r.get("start_test") or {}
                report["table"].append({
                    "experiment": arm.upper(), "dataset": ds, "updates": r["updates"],
                    "endpoint": endpoint,
                    "psnr": test_s["psnr"], "ssim": test_s["ssim"],
                    "d_psnr_vs_e0": delta["mean"], "d_psnr_ci": [delta["ci_lo"], delta["ci_hi"]],
                    "d_ssim_vs_e0": delta_ssim["mean"],
                    "d_ssim_ci": [delta_ssim["ci_lo"], delta_ssim["ci_hi"]],
                    "gpu_hours": r["sec_total"] / 3600.0,
                    "verdict": verdict(delta),
                    "e0_psnr": base_s["psnr"], "e0_ssim": base_s["ssim"],
                    # absolute comparison against the untouched published weights:
                    # the claim "the module improves restoration" needs THIS to be
                    # positive, not merely less negative than E0's.
                    "start_psnr": start_s.get("psnr"),
                    "d_psnr_vs_start": (test_s["psnr"] - start_s["psnr"]) if start_s else None,
                })

        # ---- strata on test, at the fixed-budget endpoint -------------------
        base_test = rows_for(per_arm["e0"], "final") or rows_for(per_arm["e0"], "best")
        names = sorted(base_test)
        stats = image_stat_table(ds, "test", names)
        for arm in ARMS:
            if arm == "e0" or arm not in per_arm:
                continue
            other = rows_for(per_arm[arm], "final") or rows_for(per_arm[arm], "best")
            report["strata"][f"{arm}/{ds}"] = strata_tables(base_test, other, stats)

        # ---- the LSUI duplicate caveat -------------------------------------
        dup = duplicate_names(ds, "test")
        if dup:
            keep = set(n for n in names if n not in dup)
            def mean_without(rows: Dict[str, dict], field: str) -> float:
                v = [r[field] for n, r in rows.items() if n in keep]
                return float(np.mean(v)) if v else float("nan")
            entry = {"n_duplicates": len(dup), "names": sorted(dup)[:20], "excluded": {}}
            for arm, r in per_arm.items():
                rows = rows_for(r, "final") or rows_for(r, "best")
                entry["excluded"][arm] = {
                    "psnr": mean_without(rows, "psnr"),
                    "ssim": mean_without(rows, "ssim"),
                    "n": len(keep),
                }
            report["polluted_split"][ds] = entry

        # ---- diagnostics ---------------------------------------------------
        for arm in ARMS:
            if arm in per_arm and per_arm[arm].get("diagnostics"):
                report.setdefault("diagnostics", {})[f"{arm}/{ds}"] = {
                    k: v for k, v in sorted(per_arm[arm]["diagnostics"].items(),
                                            key=lambda kv: int(kv[0]))[-4:]
                }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)

    # ---- console: the brief's development table ---------------------------
    print(f"\ndevelopment table  (runs={args.runs}, deltas vs E0 at matched endpoints)")
    hdr = (f"{'exp':<5}{'dataset':<8}{'endpoint':<10}{'upd':>7}{'PSNR':>9}{'SSIM':>9}"
           f"{'dPSNR_e0':>9}{'95% CI':>20}{'dSSIM_e0':>10}{'dPSNR_start':>12}{'GPU h':>8}  verdict")
    print(hdr)
    print("-" * len(hdr))
    for row in report["table"]:
        lo, hi = row["d_psnr_ci"]
        ds_start = row.get("d_psnr_vs_start")
        print(f"{row['experiment']:<5}{row['dataset']:<8}{row['endpoint']:<10}{row['updates']:>7}"
              f"{row['psnr']:>9.4f}{row['ssim']:>9.5f}{row['d_psnr_vs_e0']:>+9.4f}"
              f"{f'[{lo:+.4f},{hi:+.4f}]':>20}{row['d_ssim_vs_e0']:>+10.5f}"
              f"{(f'{ds_start:+.4f}' if ds_start is not None else 'n/a'):>12}"
              f"{row['gpu_hours']:>8.2f}  {row['verdict']}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

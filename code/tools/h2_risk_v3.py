"""H2 v3 -- image-level risk model: does observation support predict enhancement
error over ordinary uncertainty and degradation, out of fold?

The falsifier (novelty audit #5, and the claim the project rests on):

    if the support statistic carries no predictive power over (a) ordinary image
    statistics, (b) fitted degradation, and (c) the spread of an independently
    trained ensemble of restorers, then it is a rebadged uncertainty map and the
    "observation-supported" framing is not earned.

Why this is fitted on test-split images only
--------------------------------------------
The frozen UIEB checkpoints return ~42 dB on ``UIEB/train`` (they memorised it;
``UIEB/test`` gives 24.5 dB).  An error model fitted on memorised errors learns
only "error is zero".  Both ``val`` and ``test`` are unseen by every checkpoint
used here, and only test-split predictions already existed for all seven
methods, so the study is run there under **image-disjoint 5-fold
cross-validation**: every image is predicted out-of-fold exactly once, and no
image contributes to both sides of a fold.

Arms (identical capacity, seeds, optimiser, folds)
--------------------------------------------------
F0  ordinary: 9 observation statistics + 4 ensemble-spread statistics
    (per-pixel standard deviation across 7 independently trained restorers,
    reduced to luma / rg / gb / all-channel means).  This is the strongest
    ordinary uncertainty comparator available without training MC-dropout.
F1  F0 + 7 support statistics (leverage-weighted information per direction
    family + gauge-ambiguity budget).
F2  F0 + the same 7 columns shuffled across images (fixed seed).
F3  F0 + 7 Gaussian-noise columns.

Model: heteroscedastic Gaussian head (mean + log-variance), NLL, early stop on an
image-disjoint tenth of each fold's training images, three seeds.  Deliberately
simple: an R^2 gap that needs a tuned architecture is not a measurement.

Honest status: ``screening``.  The test splits were inspected earlier for
protocol work (the frozen anchor), so this is not a blind confirmation, and the
UIEB/LSUI test splits contain a few byte-identical training samples (see
``docs/canonical_split_audit_20261105.md``).

Outputs ``artifacts/h2_risk_v3.json``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import obsui  # noqa: E402,F401

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

# The fits are tiny; on a shared 96-core host the default thread pool costs
# more in contention than it saves.  Pin before any tensor work.
torch.set_num_threads(2)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FEAT = os.path.join(ROOT, "artifacts", "features_v3")
ART = os.path.join(ROOT, "artifacts", "h2_risk_v3.json")

ORD = ["lum_mean", "lum_std", "rg_mean", "gb_mean", "grad", "noise_lap",
       "sat_frac", "entropy", "tau_mean"]
SUP = ["lq_colour", "lq_band", "lq_pixel", "lev_colour", "frac_qeff_colour",
       "lq_colour_min", "lq_colour_max"]
ENS = ["ens_lum", "ens_rg", "ens_gb", "ens_all"]
TGT = ["psnr", "ssim", "chroma", "rel", "abs"]
LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
BACKBONES = ["phaseformer", "aquaclarity", "fdcenet"]


def ensemble_stats(std_grid: np.ndarray) -> np.ndarray:
    lum = std_grid @ LUMA
    rg = std_grid[..., 0] - std_grid[..., 1]
    gb = std_grid[..., 1] - std_grid[..., 2]
    return np.stack([lum.mean(axis=(1, 2)), np.abs(rg).mean(axis=(1, 2)),
                     np.abs(gb).mean(axis=(1, 2)), std_grid.mean(axis=(1, 2, 3))], axis=1)


def load_domain(dataset: str, backbones=None) -> dict:
    zd = np.load(os.path.join(FEAT, f"disagree_{dataset}_test.npz"), allow_pickle=True)
    dmap = {n: i for i, n in enumerate(list(zd["names"]))}
    ens = ensemble_stats(np.asarray(zd["absdiff"], dtype=np.float64))

    rows = {"names": [], "backbone": [], "ordinary": [], "support": [], "ensemble": [], "targets": []}
    used = []
    for b in (backbones or BACKBONES):
        path = os.path.join(FEAT, f"{dataset}_test_{b}.npz")
        if not os.path.exists(path):
            # A missing backbone is a coverage gap, not a zero: leave the file
            # out and record the actual list in the artifact.
            continue
        used.append(b)
        z = np.load(path, allow_pickle=True)
        for i, n in enumerate(list(z["names"])):
            if n not in dmap:
                continue
            rows["names"].append(n)
            rows["backbone"].append(b)
            rows["ordinary"].append(z["ordinary"][i])
            rows["support"].append(z["support"][i])
            rows["ensemble"].append(ens[dmap[n]])
            rows["targets"].append(z["targets"][i])
    for k in ("ordinary", "support", "ensemble", "targets"):
        rows[k] = np.asarray(rows[k], dtype=np.float64)
    rows["names"] = np.array(rows["names"])
    rows["used_backbones"] = used
    return rows


class Net(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(d, 32), nn.SiLU(), nn.Linear(32, 16), nn.SiLU())
        self.mu = nn.Linear(16, 1)
        self.logvar = nn.Linear(16, 1)

    def forward(self, x):
        h = self.body(x)
        return self.mu(h).squeeze(-1), self.logvar(h).squeeze(-1).clamp(-8, 4)


def fit_predict(Xtr, ytr, Xdv, ydv, Xte, seed, epochs=500):
    torch.manual_seed(seed)
    xm, xs = Xtr.mean(0), Xtr.std(0) + 1e-8
    ym, ys = ytr.mean(), ytr.std() + 1e-8
    T = lambda X: torch.tensor((X - xm) / xs, dtype=torch.float32)
    xtr, xdv, xte = T(Xtr), T(Xdv), T(Xte)
    a, b = torch.tensor((ytr - ym) / ys, dtype=torch.float32), torch.tensor((ydv - ym) / ys, dtype=torch.float32)
    net, best = Net(Xtr.shape[1]), (1e18, None)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-4)

    def nll(x, y):
        mu, lv = net(x)
        return (0.5 * (lv + (y - mu) ** 2 / lv.exp())).mean()

    for ep in range(epochs):
        net.train()
        opt.zero_grad()
        nll(xtr, a).backward()
        opt.step()
        if ep % 20 == 0 or ep == epochs - 1:
            net.eval()
            with torch.no_grad():
                v = float(nll(xdv, b))
            if v < best[0]:
                best = (v, {k: t.detach().clone() for k, t in net.state_dict().items()})
    if best[1] is not None:
        net.load_state_dict(best[1])
    net.eval()
    with torch.no_grad():
        mu, lv = net(xte)
    return mu.numpy() * ys + ym, lv.numpy() + 2 * np.log(ys)


def r2(y, yhat):
    return 1.0 - float(((y - yhat) ** 2).sum()) / max(float(((y - y.mean()) ** 2).sum()), 1e-12)


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def run_domain(dataset, seeds, boot, rng, folds=5, targets=None, epochs=250):
    d = load_domain(dataset)
    imgs = np.array(sorted(set(d["names"])))
    img_idx = {n: i for i, n in enumerate(imgs)}
    fold_of_img = np.arange(len(imgs)) % folds
    fold_of_row = np.array([fold_of_img[img_idx[n]] for n in d["names"]])
    base = np.concatenate([d["ordinary"], d["ensemble"]], axis=1)
    sup = d["support"]
    shuf = np.stack([rng.permutation(sup[:, j]) for j in range(sup.shape[1])], axis=1)
    noise = rng.normal(size=sup.shape)
    arms = {"F0_ordinary": base,
            "F1_plus_support": np.concatenate([base, sup], 1),
            "F2_shuffled_support": np.concatenate([base, shuf], 1),
            "F3_noise_features": np.concatenate([base, noise], 1)}

    out = {"n_images": int(len(imgs)), "n_rows": int(len(d["names"])),
           "backbones": d.get("used_backbones", BACKBONES), "folds": folds, "targets": {}}
    row_img = np.array([img_idx[n] for n in d["names"]])
    for ti, tname in enumerate(TGT):
        if targets and tname not in targets:
            continue
        y = d["targets"][:, ti]
        oof = {a: np.zeros(len(y)) for a in arms}
        fold_delta = {a: [] for a in arms if a != "F0_ordinary"}
        fold_r2s = {a: [] for a in arms}
        for f in range(folds):
            te = fold_of_row == f
            tr_all = ~te
            # image-disjoint dev carve inside the training side
            tr_imgs = np.array(sorted(set(row_img[tr_all])))
            dev_imgs = set(tr_imgs[::10])
            dv = tr_all & np.array([i in dev_imgs for i in row_img])
            tr = tr_all & ~dv
            fold_r2 = {}
            for a, X in arms.items():
                preds = []
                for sd in seeds:
                    mu, _ = fit_predict(X[tr], y[tr], X[dv], y[dv], X[te], sd, epochs=epochs)
                    preds.append(mu)
                oof[a][te] = np.mean(preds, axis=0)
                fold_r2[a] = r2(y[te], oof[a][te])
            for a in arms:
                fold_r2s[a].append(float(fold_r2[a]))
            for a in fold_delta:
                fold_delta[a].append(fold_r2[a] - fold_r2["F0_ordinary"])

        pooled = {a: r2(y, oof[a]) for a in arms}
        by_img = [np.where(row_img == k)[0] for k in range(len(imgs))]
        deltas = {a: [] for a in ("F1_plus_support", "F2_shuffled_support", "F1_vs_F2")}
        for _ in range(boot):
            pick = rng.integers(0, len(by_img), len(by_img))
            idx = np.concatenate([by_img[p] for p in pick])
            yb = y[idx]
            r0, r1, r2_ = r2(yb, oof["F0_ordinary"][idx]), r2(yb, oof["F1_plus_support"][idx]), \
                r2(yb, oof["F2_shuffled_support"][idx])
            deltas["F1_plus_support"].append(r1 - r0)
            deltas["F2_shuffled_support"].append(r2_ - r0)
            deltas["F1_vs_F2"].append(r1 - r2_)

        k = max(1, int(round(0.2 * len(y))))
        order = np.argsort(oof["F1_plus_support"])
        out["targets"][tname] = {
            "r2_pooled_oof": pooled,
            "r2_per_fold": fold_r2s,
            "delta_per_fold_mean": {a: float(np.mean(v)) for a, v in fold_delta.items()},
            "delta_bootstrap": {a: {"mean": float(np.mean(v)),
                                    "ci95": [float(np.quantile(v, .025)), float(np.quantile(v, .975))]}
                                for a, v in deltas.items()},
            "spearman_F1": spearman(y, oof["F1_plus_support"]),
            "spearman_F0": spearman(y, oof["F0_ordinary"]),
            "gating_F1_low_risk": float(y[order[:k]].mean()),
            "gating_F1_high_risk": float(y[order[-k:]].mean()),
            "y_mean": float(y.mean()), "y_std": float(y.std()),
            "oof": {a: oof[a].tolist() for a in arms},
            "row_images": row_img.tolist(),
        }
        print(f"[{dataset}/{tname}] pooled R2 " +
              "  ".join(f"{a}={pooled[a]:.4f}" for a in arms), flush=True)
        db = out["targets"][tname]["delta_bootstrap"]["F1_plus_support"]
        print(f"   dR2(F1-F0)={db['mean']:+.4f} CI[{db['ci95'][0]:+.4f},{db['ci95'][1]:+.4f}]"
              f"   dR2(F1-F2)={out['targets'][tname]['delta_bootstrap']['F1_vs_F2']['mean']:+.4f}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", nargs="+", default=["LSUI", "UIEB"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--out", default=ART)
    ap.add_argument("--targets", nargs="+", default=None,
                    help="subset of psnr ssim chroma rel abs")
    ap.add_argument("--epochs", type=int, default=250)
    args = ap.parse_args()
    rng = np.random.default_rng(20261105)
    res = {"version": "20261105.v3", "status": "screening_exploratory",
           "cv": "image-disjoint 5-fold, out-of-fold predictions pooled",
           "fit_data": "test-split predictions only (every frozen checkpoint memorised "
                       "its train split: UIEB train scores ~42 dB vs 24.5 dB on test)",
           "features": {"ordinary": ORD, "support": SUP, "ensemble_spread_7_methods": ENS},
           "model": "heteroscedastic Gaussian MLP 32-16, NLL, early stop on image-disjoint dev tenth",
           "seeds": args.seeds, "bootstrap": args.boot, "boot_unit": "image",
           "targets": args.targets or TGT, "epochs": args.epochs, "domains": {}}
    for dm in args.domains:
        print(f"=== {dm} ===", flush=True)
        res["domains"][dm] = run_domain(dm, args.seeds, args.boot, rng,
                                        targets=args.targets, epochs=args.epochs)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()

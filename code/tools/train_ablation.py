"""Train E0-E3 (first-round architecture ablation) on UIEB and LSUI.

One run = one (arm, dataset) cell of
``docs/phase1_architecture_ablation_plan_20261109.md``.  Everything except the
inserted module is held fixed across arms: same pretrained start, same loader,
same optimiser, same schedule, same loss, same seed, same number of updates.

What is fixed and where it comes from
-------------------------------------
* backbone and start weights: the published MambaCue checkpoints, loaded
  ``strict=True``; the ablation module is bolted on afterwards and is an exact
  identity at step 0, so ``update 0`` of every arm must reproduce the start
  weights' own validation scores.  ``--verify-init`` checks exactly that.
* loss: MambaCue's own ``UPSNetLoss`` imported from the repo, unmodified.  The
  L-stage adaptive-weight loss is a *separate* future stage (user decision,
  2026-11-09) and is deliberately not implemented here.
* optimiser: AdamW(betas=(0.9,0.999), weight_decay=1e-4), two parameter groups --
  backbone at ``--lr-backbone``, new module at ``--lr-new`` (brief section 2.3:
  1e-5..3e-5 and 1e-4).
* schedule: linear warmup over the first ``--warmup-updates`` updates, then
  cosine to ``--lr-min``.  Stepped per *optimiser update*, not per epoch, because
  the run length is defined in updates.
* grad clip: max_norm 1.0 (brief section 6, "stability fallback", enabled).
* no AMP: the brief keeps sensitive ops in the precision the implementation was
  verified with, and mamba_ssm 1.1.1's selective scan is not autocast-safe.
* data: input/GT pairs matched by stem, whole image resized to 256x256 bilinear;
  geometry flips applied to input *and* ground truth together, so no
  misregistration is introduced.

Selection and reporting
-----------------------
Validation runs every ``--eval-every`` updates on the fixed validation split and
picks the checkpoint by ``psnr + 10 * ssim``, the same composite MambaCue used.
``best`` and ``last`` are both written.  The test split is scored once, at the
end, for the selected checkpoint -- it is *not* used for selection.

Every run writes one JSON with the full validation curve, per-image metrics for
val and test, timings, peak memory, and the branch/gate diagnostics the brief
asks for.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from obsui import ablate_eval, dataio  # noqa: E402
from obsui.fusion import MODES, attach, count_params, module_parameters  # noqa: E402

MAMBA_ROOT = "/home/user/MambaCue"
DATA_ROOT = "/home/user/UnderWaterDataset"

START_WEIGHTS = {
    "UIEB": f"{MAMBA_ROOT}/output/L1+SSIM+Prep+Lab+grad+Angle/four/"
            "color-illumination-contrast/both/UIEB/20261009_092226/best_model.pth",
    "LSUI": f"{MAMBA_ROOT}/output/L1+SSIM+Prep+Lab+grad+Angle/four/"
            "color-illumination-contrast/both/LSUI/20261009_112952/best_model.pth",
}


# --------------------------------------------------------------------------- #
# reproducibility
# --------------------------------------------------------------------------- #
def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_backbone(device: torch.device, weights: str) -> torch.nn.Module:
    """MambaCue ``myModel`` with the published weights, unmodified."""
    if MAMBA_ROOT not in sys.path:
        sys.path.insert(0, MAMBA_ROOT)
    from model import myModel  # noqa: E402  (lives in the MambaCue repo)

    model = myModel(
        in_channels=3, hidden_channels=32, out_channels=3,
        prior_types=("color", "illumination", "contrast"),
        scan_mode="four", use_checkpoint=False, output_heads="both",
    )
    sd = torch.load(weights, map_location="cpu")
    model.load_state_dict(sd, strict=True)
    return model.to(device)


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
class PairDataset(torch.utils.data.Dataset):
    """Matched input/GT pairs; geometry flips always hit both sides.

    Notes on the loader
    -------------------
    MambaCue's own ``DatasetFromFolder`` sorts ``input`` and ``GT`` separately and
    pairs by index.  Verified on this data: the stems agree at every index for
    UIEB and LSUI train/val/test (0/712, 0/89, 0/89, 0/3423, 0/428, 0/428
    mismatches), so index pairing and stem matching describe the same pairs.
    Stem matching is used here anyway because it fails loudly instead of quietly
    mispairing if the two directory listings ever diverge.
    """

    def __init__(self, dataset: str, split: str, size: int = 256,
                 train: bool = True, data_root: str = DATA_ROOT):
        root = os.path.join(data_root, dataset, split)
        ps = dataio.resolve_pairs(os.path.join(root, "input"), os.path.join(root, "GT"), dataset)
        unmatched = [n for n in ps.notes if n.startswith("UNMATCHED_SAMPLE")]
        if unmatched:
            raise SystemExit(f"{dataset}/{split}: unmatched references {unmatched}")
        if not ps.pairs:
            raise SystemExit(f"{dataset}/{split}: no pairs")
        self.pairs = ps.pairs
        self.size = size
        self.train = train

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, i: int):
        p = self.pairs[i]
        a = Image.fromarray(dataio.load_rgb(p.input_path), mode="RGB").resize(
            (self.size, self.size), Image.BILINEAR)
        b = Image.fromarray(dataio.load_rgb(p.ref_path), mode="RGB").resize(
            (self.size, self.size), Image.BILINEAR)
        if self.train:
            if random.random() < 0.5:
                a, b = ImageOps.flip(a), ImageOps.flip(b)
            if random.random() < 0.5:
                a, b = ImageOps.mirror(a), ImageOps.mirror(b)
        x = torch.from_numpy(np.asarray(a, dtype=np.float32) / 255.0).permute(2, 0, 1)
        y = torch.from_numpy(np.asarray(b, dtype=np.float32) / 255.0).permute(2, 0, 1)
        return x, y, p.name


def make_loader(dataset: str, split: str, batch: int, train: bool,
                workers: int, seed: int) -> torch.utils.data.DataLoader:
    ds = PairDataset(dataset, split, train=train)
    g = torch.Generator()
    g.manual_seed(seed)
    return torch.utils.data.DataLoader(
        ds, batch_size=batch, shuffle=train, num_workers=workers,
        pin_memory=True, drop_last=train, generator=g if train else None,
        persistent_workers=workers > 0,
    )


# --------------------------------------------------------------------------- #
# fixed validation set (pre-loaded once, evaluated many times)
# --------------------------------------------------------------------------- #
def load_eval_set(dataset: str, split: str) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    return ablate_eval.make_pairs(dataset, split)


@torch.no_grad()
def evaluate(model: torch.nn.Module, pairs, device, batch: int = 8,
             module: Optional[torch.nn.Module] = None) -> Dict[str, object]:
    """Score one split; ``module`` is toggled alongside the backbone.

    The ablation module is a plain sequential stack with no batch-norm or
    dropout, so the toggle is numerically inert today -- it is done explicitly so
    that it stays correct if the module ever gains such a layer.
    """
    if module is not None:
        module.eval()
    try:
        rows = ablate_eval.evaluate_model(model, pairs, device, batch=batch)
    finally:
        if module is not None:
            module.train()
    agg = ablate_eval.aggregate(rows)
    agg["rows"] = rows
    return agg


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #
def lr_at(step: int, total: int, warmup: int, base: float, floor: float) -> float:
    """Linear warmup then cosine, in *updates*."""
    if warmup > 0 and step < warmup:
        return base * float(step + 1) / float(warmup)
    span = max(1, total - warmup)
    t = min(1.0, max(0.0, (step - warmup) / span))
    return floor + 0.5 * (base - floor) * (1.0 + np.cos(np.pi * t))


def main() -> int:
    ap = argparse.ArgumentParser(description="Train one E0-E3 ablation cell")
    ap.add_argument("--arm", required=True, choices=list(MODES))
    ap.add_argument("--dataset", required=True, choices=["UIEB", "LSUI"])
    ap.add_argument("--updates", type=int, default=10000,
                    help="optimiser updates (brief: 5000-10000; extended run: 20000)")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=2,
                    help="gradient accumulation; effective batch = batch * accum (MambaCue used 8*2)")
    ap.add_argument("--lr-backbone", type=float, default=1e-5)
    ap.add_argument("--lr-new", type=float, default=1e-4)
    ap.add_argument("--lr-min", type=float, default=1e-6)
    ap.add_argument("--warmup-updates", type=int, default=300)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--e1-blocks", type=int, default=None,
                    help="override the parameter-matched E1 depth (default: matched to E2)")
    ap.add_argument("--start-weights", default=None, help="override the per-dataset start checkpoint")
    ap.add_argument("--out-dir", default="/home/user/obsui/runs/ablation_phase1")
    ap.add_argument("--tag", default=None, help="run name; default arm_dataset_updates")
    ap.add_argument("--verify-init", action="store_true",
                    help="score the untrained start point and exit (no optimisation)")
    ap.add_argument("--resume", default=None, help="checkpoint to continue a run from")
    args = ap.parse_args()

    tag = args.tag or f"{args.arm}_{args.dataset}_{args.updates}"
    run_dir = os.path.join(args.out_dir, tag)
    os.makedirs(run_dir, exist_ok=True)
    ckpt_dir = os.path.join(run_dir, "ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)

    seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    weights = args.start_weights or START_WEIGHTS[args.dataset]
    if not os.path.isfile(weights):
        raise SystemExit(f"start weights not found: {weights}")

    # ---- model ----------------------------------------------------------- #
    model = build_backbone(device, weights)
    backbone_params = [p for p in model.parameters() if p.requires_grad]
    module, hook = attach(model, args.arm, e1_blocks=args.e1_blocks)
    module = module.to(device)
    new_params = module_parameters(module)

    n_backbone = sum(p.numel() for p in backbone_params)
    n_new = count_params(module)
    print(f"[{tag}] backbone={n_backbone} new={n_new} arm={args.arm}", flush=True)

    if args.resume:
        ck = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ck["model"], strict=True)
        if n_new:
            module.load_state_dict(ck["module"], strict=True)
        print(f"[{tag}] resumed from {args.resume}", flush=True)

    # ---- fixed eval sets ------------------------------------------------- #
    val_pairs = load_eval_set(args.dataset, "val")
    print(f"[{tag}] val pairs={len(val_pairs)}", flush=True)

    def score_now() -> Dict[str, object]:
        r = evaluate(model, val_pairs, device, module=module if n_new else None)
        print(f"[{tag}] val n={r['n']} PSNR={r['psnr']:.4f} SSIM={r['ssim']:.6f}", flush=True)
        return r

    # ---- anchor: score the untouched start point ------------------------- #
    # Recorded for every arm.  The published checkpoint was itself selected on
    # this validation split by the same psnr + 10*ssim score, so continuing to
    # train is expected to move *below* it for a while; the anchor is what makes
    # that visible per arm instead of only in a note.  Test is scored too, so the
    # report can separate "improved over the start point" from "degraded more
    # slowly than E0" -- two very different claims.
    r_start_val = score_now()
    r_start_test = evaluate(model, load_eval_set(args.dataset, "test"), device,
                            module=module if n_new else None)
    print(f"[{tag}] start test n={r_start_test['n']} PSNR={r_start_test['psnr']:.4f} "
          f"SSIM={r_start_test['ssim']:.6f}", flush=True)

    # ---- optional: score the start point, then stop ---------------------- #
    if args.verify_init:
        r = score_now()
        # Score test as well: the start weights' own test score is the anchor the
        # repository already trusts (``artifacts/mambacue_verify.json``), so this
        # both pins "update 0" per arm and re-proves the frozen protocol end to end.
        test_pairs = load_eval_set(args.dataset, "test")
        r_test = evaluate(model, test_pairs, device, module=module if n_new else None)
        print(f"[{tag}] test n={r_test['n']} PSNR={r_test['psnr']:.4f} SSIM={r_test['ssim']:.6f}", flush=True)
        out = {
            "tag": tag, "arm": args.arm, "dataset": args.dataset, "mode": "verify_init",
            "start_weights": weights, "n_backbone_params": n_backbone, "n_new_params": n_new,
            "val": {k: v for k, v in r.items() if k != "rows"}, "val_rows": r["rows"],
            "test": {k: v for k, v in r_test.items() if k != "rows"}, "test_rows": r_test["rows"],
        }
        path = os.path.join(run_dir, "verify_init.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=1)
        print(f"[{tag}] wrote {path}", flush=True)
        return 0

    # ---- optimiser ------------------------------------------------------- #
    groups = [{"params": backbone_params, "lr": args.lr_backbone, "name": "backbone"}]
    if new_params:
        groups.append({"params": new_params, "lr": args.lr_new, "name": "new"})
    optim = torch.optim.AdamW(groups, betas=(0.9, 0.999), weight_decay=1e-4)

    # ---- loss: MambaCue's own, unmodified -------------------------------- #
    from utils.loss_funcs import UPSNetLoss  # noqa: E402
    criterion = UPSNetLoss().to(device)

    loader = make_loader(args.dataset, "train", args.batch, True, args.workers, args.seed)
    it = iter(loader)

    # ---- training loop --------------------------------------------------- #
    t_start = time.time()
    diag: Dict[str, object] = {}
    curve: List[Dict[str, object]] = []
    start_score = r_start_val["psnr"] + 10.0 * r_start_val["ssim"]
    best = {"score": start_score, "update": 0, "psnr": r_start_val["psnr"],
            "ssim": r_start_val["ssim"], "from_start": True}
    # The start point is the selection baseline, so an arm that never beats it
    # keeps the *untrained* weights -- that is the honest "no gain" outcome and
    # it is recorded as update 0 rather than silently replaced by a worse one.
    torch.save({"model": model.state_dict(), "module": module.state_dict() if n_new else {},
                "update": 0, "args": vars(args)},
               os.path.join(ckpt_dir, "best.pth"))
    curve.append({
        "update": 0, "psnr": r_start_val["psnr"], "ssim": r_start_val["ssim"],
        "psnr_std": r_start_val["psnr_std"], "ssim_std": r_start_val["ssim_std"],
        "score": start_score, "sec": 0.0,
        "peak_mem_gb": (torch.cuda.max_memory_allocated() / 1e9) if device.type == "cuda" else None,
        "anchor": True,
    })
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    model.train()
    for upd in range(args.updates):
        step_t0 = time.time()
        for i, pg in enumerate(optim.param_groups):
            base = args.lr_backbone if i == 0 else args.lr_new
            pg["lr"] = lr_at(upd, args.updates, args.warmup_updates, base, args.lr_min)

        optim.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for _ in range(args.accum):
            try:
                x, y, _names = next(it)
            except StopIteration:
                it = iter(loader)
                x, y, _names = next(it)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            pred = model(x, prior_map=None, return_aux=False)
            if isinstance(pred, (tuple, list)):
                pred = pred[0]
            loss, parts = criterion(pred, y, None)
            (loss / args.accum).backward()
            loss_sum += float(parts["total"])

        if args.grad_clip and args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], args.grad_clip)
        optim.step()

        if upd % 100 == 0:
            lr_now = optim.param_groups[0]["lr"]
            print(f"[{tag}] upd {upd}/{args.updates} loss={loss_sum / args.accum:.4f} "
                  f"lr={lr_now:.2e} {time.time() - step_t0:.2f}s", flush=True)

        # diagnostics (branch norms, gate distribution) on a fixed probe batch.
        # ``dec3`` produces exactly the 1/4-resolution feature the module acts on,
        # so hooking its output records the module's real input without depending
        # on pre-hook ordering relative to the insertion hook.
        if args.arm != "e0" and (upd == 0 or (upd + 1) % args.eval_every == 0):
            with torch.no_grad():
                probe_x = x[:min(2, x.shape[0])]
                captured: Dict[str, torch.Tensor] = {}

                def _grab(_m, _inp, out):
                    captured["d3"] = out

                h = model.decoder.dec3.register_forward_hook(_grab)
                try:
                    model(probe_x, prior_map=None, return_aux=False)
                finally:
                    h.remove()
                if "d3" in captured:
                    diag[str(upd)] = module.stats(captured["d3"])

        if (upd + 1) % args.eval_every == 0 or (upd + 1) == args.updates:
            r = score_now()
            comp = r["psnr"] + 10.0 * r["ssim"]
            entry = {
                "update": upd + 1, "psnr": r["psnr"], "ssim": r["ssim"],
                "psnr_std": r["psnr_std"], "ssim_std": r["ssim_std"], "score": comp,
                "sec": time.time() - t_start,
                "peak_mem_gb": (torch.cuda.max_memory_allocated() / 1e9) if device.type == "cuda" else None,
            }
            curve.append(entry)
            improved = comp > best["score"]
            print(f"[{tag}] upd {upd + 1}: score={comp:.4f} best={best['score']:.4f} "
                  f"{'*' if improved else ''}", flush=True)
            if improved:
                best = {"score": comp, "update": upd + 1, "psnr": r["psnr"], "ssim": r["ssim"]}
                torch.save({"model": model.state_dict(), "module": module.state_dict() if n_new else {},
                            "update": upd + 1, "args": vars(args)},
                           os.path.join(ckpt_dir, "best.pth"))
            torch.save({"model": model.state_dict(), "module": module.state_dict() if n_new else {},
                        "update": upd + 1, "args": vars(args)},
                       os.path.join(ckpt_dir, "last.pth"))

    # ---- test, once, for each of the two checkpoints --------------------- #
    # ``final`` is the fixed-budget endpoint -- the arm's result after exactly the
    # same number of updates as every other arm, which is the controlled
    # comparison the brief asks for.  ``best`` is the validation-selected
    # checkpoint, reported alongside because it is what a practitioner would
    # ship; the two disagreeing is itself a result, not something to hide.
    test_pairs = load_eval_set(args.dataset, "test")
    endpoints: Dict[str, object] = {}
    for ck_name, ck_update in (("best", best["update"]),
                               ("final", args.updates if args.updates > 0 else 0)):
        ck_path = os.path.join(ckpt_dir, ("best" if ck_name == "best" else "last") + ".pth")
        if not os.path.isfile(ck_path):
            continue
        print(f"[{tag}] scoring {ck_name} (update {ck_update}) on val+test", flush=True)
        ck = torch.load(ck_path, map_location="cpu")
        model.load_state_dict(ck["model"], strict=True)
        if n_new:
            module.load_state_dict(ck["module"], strict=True)
        r_test = evaluate(model, test_pairs, device, module=module if n_new else None)
        r_val = evaluate(model, val_pairs, device, module=module if n_new else None)
        print(f"[{tag}] {ck_name}: test n={r_test['n']} PSNR={r_test['psnr']:.4f} "
              f"SSIM={r_test['ssim']:.6f} | val PSNR={r_val['psnr']:.4f}", flush=True)
        endpoints[ck_name] = {
            "update": ck_update,
            "val": {k: v for k, v in r_val.items() if k != "rows"},
            "test": {k: v for k, v in r_test.items() if k != "rows"},
            "val_rows": r_val["rows"],
            "test_rows": r_test["rows"],
        }

    # Top-level fields keep the historical ``best``/``test`` names so the report
    # tool has one shape to read; ``endpoints`` carries both checkpoints in full.
    r_test = endpoints.get("best", endpoints.get("final"))
    r_val_best = None
    for name in ("best", "final"):
        if name in endpoints:
            r_val_best = endpoints[name]["val"]
            break
    total_sec = time.time() - t_start
    out = {
        "tag": tag, "arm": args.arm, "dataset": args.dataset,
        "start_weights": weights,
        "seed": args.seed, "updates": args.updates,
        "batch": args.batch, "accum": args.accum, "effective_batch": args.batch * args.accum,
        "lr_backbone": args.lr_backbone, "lr_new": args.lr_new, "lr_min": args.lr_min,
        "warmup_updates": args.warmup_updates, "grad_clip": args.grad_clip,
        "eval_every": args.eval_every, "e1_blocks": getattr(module, "e1_blocks", None),
        "n_backbone_params": n_backbone, "n_new_params": n_new,
        "n_val": len(val_pairs), "n_test": len(test_pairs),
        "start": {k: v for k, v in r_start_val.items() if k != "rows"},
        "start_test": {k: v for k, v in r_start_test.items() if k != "rows"},
        "start_test_rows": r_start_test["rows"],
        "best": best, "curve": curve,
        "val_best": r_val_best,
        "val_rows_best": endpoints["best"]["val_rows"] if "best" in endpoints else [],
        "test": r_test["test"] if r_test else None,
        "test_rows": endpoints["best"]["test_rows"] if "best" in endpoints else [],
        "endpoints": endpoints,
        "diagnostics": diag,
        "sec_total": total_sec,
        "sec_per_update": total_sec / args.updates,
        "peak_mem_gb": (torch.cuda.max_memory_allocated() / 1e9) if device.type == "cuda" else None,
        "env": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
            "gpu_total_mib": (torch.cuda.get_device_properties(0).total_memory // (1024 ** 2))
            if device.type == "cuda" else None,
        },
    }
    path = os.path.join(run_dir, "result.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(f"[{tag}] wrote {path}  ({total_sec / 60:.1f} min)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

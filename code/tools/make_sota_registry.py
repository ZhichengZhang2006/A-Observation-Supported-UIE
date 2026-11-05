#!/usr/bin/env python
"""Merge the per-method baseline artifacts into ``docs/sota_registry.csv``.

Columns (exactly)::

    method,dataset,reported_psnr,reported_ssim,unified_psnr,unified_ssim,
    unified_n,unified_source,retrain_psnr,retrain_ssim,notes

``reported_*``
    The original authors' numbers, transcribed verbatim from the user's
    ``UIE-data.xlsx`` (their own preprocessing, their own image sizes). These
    are *not* comparable with each other and must never be used as a
    same-protocol ranking; they are kept only so the gap between the reported
    and the re-measured column is visible. Blank where the paper gave no
    number for that dataset.

``unified_*``
    What this project measured by running the **official weights** through
    ``code/tools/run_baseline.py`` under the frozen protocol: whole image
    bilinear-resized to 256x256, per-image PSNR/SSIM, arithmetic mean over
    all images. ``unified_n`` is the number of images that produced a
    prediction -- every image in the split is expected to count.

``unified_source``
    ``official_weights_rerun`` -- measured here from the shipped checkpoint.
    ``not_run``                -- no measurement (reason in ``notes``).
    ``anchor_saved_tiles``     -- our own MambaCue rows only: the numbers were
    reproduced by scoring its saved prediction tiles, which is how the
    protocol itself was validated. It is *not* a weights rerun, and MambaCue
    is not one of the third-party baselines this registry exists to audit.

``retrain_*``
    Reserved for the same-protocol retraining track; out of scope here and
    therefore always blank.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(HERE))

ART_ROOT = os.path.join(PROJECT_ROOT, "artifacts")
DOCS_ROOT = os.path.join(PROJECT_ROOT, "docs")

COLUMNS = [
    "method",
    "dataset",
    "reported_psnr",
    "reported_ssim",
    "unified_psnr",
    "unified_ssim",
    "unified_n",
    "unified_source",
    "retrain_psnr",
    "retrain_ssim",
    "notes",
]

DATASETS = ("UIEB", "LSUI", "UFO")

# --------------------------------------------------------------------------- #
# reported numbers -- transcribed from UIE-data.xlsx Sheet1 (SSIM/PSNR columns)
# --------------------------------------------------------------------------- #
# Rows are {dataset: (psnr, ssim)}. Full precision exactly as in the workbook.
REPORTED = {
    "WWE-UIE": {
        "UIEB": ("24.3306", "0.9159"),
        "LSUI": ("30.7335", "0.9287"),
        "UFO": ("28.3005", "0.8646"),
    },
    "Phaseformer": {
        "UIEB": ("24.0850", "0.9133"),
        "LSUI": ("29.2322", "0.9055"),
        "UFO": ("27.9119", "0.8635"),
    },
    "CDF-UIE": {
        "UIEB": ("20.194892", "0.817281"),
        "LSUI": ("22.689833", "0.846999"),
        "UFO": ("23.053718", "0.725072"),
    },
    "SS-UIE": {
        "UIEB": ("21.330721", "0.860309"),
        "LSUI": ("27.161259", "0.907197"),
        "UFO": ("25.339024", "0.808571"),
    },
    "AquaClarity": {
        "UIEB": ("23.3939", "0.9120"),
        "LSUI": ("30.1496", "0.9405"),
        "UFO": ("27.6985", "0.8660"),
    },
    "CCL-Net": {
        "UIEB": ("21.1274", "0.866301"),
        "LSUI": ("25.6013", "0.874615"),
        "UFO": ("24.3623", "0.795100"),
    },
    "FDCE-Net": {
        "UIEB": ("22.688131", "0.900528"),
        "LSUI": ("26.177544", "0.918709"),
        "UFO": ("24.0020", "0.827170"),
    },
    "WaterNet": {
        "UIEB": ("23.2669414", "0.8902"),
        "LSUI": ("24.3388", "0.8931"),
        "UFO": ("24.6429", "0.7810"),
    },
    "SFGNet": {
        "UIEB": ("18.323525248356951", "0.64757554869303546"),
        "LSUI": ("23.593223739612299", "0.80987133113580334"),
        "UFO": ("25.947737063588171", "0.84240526407957073"),
    },
    "MambaCue": {
        "UIEB": ("24.3785", "0.9245"),
        "LSUI": ("31.4088", "0.9359"),
        "UFO": ("28.6144", "0.8794"),
    },
}

# method name -> --method key used by run_baseline.py (None = never run)
METHOD_KEYS = {
    "Phaseformer": "phaseformer",
    "SS-UIE": "ssuie",
    "CCL-Net": "cclnet",
    "AquaClarity": "aquaclarity",
    "CDF-UIE": "cdfuie",
    "SFGNet": "sfgnet",
    "FDCE-Net": "fdcenet",
    "WWE-UIE": None,
    "WaterNet": None,
    # MambaCue is not run through the generic harness: its official predictions are
    # already on disk as side-by-side tiles (raw | GT | pred), so ``load_artifact``
    # reads ``mambacue_verify.json`` instead. It must NOT be left as "not_run" --
    # it is the frozen SOTA the whole 5% gate is measured against, and a blank
    # reference row would make every gap unfalsifiable.
    "MambaCue": "mambacue",
}

# Order the rows the way the task lists the baselines.
METHOD_ORDER = [
    "AquaClarity",
    "CDF-UIE",
    "Phaseformer",
    "SFGNet",
    "SS-UIE",
    "CCL-Net",
    "FDCE-Net",
    "WWE-UIE",
    "WaterNet",
    "MambaCue",
]

# Why a method produced no measurement at all.
NOT_RUN_REASON = {
    "WWE-UIE": (
        "未查到: WWE-UIE-main ships no main-network weights on the server "
        "(only the URanker/CIDNet auxiliary checkpoints), so the enhancement "
        "network cannot be instantiated from official weights. Not run -- no "
        "number is reported rather than an estimate."
    ),
    "WaterNet": (
        "未查到: no WaterNet repository with weights on the server, and the "
        "local copy (Desktop/waternet-main) contains no checkpoint either. "
        "Not run -- no number is reported rather than an estimate."
    ),
}

# Per-method caveats that apply to every dataset row for that method.
METHOD_NOTES = {
    "Phaseformer": (
        "local checkpoint (publication origin unverified) = checkpoint/<DS>/netG_model_epoch_800.pth (repo ships "
        "801 epochs with no 'best' marker; epoch 800 is the one the repo's own "
        "evaluation used). Input normalised to [-1,1]; output taken from "
        "tuple[0]. Repo test.py resizes BICUBIC and its metric uses "
        "win_size=min(7,..); both are overridden here."
    ),
    "SS-UIE": (
        "local checkpoint (publication origin unverified) = saved_models/<DS>/SS_UIE_best.pth['model'] (dict "
        "checkpoint; carries epoch/best_psnr). Model built with num_memblock=4, "
        "num_resblock=4 to match the shipped weights. Forward uses spectral "
        "ops, which cannot run on CUDA on this host (see notes below)."
    ),
    "CCL-Net": (
        "official configuration is the TWO-STAGE HRNet graph, not the CC "
        "branch alone: checkpoints/<sub>/train_opt.txt records model: HRNet, "
        "and models/HRNet_model.py runs netG_CC(Lab) -> lab2rgb -> netG(RGB) "
        "and reports pred_hr. Weights best_net_G_CC.pth + best_net_G.pth. "
        "Dataset dir for UFO is UFO_CCL (not UFO_CC)."
    ),
    "AquaClarity": (
        "local checkpoint (publication origin unverified) = checkpoints/<DS>/best.pth['state_dict'] (dict "
        "checkpoint; carries epoch/psnr/ssim). Model dim=36 as in the shipped "
        "weights. Output already clamped to [0,1] by the model's NILUT."
    ),
    "CDF-UIE": (
        "official weights = ckpts/<DS-...>/ UHDFour-epoch*.pt; the repo ships no "
        "'best' marker, so the file with the highest validation PSNR embedded "
        "in its filename is used (the authors' own selection signal). "
        "Checkpoint keys carry a DataParallel 'module.' prefix. Output taken "
        "from tuple[0]."
    ),
    "SFGNet": (
        "local checkpoint (publication origin unverified) = models/<DS>/net_last.pth (the default in the repo's "
        "eval.py). Output taken from tuple[1] (restored). Two Sobel-kernel "
        "buffers (g.weight_v/weight_h) are absent from the checkpoint because "
        "they are fixed constants re-registered by the module, not trained."
    ),
    "FDCE-Net": (
        "local checkpoint (publication origin unverified) = training_data/<DS>/checkpoints/best.pth['model']. "
        "Output taken from tuple[2] (img_enhance). Their test.py resizes "
        "max-side 512; that is overridden by the frozen 256x256 resize."
    ),
    "MambaCue": (
        "LEGACY REFERENCE ONLY -- NOT a unified-protocol rerun. The numbers in "
        "this row were re-derived by splitting the repo's saved prediction "
        "tiles (pred/*.png = hstack(raw, GT, pred)) at w//3. That is a rescoring "
        "of pre-existing outputs, not fresh inference: the tiles were produced "
        "with the repo's own per-dataset preprocessing (UFO in particular is "
        "240x320 cropped, not a whole-image 256x256 resize), and the checkpoint "
        "is the repo's variant_07 user-modified model whose publication status "
        "is unverified. The unified_psnr/ssim/n columns are therefore left blank "
        "and no method's gap is computed against this row. It is retained only "
        "because it is the one measurement that reproduces a recorded result "
        "(run result.txt 24.4629 dB vs 24.4762 re-derived) and thus corroborates "
        "the estimator itself."
    ),
}

# Caveats about the evaluation host that affected several methods.
HOST_NOTES = (
    "Host caveat: on this shared GPU node, device-side torch.fft raised "
    "CUFFT_INTERNAL_ERROR during our runs (reproduced with a length-8 1-D "
    "float32 transform in a fresh process). Spectral baselines (Phaseformer, "
    "SS-UIE, SFGNet, CDF-UIE, FDCE-Net) were therefore run with their "
    "transforms on CPU and the result moved back. The transforms are float32 "
    "on either device, so this is an execution-location change, not a "
    "numerical-protocol change -- but it does mean the wall-clock times in "
    "these artifacts must NOT be compared with published runtimes. "
    "torch.set_num_threads(8) was also pinned: the default 48 threads on a "
    "shared 96-core host made Phaseformer ~32x slower through oversubscription "
    "without changing any metric."
)

SSIM_NOTE = (
    "unified_ssim uses skimage.metrics.structural_similarity(win_size=5, "
    "channel_axis=-1, data_range=1.0) as the protocol specifies; this "
    "reproduces the MambaCue anchor (0.9224 recorded vs 0.9221 measured). "
    "The frozen code/obsui/metrics.py implements an 11x11 Gaussian instead and "
    "reads ~0.002-0.003 lower; that value is retained per-image in the "
    "artifact JSON as ssim_frozen11 for sensitivity checking. PSNR and MSE are "
    "identical under either estimator."
)


def _fmt(x, nd):
    if x is None:
        return ""
    try:
        return ("%." + str(nd) + "f") % float(x)
    except (TypeError, ValueError):
        return ""


def load_mambacue_artifact(dataset):
    """Adapt ``mambacue_verify.json`` to the shape ``build_rows`` expects.

    MambaCue's repo writes ``pred/*.png`` as ``hstack(raw, GT, pred)`` tiles rather
    than standalone predictions, so it is scored by `verify_mambacue.py` into a
    different artifact layout.  The numbers are re-derived from those official
    tiles under the same frozen protocol (whole image -> 256x256 bilinear, skimage
    PSNR/SSIM win_size=5); only the bookkeeping shape differs.

    This adapter exists so the frozen-SOTA row is a *measurement* in the registry
    rather than a blank, and so the recorded-vs-measured agreement (<=0.04 dB on
    all three datasets) travels with the number as evidence that the protocol is
    the original authors' own.
    """
    path = os.path.join(ART_ROOT, "mambacue_verify.json")
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    rec = raw.get(dataset)
    if not rec:
        return None
    return {
        "method": "MambaCue",
        "dataset": dataset,
        "n_scored": int(rec.get("n", 0)),
        "n_total": int(rec.get("n", 0)),
        "aggregate": {"psnr": rec.get("psnr"), "ssim": rec.get("ssim"),
                      "mse": rec.get("mse"),
                      "psnr_std": rec.get("psnr_std"),
                      "ssim_std": rec.get("ssim_std"), "n": rec.get("n")},
        "errors": [],
        "run": rec.get("run"),
        "recorded": rec.get("recorded"),
        "model_meta": {},
    }


def load_artifact(method_key, dataset):
    """Return the artifact dict for (method, dataset) or None."""
    if not method_key:
        return None
    if method_key == "mambacue":
        return load_mambacue_artifact(dataset)
    path = os.path.join(ART_ROOT, "baseline_%s_%s.json" % (method_key, dataset))
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def build_rows(only_measured=False):
    rows = []
    for method in METHOD_ORDER:
        key = METHOD_KEYS.get(method)
        for dataset in DATASETS:
            rec = {
                "method": method,
                "dataset": dataset,
                "reported_psnr": "",
                "reported_ssim": "",
                "unified_psnr": "",
                "unified_ssim": "",
                "unified_n": "",
                "unified_source": "not_run",
                "retrain_psnr": "",
                "retrain_ssim": "",
                "notes": "",
            }
            rep = REPORTED.get(method, {}).get(dataset)
            if rep:
                rec["reported_psnr"], rec["reported_ssim"] = rep[0], rep[1]

            notes = []
            art = load_artifact(key, dataset)
            if art is not None:
                agg = art.get("aggregate") or {}
                n_scored = art.get("n_scored") or agg.get("n") or 0
                n_total = art.get("n_total") or n_scored
                if n_scored:
                    rec["unified_psnr"] = _fmt(agg.get("psnr"), 4)
                    rec["unified_ssim"] = _fmt(agg.get("ssim"), 4)
                    rec["unified_n"] = str(n_scored)
                    rec["unified_source"] = "local_checkpoint_rerun_provenance_unverified"
                    if n_scored != n_total:
                        notes.append(
                            "ONLY %d of %d images produced a prediction; the "
                            "remaining %d failed and are listed in the "
                            "artifact's errors[]." % (n_scored, n_total, n_total - n_scored)
                        )
                else:
                    notes.append(
                        "inference produced no usable prediction for any of "
                        "the %d test images (see errors[] in the artifact)."
                        % n_total
                    )
                if art.get("model_meta", {}).get("load_missing"):
                    notes.append(
                        "checkpoint load: %s keys absent from the state_dict "
                        "(documented per-method below)."
                        % art["model_meta"]["load_missing"]
                    )
            elif key:
                notes.append("not measured: no artifact for this dataset.")

            if method in NOT_RUN_REASON:
                notes.append(NOT_RUN_REASON[method])
            if method == "MambaCue":
                # Rescoring saved tiles cannot establish their inference provenance.
                # In particular UFO tiles are 240x320, not the primary whole256
                # track. Keep measured legacy values in the original artifact.
                rec["unified_psnr"] = ""
                rec["unified_ssim"] = ""
                rec["unified_n"] = ""
                rec["unified_source"] = "saved_tiles_only_not_qualified_unified_rerun"
                notes.append(
                    "Saved-tile rescoring only, NOT a fresh checkpoint inference. "
                    "Legacy PSNR/SSIM: UIEB 24.4762/0.9221, LSUI 31.1957/0.9351, "
                    "UFO 28.5965/0.8780; artifacts/mambacue_verify.json preserves "
                    "full precision. UFO tiles are 240x320 and the source loader "
                    "uses a different preprocessing path. Numeric agreement with "
                    "result.txt does not establish canonical split or protocol "
                    "identity. Excluded from unified SOTA qualification."
                )
            if method in METHOD_NOTES:
                notes.append(METHOD_NOTES[method])

            rec["notes"] = " ".join(notes) if notes else ""
            if only_measured and not rec["unified_n"]:
                continue
            rows.append(rec)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out",
        default=os.path.join(DOCS_ROOT, "sota_registry.csv"),
        help="output CSV path (default docs/sota_registry.csv)",
    )
    ap.add_argument(
        "--only-measured",
        action="store_true",
        help="omit rows without a measurement (default: keep them, marked not_run)",
    )
    ap.add_argument(
        "--no-host-note",
        action="store_true",
        help="omit the shared host caveats from every row's notes",
    )
    args = ap.parse_args()

    rows = build_rows(only_measured=args.only_measured)
    if not rows:
        raise SystemExit("no rows produced -- did run_baseline.py run?")

    tail = []
    if not args.no_host_note:
        tail.append(HOST_NOTES)
    tail.append(SSIM_NOTE)
    tail_s = "\n\n".join(tail)

    # The host and estimator caveats apply to every row identically, so writing
    # them into each row's ``notes`` made the CSV 95% duplicated text and pushed
    # the one column a reader actually scans off the right edge of every line.
    # They live in one companion file instead, with a pointer in each row, so a
    # row can still be read on its own without carrying 2 kB of shared prose.
    protocol_path = os.path.join(os.path.dirname(args.out) or ".", "sota_protocol.md")
    if tail_s:
        with open(protocol_path, "w", encoding="utf-8") as fh:
            fh.write("# Shared protocol and host caveats for sota_registry.csv\n\n")
            fh.write("Every row of `sota_registry.csv` was produced under these "
                     "conditions; the `notes` column there records only what is "
                     "specific to that row.\n\n")
            fh.write(tail_s + "\n")

    pointer = "protocol: see sota_protocol.md" if tail_s else ""
    for rec in rows:
        rec["notes"] = " ".join(x for x in (rec.get("notes", ""), pointer) if x).strip()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        for rec in rows:
            w.writerow({k: rec.get(k, "") for k in COLUMNS})

    # console summary
    print("%-14s %-5s %10s %8s %10s %8s %5s  %s" % (
        "method", "ds", "reported", "ssim", "unified", "ssim", "n", "source"))
    for rec in rows:
        print("%-14s %-5s %10s %8s %10s %8s %5s  %s" % (
            rec["method"], rec["dataset"],
            rec["reported_psnr"] or "-", rec["reported_ssim"] or "-",
            rec["unified_psnr"] or "-", rec["unified_ssim"] or "-",
            rec["unified_n"] or "-", rec["unified_source"]))
    n_meas = sum(1 for r in rows if r["unified_n"])
    print("\n%d rows, %d measured, %d not_run -> %s" % (
        len(rows), n_meas, len(rows) - n_meas, args.out))


if __name__ == "__main__":
    main()

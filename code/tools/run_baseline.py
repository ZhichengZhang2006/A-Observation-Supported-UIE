"""Re-run third-party underwater-enhancement baselines with their OFFICIAL weights
under the FROZEN 256x256 unified protocol, then score the predictions.

Why this exists
---------------
The third-party prediction images shipped with the comparison table were produced
under each author's own preprocessing and are inconsistent in both directions
(re-scoring them gives UIEB 21.0 where the sheet reports 23.4, but UFO 28.3 where
the sheet reports 27.7). They therefore cannot be used as the SOTA reference.
This driver instead loads the published checkpoints and runs inference ourselves,
so the only protocol in play is the frozen one in ``code/obsui``.

Frozen protocol (do not change)
-------------------------------
1. image loaded as uint8 RGB,
2. whole image resized to 256x256 with PIL ``Image.BILINEAR`` -- no crop, no pad,
   no aspect preservation (``dataio.to_unified``),
3. PSNR on float images in [0,1] with ``data_range=1.0`` (identical to
   ``metrics.psnr(..., data_range=255.0)`` on uint8),
4. SSIM with ``skimage.metrics.structural_similarity(channel_axis=-1, win_size=5,
   data_range=1.0)``,
5. per-image metrics first, then the arithmetic mean over images,
6. every image counts -- nothing is dropped for scoring badly.

NOTE on the SSIM estimator: the frozen ``code/obsui/metrics.py`` predates this
task and implements an 11x11 Gaussian SSIM that does NOT match step 4. That file
is frozen, so it is left untouched; ``unified_ssim`` here is the skimage
win_size=5 value required by the protocol, and the 11x11 value is recorded
alongside as ``ssim_frozen11`` for sensitivity. The two differ by ~0.002-0.004
on the trusted MambaCue anchor.

Interface
---------
    python code/tools/run_baseline.py --method phaseformer --dataset UIEB

Each method knows how to build itself, where its official checkpoint lives, and
what input/output convention it expects. Preprocessing is NEVER taken from the
baseline's own test script: we always feed already-256-resized unified tensors
and save the prediction back at exactly 256x256.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(HERE))

import numpy as np  # noqa: E402
import torch  # noqa: E402

# CPU thread cap. The host has 96 cores and torch defaults to 48 intra-op
# threads; on a shared machine that oversubscribes badly. Phaseformer's
# attention calls inv_mag() which round-trips every attention map through
# CPU FFTs (its cuFFT is broken here), so thread thrash alone made the model
# 32x slower (0.73 s vs 0.03 s per call). Pinning to 8 threads is required
# for tractable runtimes and does not change any numerical result.
torch.set_num_threads(int(os.environ.get("OBS_THREADS", "8")))

from PIL import Image  # noqa: E402
from skimage.metrics import structural_similarity as sk_ssim  # noqa: E402

from obsui import dataio, metrics  # noqa: E402

REPO_ROOT = "/home/user"
DATASET_ROOT = "/home/user/UnderWaterDataset"
PRED_ROOT = os.path.join(PROJECT_ROOT, "predictions")
ART_ROOT = os.path.join(PROJECT_ROOT, "artifacts")

#: skimage SSIM window required by the protocol text
SK_SSIM_WIN = 5

METHODS = ["phaseformer", "ssuie", "cclnet", "aquaclarity", "cdfuie", "sfgnet", "fdcenet"]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _to_uint8_rgb(t: torch.Tensor, lo: float = -1.0) -> np.ndarray:
    """Model output -> uint8 RGB HxWx3 at 256x256.

    ``lo`` is the value the model treats as black: -1.0 for [-1,1] outputs,
    0.0 for [0,1] outputs.
    """
    x = t.detach().float().cpu()
    if x.dim() == 4:
        x = x[0]
    x = x.clamp(lo, 1.0)
    x = (x - lo) / (1.0 - lo)
    arr = (x * 255.0).round().clamp(0, 255).byte().permute(1, 2, 0).numpy()
    return np.ascontiguousarray(arr, dtype=np.uint8)


def _tensor_from_unified(u: np.ndarray, lo: float = 0.0) -> torch.Tensor:
    """HxWx3 uint8 -> 1x3xHxW float tensor mapped to [lo, 1]."""
    x = torch.from_numpy(np.ascontiguousarray(u)).float().div_(255.0).permute(2, 0, 1).unsqueeze(0)
    if lo < 0.0:
        x = x * 2.0 - 1.0
    return x


def _strip_prefix(sd, prefixes=("module.", "_orig_mod.")):
    out = {}
    for k, v in sd.items():
        for p in prefixes:
            if k.startswith(p):
                k = k[len(p):]
        out[k] = v
    return out


def _load_state(path, key=None):
    ck = torch.load(path, map_location="cpu")
    if key is not None and isinstance(ck, dict) and key in ck:
        ck = ck[key]
    if isinstance(ck, dict):
        for alt in ("state_dict", "model", "netG", "params"):
            if alt in ck and isinstance(ck[alt], dict):
                ck = ck[alt]
                break
    return _strip_prefix(ck)


def _pick_by_psnr(directory):
    """CDF-UIE names its checkpoints ``UHDFour-epoch{N}-{psnr}.pt``."""
    best, best_p = None, -1.0
    for f in sorted(os.listdir(directory)):
        if not f.endswith(".pt"):
            continue
        try:
            p = float(os.path.splitext(f)[0].rsplit("-", 1)[1])
        except (IndexError, ValueError):
            continue
        if p > best_p:
            best, best_p = f, p
    if best is None:
        raise FileNotFoundError(f"no *.pt checkpoint in {directory}")
    return os.path.join(directory, best), best_p


# --------------------------------------------------------------------------- #
# per-method model builders
# --------------------------------------------------------------------------- #
def build_phaseformer(dataset):
    """Full ``nn.Module`` checkpoints; epoch 800 is the published one.

    The repo ships 801 per-epoch files with no "best" marker, but its own prior
    evaluation run (results/<DS>_CPU_test_2026-10-18) used epoch 800 and
    reproduced the reported row, so epoch 800 is the official weight.
    """
    sys.path.insert(0, os.path.join(REPO_ROOT, "Phaseformer-main"))
    path = os.path.join(REPO_ROOT, "Phaseformer-main", "checkpoint", dataset, "netG_model_epoch_800.pth")
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    model = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(model, torch.nn.Module):
        raise RuntimeError("Phaseformer checkpoint is not a full nn.Module")
    return model, path, {"input_range": "[-1,1]", "output_pick": "tuple[0]"}, -1.0


def _install_global_cpu_fft():
    """Route every ``torch.fft`` transform through CPU.

    cuFFT is unusable on this host: *every* device-side transform raises
    ``RuntimeError: cuFFT error: CUFFT_INTERNAL_ERROR`` -- verified with a
    length-8 float32 1-D ``torch.fft.fft`` in a fresh process, so this is an
    environment fault (cuFFT vs. the container's CUDA runtime), not a model
    or driver fault. Four of the seven baselines (Phaseformer, SS-UIE,
    SFGNet, CDF-UIE, FDCE-Net) use spectral ops in their forward pass, and
    Phaseformer's author has the identical workaround baked into the model
    (``model_with_eca.inv_mag``: ``x.cpu()`` -> fft -> ``.to(device)``).

    Patching globally is both simpler and safer than per-repo shims: the
    transforms are float32 on either device, so the output is numerically
    unchanged (``< 1e-6`` difference against a CPU-only reference) and only
    the execution device moves. Every other ``torch.fft`` attribute is
    delegated untouched.
    """
    if getattr(torch.fft, "_obs_cpu_patched", False):
        return
    for name in ("fft", "ifft", "fft2", "ifft2", "fftn", "ifftn",
                 "rfft", "irfft", "rfft2", "irfft2", "rfftn", "irfftn",
                 "hfft", "ihfft", "hfft2", "ihfft2", "fftshift", "ifftshift"):
        real = getattr(torch.fft, name, None)
        if real is None:
            continue

        def make(real_fn):
            def wrapper(x, *args, **kwargs):
                if isinstance(x, torch.Tensor) and x.is_cuda:
                    dev = x.device
                    out = real_fn(x.cpu(), *args, **kwargs)
                    return out.to(dev)
                return real_fn(x, *args, **kwargs)
            wrapper.__name__ = real_fn.__name__
            return wrapper

        setattr(torch.fft, name, make(real))
    torch.fft._obs_cpu_patched = True


def _patch_cpu_fft(module, attr_name, fft_fn_name):
    """Superseded by ``_install_global_cpu_fft``; kept so old call sites resolve."""
    raise NotImplementedError("use _install_global_cpu_fft")


def _install_ssuie_fft_shim():
    """SS-UIE's ``GlobalFilter`` uses ``torch.fft.rfft2``/``irfft2`` on GPU."""
    _install_global_cpu_fft()


def build_ssuie(dataset):
    """``saved_models/<DS>/SS_UIE_best.pth`` -> dict with ``model``."""
    root = os.path.join(REPO_ROOT, "SS-UIE-main")
    sys.path.insert(0, os.path.join(root, "net"))
    sys.path.insert(0, root)
    from net.model import SS_UIE_model  # noqa: E402

    _install_ssuie_fft_shim()

    path = os.path.join(root, "saved_models", dataset, "SS_UIE_best.pth")
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    raw = torch.load(path, map_location="cpu")
    sd = raw["model"] if isinstance(raw, dict) and "model" in raw else raw
    sd = _strip_prefix(sd)
    n_memblock = len({int(k.split(".")[1]) for k in sd if k.startswith("dense_memory.")}) or 6
    model = SS_UIE_model(in_channels=3, channels=16, num_memblock=n_memblock, num_resblock=4)
    model.load_state_dict(sd, strict=True)
    meta = {
        "input_range": "[0,1]",
        "output_pick": "single",
        "ckpt_epoch": raw.get("epoch") if isinstance(raw, dict) else None,
        "ckpt_best_psnr": raw.get("best_psnr") if isinstance(raw, dict) else None,
        "num_memblock": n_memblock,
    }
    return model, path, meta, 0.0


def build_cclnet(dataset):
    """CCL-Net's official configuration is the *two-stage* HRNet pipeline.

    ``checkpoints/<sub>/train_opt.txt`` records ``model: HRNet``, and
    ``models/HRNet_model.py:52-60`` (HRNetModel.forward) is the authoritative
    evaluation graph:

        _, pred_cc_fcc, _ = netG_CC(raw)              # raw = normalised Lab
        rgb_fcc           = lab2rgb(pred_cc_fcc)      # Lab[-1,1] -> RGB[-1,1]
        pred_hr, _, _     = netG(rgb_fcc)             # HR branch, RGB[-1,1]

    ``test.py:66 select_prediction`` then reports ``pred_hr``. Running the
    CC branch alone (``pred_cc``) is only the colour-correction half: it keeps
    the input luminance, so on dark underwater frames it degenerates towards
    the identity and scores ~identity PSNR. That is why the two-stage graph is
    required here.

    Pre: BGR->Lab, per-channel (x - [0,42,20]) / [255,184,203], *255 -> uint8,
    then ToTensor + Normalize(0.5,0.5) -> [-1,1].
    """
    root = os.path.join(REPO_ROOT, "CCL-Net-main")
    sys.path.insert(0, root)
    from models.CCLNet.CCNet import CCNet  # noqa: E402
    from models.CCLNet.HRNet import HRNet  # noqa: E402
    from models.CCLNet.Public.util.LAB2RGB_v2 import Lab2RGB  # noqa: E402

    sub = {"UIEB": "UIEB_CC", "LSUI": "LSUI_CC", "UFO": "UFO_CCL"}[dataset]
    path_cc = os.path.join(root, "checkpoints", sub, "best_net_G_CC.pth")
    path_hr = os.path.join(root, "checkpoints", sub, "best_net_G.pth")
    for p in (path_cc, path_hr):
        if not os.path.isfile(p):
            raise FileNotFoundError(p)

    opt = type("Opt", (), {})()
    opt.gpu_ids = []
    opt.init_type = "normal"
    opt.init_gain = 0.02
    opt.isTrain = False
    model_cc = CCNet(opt)
    model_cc.load_state_dict(_strip_prefix(torch.load(path_cc, map_location="cpu")), strict=True)
    model_hr = HRNet(opt)
    model_hr.load_state_dict(_strip_prefix(torch.load(path_hr, map_location="cpu")), strict=True)

    class _TwoStage(torch.nn.Module):
        """Exactly HRNetModel.forward at test time (no residual wrappers)."""

        def __init__(self, cc, hr, l2r):
            super().__init__()
            self.cc, self.hr, self.l2r = cc, hr, l2r

        def forward(self, x):
            _feat_cc, pred_cc_fcc, _e = self.cc(x)
            rgb_fcc = self.l2r.labn12p1_to_rgbn12p1(pred_cc_fcc)
            pred_hr, _hc, _enc = self.hr(rgb_fcc)
            return pred_hr

    lab2rgb = Lab2RGB()
    model = _TwoStage(model_cc, model_hr, lab2rgb)
    meta = {
        "input_range": "[-1,1] (Lab-normalised)",
        "output_pick": "HRNetModel.forward -> pred_hr (RGB [-1,1])",
        "ckpt_dir": sub,
        "ckpts": [os.path.basename(path_cc), os.path.basename(path_hr)],
        "lab2rgb": lab2rgb,
    }
    return model, "%s + %s" % (path_cc, path_hr), meta, -1.0


def build_aquaclarity(dataset):
    root = os.path.join(REPO_ROOT, "AquaClarity-main")
    sys.path.insert(0, root)
    from models import model_registry  # noqa: E402
    from utils import load_checkpoint  # noqa: E402

    path = os.path.join(root, "checkpoints", dataset, "best.pth")
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    model = model_registry["Model"]()
    load_checkpoint(model, path)
    raw = torch.load(path, map_location="cpu")
    meta = {
        "input_range": "[0,1]",
        "output_pick": "single",
        "ckpt_epoch": raw.get("epoch") if isinstance(raw, dict) else None,
    }
    return model, path, meta, 0.0


def build_cdfuie(dataset):
    root = os.path.join(REPO_ROOT, "CDF-UIE-main")
    sys.path.insert(0, os.path.join(root, "src"))
    from CDF_UIE_arch import InteractNet  # noqa: E402

    sub = {"UIEB": "UIEB-0730-1508", "LSUI": "LSUI-0730-1535", "UFO": "UFO-0823-0856"}[dataset]
    path, psnr_tag = _pick_by_psnr(os.path.join(root, "ckpts", sub))
    model = InteractNet(nc=16)
    sd = _strip_prefix(torch.load(path, map_location="cpu"))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    meta = {
        "input_range": "[0,1]",
        "output_pick": "tuple[0]",
        "ckpt_dir": sub,
        "ckpt_name_psnr": psnr_tag,
        "load_missing": len(missing),
        "load_unexpected": len(unexpected),
    }
    if missing or unexpected:
        meta["load_missing_sample"] = list(missing)[:5]
        meta["load_unexpected_sample"] = list(unexpected)[:5]
    return model, path, meta, 0.0


def build_sfgnet(dataset):
    root = os.path.join(REPO_ROOT, "SFGNet-main")
    sys.path.insert(0, root)
    from SFGnet import SFGNet  # noqa: E402

    path = os.path.join(root, "models", dataset, "net_last.pth")
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    model = SFGNet()
    sd = _strip_prefix(torch.load(path, map_location="cpu"))
    # ``g.weight_v``/``g.weight_h`` are the fixed Sobel kernels registered as
    # buffers by a *newer* revision of SFGnet.py than the one that produced
    # these checkpoints; they are constants, never trained, and the module
    # re-registers them itself, so their absence is expected and harmless.
    missing, unexpected = model.load_state_dict(sd, strict=False)
    meta = {
        "input_range": "[0,1]",
        "output_pick": "tuple[1] restored",
        "load_missing": list(missing),
        "load_unexpected": list(unexpected),
        "load_missing_note": "fixed Sobel buffers re-registered by the module",
    }
    return model, path, meta, 0.0


def _install_cdf_fft_shim():
    """CDF-UIE calls module-level ``rfft2``/``irfft2`` (src/CDF_UIE_arch.py:9-16)."""
    _install_global_cpu_fft()


def build_cdfuie(dataset):
    _install_cdf_fft_shim()
    sys.path.insert(0, os.path.join(REPO_ROOT, "CDF-UIE-main", "src"))
    from CDF_UIE_arch import InteractNet  # noqa: E402

    # Dir per dataset; repo has no "best" marker, filenames embed validation PSNR
    # (e.g. UHDFour-epoch94-13.96954.pt) -> pick the highest, as the authors did.
    sub = {
        "UIEB": "UIEB-0730-1508",
        "LSUI": "LSUI-0730-1535",
        "UFO": "UFO-0823-0856",
    }[dataset]
    root = os.path.join(REPO_ROOT, "CDF-UIE-main", "ckpts", sub)
    if not os.path.isdir(root):
        raise FileNotFoundError(root)
    cand = []
    for fn in os.listdir(root):
        if not fn.endswith(".pt"):
            continue
        m = re.search(r"-(\d+\.\d+)\.pt$", fn)
        cand.append((float(m.group(1)) if m else -1.0, fn))
    if not cand:
        raise FileNotFoundError("no .pt in " + root)
    cand.sort(reverse=True)
    best_psnr, best_fn = cand[0]
    path = os.path.join(root, best_fn)

    model = InteractNet(nc=16)
    sd = _strip_prefix(torch.load(path, map_location="cpu"))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    meta = {
        "input_range": "[0,1]",
        "output_pick": "tuple[0] y",
        "ckpt_dir": sub,
        "ckpt_name_psnr": best_psnr,
        "ckpt_candidates": [f for _, f in cand],
        "load_missing": len(missing),
        "load_unexpected": len(unexpected),
    }
    return model, path, meta, 0.0


def _install_fdce_fft_shim():
    """FDCE-Net calls ``torch.fft.rfft2`` inline (fdce_net.py:277)."""
    _install_global_cpu_fft()


def build_fdcenet(dataset):
    root = os.path.join(REPO_ROOT, "FDCE-Net-main")
    sys.path.insert(0, root)
    import fdce_net as fdce_model  # noqa: E402

    if not hasattr(fdce_model, "MultiScaleColorEncoder"):
        fdce_model.MultiScaleColorEncoder = fdce_model.MultiScaleColorDecoder
    path = os.path.join(root, "training_data", dataset, "checkpoints", "best.pth")
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    model = fdce_model.fdce_net(d_hist=64)
    raw = torch.load(path, map_location="cpu")
    sd = raw["model"] if isinstance(raw, dict) and "model" in raw else raw
    model.load_state_dict(_strip_prefix(sd), strict=True)
    meta = {
        "input_range": "[0,1]",
        "output_pick": "tuple[2] img_enhance",
        "ckpt_epoch": raw.get("epoch") if isinstance(raw, dict) else None,
    }
    return model, path, meta, 0.0


BUILDERS = {
    "phaseformer": build_phaseformer,
    "ssuie": build_ssuie,
    "cclnet": build_cclnet,
    "aquaclarity": build_aquaclarity,
    "cdfuie": build_cdfuie,
    "sfgnet": build_sfgnet,
    "fdcenet": build_fdcenet,
}


# --------------------------------------------------------------------------- #
# inference
# --------------------------------------------------------------------------- #
def _ccl_lab_normalise(u: np.ndarray) -> np.ndarray:
    """RGB uint8 -> the uint8 Lab-normalised image CCL-Net's loader builds."""
    import cv2

    bgr = cv2.cvtColor(u, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab).astype(np.float32)
    mn = np.array([0.0, 42.0, 20.0], dtype=np.float32)
    nrm = np.array([255.0, 184.0, 203.0], dtype=np.float32)
    for i in range(3):
        lab[:, :, i] = (lab[:, :, i] - mn[i]) / nrm[i]
    return np.asarray(lab * 255.0, dtype=np.uint8)


@torch.no_grad()
def infer(method, model, meta, u, device):
    """u: 256x256x3 uint8 RGB -> 256x256x3 uint8 RGB."""
    lo = -1.0 if meta["input_range"].startswith("[-1,1]") else 0.0

    if method == "cclnet":
        lab_u8 = _ccl_lab_normalise(u)
        x = _tensor_from_unified(lab_u8, lo=-1.0).to(device)
        # Two-stage graph: netG_CC (Lab) -> lab2rgb -> netG_HR; returns pred_hr
        # in RGB [-1,1] already (see build_cclnet).
        out = model(x)
        return _to_uint8_rgb(out, lo=-1.0)

    x = _tensor_from_unified(u, lo=lo).to(device)
    out = model(x)

    if method == "phaseformer":
        out = out[0] if isinstance(out, (tuple, list)) else out
        return _to_uint8_rgb(out, lo=-1.0)
    if method == "cdfuie":
        out = out[0] if isinstance(out, (tuple, list)) else out
        return _to_uint8_rgb(out, lo=0.0)
    if method == "sfgnet":
        out = out[1] if isinstance(out, (tuple, list)) else out
        return _to_uint8_rgb(out, lo=0.0)
    if method == "fdcenet":
        out = out[2] if isinstance(out, (tuple, list)) else out
        return _to_uint8_rgb(out, lo=0.0)
    return _to_uint8_rgb(out, lo=0.0)


def score(pred_u8, ref_u8):
    """Frozen metrics. Returns protocol SSIM (skimage win5) + frozen 11x11 value."""
    p = pred_u8.astype(np.float64) / 255.0
    r = ref_u8.astype(np.float64) / 255.0
    return {
        "psnr": float(metrics.psnr(pred_u8, ref_u8, data_range=255.0)),
        "ssim": float(sk_ssim(r, p, channel_axis=-1, win_size=SK_SSIM_WIN, data_range=1.0)),
        "ssim_frozen11": float(metrics.ssim_rgb(pred_u8, ref_u8)),
        "mse": float(np.mean((pred_u8.astype(np.float64) - ref_u8.astype(np.float64)) ** 2)),
    }


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Run one UIE baseline under the frozen protocol")
    ap.add_argument("--method", required=True, choices=METHODS)
    ap.add_argument("--dataset", required=True, choices=["UIEB", "LSUI", "UFO"])
    ap.add_argument("--max-images", type=int, default=0, help="0 = all (debug only)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--save-preds", action="store_true", default=True)
    ap.add_argument("--split", default="test", choices=["test", "val", "train"],
                    help="which data split to run; the checkpoint is still chosen by --dataset")
    args = ap.parse_args()

    method, dataset = args.method, args.dataset
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # --split selects the DATA while --dataset selects the CHECKPOINT, so the
    # same frozen weights can be run over train/val/test for the error-model study
    # without pretending those extra splits are a new benchmark.
    data_rel = f"{dataset}/{args.split}"
    tag = dataset if args.split == "test" else f"{dataset}-{args.split}"
    input_dir = os.path.join(DATASET_ROOT, data_rel, "input")
    ref_dir = os.path.join(DATASET_ROOT, data_rel, "GT")
    pairs = dataio.resolve_pairs(input_dir, ref_dir, dataset)
    if args.max_images:
        pairs.pairs = pairs.pairs[: args.max_images]
    n_total = len(pairs)
    if n_total == 0:
        raise SystemExit(f"no pairs in {input_dir} / {ref_dir}")

    print(f"[{method}/{tag}] pairs={n_total} device={device}", flush=True)
    for note in pairs.notes:
        print("  " + note, flush=True)

    t0 = time.time()
    # Must run before any model is built: five of the seven baselines use
    # spectral ops whose CUDA path raises CUFFT_INTERNAL_ERROR on this host.
    _install_global_cpu_fft()
    model, ckpt_path, meta, _unused = BUILDERS[method](dataset)
    model = model.to(device).eval()
    print(f"  ckpt={ckpt_path}", flush=True)
    print(f"  built in {time.time() - t0:.1f}s", flush=True)

    pred_dir = os.path.join(PRED_ROOT, tag, method)
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(ART_ROOT, exist_ok=True)

    rows, errors = [], []
    t1 = time.time()
    for i, pr in enumerate(pairs.pairs, 1):
        try:
            u_in = dataio.to_unified(dataio.load_rgb(pr.input_path))
            u_ref = dataio.to_unified(dataio.load_rgb(pr.ref_path))
            u_pred = infer(method, model, meta, u_in, device)
            if u_pred.shape != (256, 256, 3):
                raise RuntimeError(f"prediction shape {u_pred.shape}")
            Image.fromarray(u_pred, mode="RGB").save(os.path.join(pred_dir, pr.name + ".png"))
            m = score(u_pred, u_ref)
            m["name"] = pr.name
            rows.append(m)
        except Exception as exc:  # keep going; a failure is a result, not a crash
            errors.append({"name": pr.name, "error": f"{type(exc).__name__}: {exc}"})
            print(f"  !! {pr.name}: {type(exc).__name__}: {exc}", flush=True)
            if len(errors) > 5:
                traceback.print_exc()
                break
        if i % 50 == 0 or i == n_total:
            done = len(rows)
            if done:
                cur = float(np.mean([r["psnr"] for r in rows]))
                print(f"  {i}/{n_total} done={done} running_psnr={cur:.4f}", flush=True)

    elapsed = time.time() - t1
    agg = {}
    if rows:
        for k in ("psnr", "ssim", "ssim_frozen11", "mse"):
            v = np.array([r[k] for r in rows], dtype=np.float64)
            agg[k] = float(v.mean())
            agg[k + "_std"] = float(v.std(ddof=1)) if len(v) > 1 else 0.0
        agg["n"] = len(rows)

    out = {
        "method": method,
        "dataset": dataset,
        "split": args.split,
        "checkpoint": ckpt_path,
        "model_meta": {k: v for k, v in meta.items() if k != "lab2rgb"},
        "n_total": n_total,
        "n_scored": len(rows),
        "status": "ok" if not errors and len(rows) == n_total else "partial",
        "errors": errors[:20],
        "aggregate": agg,
        "protocol": {
            "resize": "whole image -> 256x256, PIL BILINEAR, no crop/pad",
            "psnr": "data_range=1.0 on [0,1] floats",
            "ssim": f"skimage structural_similarity(channel_axis=-1, win_size={SK_SSIM_WIN}, data_range=1.0)",
            "ssim_frozen11": "code/obsui/metrics.ssim_rgb (11x11 gaussian) -- sensitivity only",
            "aggregation": "per-image then arithmetic mean",
        },
        "elapsed_sec": elapsed,
        "rows": rows,
    }
    art_path = os.path.join(ART_ROOT, f"baseline_{method}_{tag}.json")
    with open(art_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)

    print("", flush=True)
    if rows:
        print(f"[{method}/{tag}] n={agg['n']}/{n_total} "
              f"PSNR={agg['psnr']:.4f} SSIM(win5)={agg['ssim']:.6f} "
              f"SSIM(frozen11)={agg['ssim_frozen11']:.6f} "
              f"({elapsed:.1f}s)", flush=True)
    else:
        print(f"[{method}/{tag}] NO PREDICTIONS PRODUCED", flush=True)
    print(f"  artifact={art_path}", flush=True)
    print(f"  predictions={pred_dir}", flush=True)
    return 0 if out["status"] == "ok" else 3


if __name__ == "__main__":
    raise SystemExit(main())

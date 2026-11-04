"""Verify MambaCue-recorded metrics by recomputing from its saved (raw|GT|pred) tiles.

Locked protocol (matches /home/user/MambaCue/utils/metrics.py + utils/dataset.py):
  PIL Image.BILINEAR resize to (256,256)  [UIEB, LSUI]; center-crop 256 [UFO, EUVP]
  skimage PSNR data_range=1.0 ; SSIM channel_axis=-1 win_size=5 ; per-image mean
"""
import os, sys, glob, json
import numpy as np
import cv2
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim

RUNS = {
    "UIEB": "/home/user/MambaCue/output/L1+SSIM+Prep+Lab+grad+Angle/four/color-illumination-contrast/both/UIEB/20261009_092226",
    "LSUI": "/home/user/MambaCue/output/L1+SSIM+Prep+Lab+grad+Angle/four/color-illumination-contrast/both/LSUI/20261009_112952",
    "UFO":  "/home/user/MambaCue/output/L1+SSIM+Prep+Lab+grad+Angle/four/color-illumination-contrast/both/UFO/20261010_020343",
}


def read_tile(path):
    """Tile is hstack(raw, GT, pred) at 256x768 (or 256x256*3 after crop)."""
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    h, w, _ = img.shape
    assert w % 3 == 0, f"unexpected tile shape {img.shape} for {path}"
    t = w // 3
    return img[:, :t], img[:, t:2 * t], img[:, 2 * t:3 * t]


def main():
    out = {}
    for ds, run in RUNS.items():
        pred_dir = os.path.join(run, "pred")
        if not os.path.isdir(pred_dir):
            out[ds] = {"error": "no pred dir", "run": run}
            continue
        files = sorted(glob.glob(os.path.join(pred_dir, "*")))
        ps, ss, ms = [], [], []
        for f in files:
            _, gt, pred = read_tile(f)
            ps.append(float(sk_psnr(gt, pred, data_range=1.0)))
            ss.append(float(sk_ssim(gt, pred, data_range=1.0, channel_axis=-1, win_size=5)))
            ms.append(float(np.mean(((pred - gt) * 255.0) ** 2)))
        rec = {}
        rt = os.path.join(run, "result.txt")
        if os.path.exists(rt):
            rec["raw"] = open(rt).read().strip()
        out[ds] = {
            "run": run, "n": len(files),
            "psnr": float(np.mean(ps)), "ssim": float(np.mean(ss)), "mse": float(np.mean(ms)),
            "psnr_std": float(np.std(ps)), "ssim_std": float(np.std(ss)),
            "recorded": rec,
        }
        print(f"{ds}: n={len(files)} PSNR={np.mean(ps):.4f} SSIM={np.mean(ss):.4f} MSE={np.mean(ms):.4f}")
        print(f"   recorded: {rec.get('raw','<none>')}")
    os.makedirs("/home/user/obsui/artifacts", exist_ok=True)
    json.dump(out, open("/home/user/obsui/artifacts/mambacue_verify.json", "w"), indent=2)


if __name__ == "__main__":
    main()

"""Metric implementation, frozen once, used by EVERY method in this study.

Protocol (preregistered, see research_plan.md):
  * every image is resized so min(H,W) -> 256 with a single fixed resampling
    (INTER_AREA when downscaling, INTER_LINEAR when upscaling), no crop, no pad
  * RGB, uint8 in [0,255] at I/O; float64 for accumulation
  * SSIM: 11x11 Gaussian window, sigma=1.5, data_range=255, K1=0.01, K2=0.03,
    'valid' convolution (no padding), channel-averaged
  * per-image metric first, then arithmetic mean over images (NOT mean-of-MSE)
  * no image is dropped for any reason, including low confidence
"""
from __future__ import annotations

import numpy as np
import cv2

TARGET_MIN_SIDE = 256
SSIM_WIN = 11
SSIM_SIGMA = 1.5
SSIM_K1 = 0.01
SSIM_K2 = 0.03


def resize_unified(img: np.ndarray, min_side: int = TARGET_MIN_SIDE) -> np.ndarray:
    """Resize so that min(H, W) == min_side, preserving aspect ratio."""
    h, w = img.shape[:2]
    if min(h, w) == min_side:
        return img
    scale = min_side / float(min(h, w))
    nh, nw = int(round(h * scale)), int(round(w * scale))
    nh, nw = max(nh, 1), max(nw, 1)
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(img, (nw, nh), interpolation=interp)


def _gaussian_window(win: int = SSIM_WIN, sigma: float = SSIM_SIGMA) -> np.ndarray:
    g = cv2.getGaussianKernel(win, sigma)
    return np.outer(g, g)


def ssim_channel(a: np.ndarray, b: np.ndarray,
                 win: int = SSIM_WIN, sigma: float = SSIM_SIGMA,
                 data_range: float = 255.0) -> float:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    w = _gaussian_window(win, sigma)
    c1 = (SSIM_K1 * data_range) ** 2
    c2 = (SSIM_K2 * data_range) ** 2
    mu_a = cv2.filter2D(a, -1, w, borderType=cv2.BORDER_REFLECT)
    mu_b = cv2.filter2D(b, -1, w, borderType=cv2.BORDER_REFLECT)
    mu_a2, mu_b2, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b
    va = cv2.filter2D(a * a, -1, w, borderType=cv2.BORDER_REFLECT) - mu_a2
    vb = cv2.filter2D(b * b, -1, w, borderType=cv2.BORDER_REFLECT) - mu_b2
    cab = cv2.filter2D(a * b, -1, w, borderType=cv2.BORDER_REFLECT) - mu_ab
    s = ((2 * mu_ab + c1) * (2 * cab + c2)) / ((mu_a2 + mu_b2 + c1) * (va + vb + c2))
    return float(s.mean())


def psnr(a: np.ndarray, b: np.ndarray, data_range: float = 255.0) -> float:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    mse = float(np.mean((a - b) ** 2))
    if mse <= 0.0:
        return 99.0
    return float(10.0 * np.log10(data_range * data_range / mse))


def ssim_rgb(a: np.ndarray, b: np.ndarray) -> float:
    if a.ndim == 2:
        return ssim_channel(a, b)
    return float(np.mean([ssim_channel(a[..., c], b[..., c]) for c in range(a.shape[2])]))


def evaluate_pair(pred: np.ndarray, ref: np.ndarray) -> dict:
    """pred/ref: uint8 RGB (or gray). Applies the unified resize to BOTH."""
    assert pred.shape == ref.shape, f"shape mismatch {pred.shape} vs {ref.shape}"
    p = resize_unified(pred)
    r = resize_unified(ref)
    return {"psnr": psnr(p, r), "ssim": ssim_rgb(p, r), "mse": float(np.mean((p.astype(np.float64) - r.astype(np.float64)) ** 2))}


def aggregate(rows: list) -> dict:
    """rows: list of dicts with psnr/ssim/mse. Equal weight per image."""
    if not rows:
        return {"n": 0}
    out = {"n": len(rows)}
    for k in ("psnr", "ssim", "mse"):
        if k in rows[0]:
            v = np.array([r[k] for r in rows], dtype=np.float64)
            out[k] = float(v.mean())
            out[k + "_std"] = float(v.std(ddof=1)) if len(v) > 1 else 0.0
            out[k + "_median"] = float(np.median(v))
    return out


def relative_gap(model: float, ref: float, higher_better: bool = True) -> float:
    """g_m = max(0, (S_m - R_m)/S_m) for higher-better; max(0,(R_m-S_m)/S_m) otherwise."""
    if higher_better:
        return max(0.0, (ref - model) / ref)
    return max(0.0, (model - ref) / ref)


def _selftest():
    rng = np.random.RandomState(0)
    a = rng.randint(0, 256, (300, 400, 3), dtype=np.uint8)
    assert psnr(a, a) > 90
    assert abs(ssim_rgb(a, a) - 1.0) < 1e-6
    # shift inside a range that cannot clip, so MSE is exactly 25
    a0 = rng.randint(0, 200, (300, 400, 3), dtype=np.uint8)
    b = (a0.astype(np.int16) + 5).astype(np.uint8)
    p = psnr(a0, b)
    s = ssim_rgb(a0, b)
    # closed form: MSE = 25 over all channels -> 10*log10(65025/25)
    assert abs(p - 10 * np.log10(65025 / 25)) < 1e-6, p
    assert 0.5 < s < 1.0, s
    assert resize_unified(a).shape[:2] == (256, 341), resize_unified(a).shape
    assert relative_gap(28.6, 30.0) > 0.045
    assert relative_gap(30.5, 30.0) == 0.0
    print("metrics selftest OK")


if __name__ == "__main__":
    _selftest()

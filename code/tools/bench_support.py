"""Where does fit_water spend its time?  Run before optimizing anything."""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import obsui.support as S  # noqa: E402

rng = np.random.default_rng(0)
h = w = 128
yy, xx = np.mgrid[0:h, 0:w] / h
J = np.clip(np.stack([0.35 + 0.3 * xx, 0.45 + 0.2 * yy, 0.55 - 0.25 * xx], -1), 0.02, 1.0)
d = 0.25 + 0.5 * (0.5 + 0.5 * np.sin(2 * np.pi * yy))
wm = S.WaterModel(np.array([0.9, 0.6, 0.45]), np.ones(3), np.array([0.1, 0.18, 0.25]))
I = np.clip(S.render(J.reshape(-1, 3), d.reshape(-1, 1) * wm.beta.reshape(1, 3), wm), 0, 1)
I = I.reshape(h, w, 3) + rng.normal(0, 0.006, (h, w, 3))
I = np.clip(I, 0, 1)

Iw = S._resize_f(I, h, w, 64).reshape(-1, 3)
Jw = S._resize_f(J, h, w, 64).reshape(-1, 3)
phi = S.depth_basis(64, 64, 9)
sig = S.noise_sigma(Iw)

print(f"grid {Iw.shape}  phi {phi.shape}")

t0 = time.perf_counter()
for _ in range(200):
    S._implied_tau(Iw, Jw, wm)
print(f"_implied_tau     x200 : {time.perf_counter() - t0:.3f}s")

t0 = time.perf_counter()
for _ in range(200):
    S._sigma_tau(Iw, wm, sig)
print(f"_sigma_tau       x200 : {time.perf_counter() - t0:.3f}s")

t0 = time.perf_counter()
for _ in range(200):
    S._concentrated(Iw, Jw, wm, phi, sig)
dt = time.perf_counter() - t0
print(f"_concentrated    x200 : {dt:.3f}s  ({dt / 200 * 1000:.2f} ms each)")

A = phi * np.sqrt(np.maximum(np.ones(phi.shape[0]), 0)).reshape(-1, 1)
b = np.ones(phi.shape[0])
t0 = time.perf_counter()
for _ in range(200):
    S._lsq_normal(A, b)
print(f"_lsq_normal      x200 : {time.perf_counter() - t0:.3f}s")

t0 = time.perf_counter()
for _ in range(200):
    np.linalg.lstsq(A, b, rcond=None)
print(f"np.linalg.lstsq  x200 : {time.perf_counter() - t0:.3f}s")

t0 = time.perf_counter()
S.fit_water(I, J, h, w, iters=10, noise_mode="declared")
print(f"fit_water iters=10    : {time.perf_counter() - t0:.3f}s")

t0 = time.perf_counter()
S.fit_water(I, J, h, w, iters=60, noise_mode="declared")
print(f"fit_water iters=60    : {time.perf_counter() - t0:.3f}s")

t0 = time.perf_counter()
fit = S.fit_water(I, J, h, w, iters=16, noise_mode="declared")
print(f"fit_water iters=16    : {time.perf_counter() - t0:.3f}s "
      f"rmse={fit['rmse']:.5f} chi2={fit['chi2']:.1f}")

t0 = time.perf_counter()
an = S.analyse(I, J, fit)
print(f"analyse               : {time.perf_counter() - t0:.3f}s  A={an['A'].shape}")

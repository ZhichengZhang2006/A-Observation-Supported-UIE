"""Pin BLAS/OMP threads before numpy is imported anywhere in this package.

Import this module *first*.  ``obsui.support`` already does it at its own top,
but a script that imports numpy before ``obsui`` has OpenBLAS initialised with
96 threads by then and the setting is ignored.  Measured on school-gpu2:

    normal equations, 200x (4096x9):   5.386 s (96 threads) -> 0.289 s (1)
    np.linalg.lstsq,  200x:            0.644 s (96 threads) -> 0.413 s (1)

The machine is a shared 96-core host with a 20 GB GPU slice; single-threaded
BLAS is both faster here and fairer to the other tenants.
"""
from __future__ import annotations

import os

N_THREADS = int(os.environ.get("OBSUI_THREADS", "1"))

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, str(N_THREADS))

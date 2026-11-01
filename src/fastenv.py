"""Single-thread pinning. MUST be imported before numpy / cv2 / torch.

The assignment requires the timed pipeline to run on one CPU thread with no
extra processes, so every BLAS / OpenMP backend is pinned to a single thread
*before* the libraries that read these variables are loaded.
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import cv2  # noqa: E402
import numpy as np  # noqa: F401,E402

cv2.setNumThreads(1)
try:  # torch is not required by Track A, but pin it if present
    import torch
    torch.set_num_threads(1)
except Exception:
    pass

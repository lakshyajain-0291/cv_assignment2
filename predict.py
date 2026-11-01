#!/usr/bin/env python3
"""Segment one photo and report the timing and model-size figures.

    python predict.py --image photo.jpg --out mask.png

Loads the trained parameters, resizes the photo with the Section 1b rule,
segments it, writes the mask as a 0/255 PNG, then times 20 further runs on the
same photo and prints the average CPU time per run and the parameter file size.
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
import fastenv                      # noqa: F401  pins threads before numpy/cv2
import time
import numpy as np
import cv2

from pipeline import Segmenter, load_params

DEFAULT_PARAMS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "params", "model.npz")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--params", default=DEFAULT_PARAMS)
    ap.add_argument("--runs", type=int, default=20)
    a = ap.parse_args()

    seg = Segmenter(load_params(a.params))

    # Reading and resizing the file are not part of the timed pipeline.
    bgr = cv2.imread(a.image, cv2.IMREAD_COLOR)
    if bgr is None:
        raise SystemExit(f"could not read {a.image}")
    img256 = np.ascontiguousarray(
        cv2.resize(bgr, (256, 256), interpolation=cv2.INTER_AREA))

    # First run is slow while libraries warm up, so it is excluded.
    mask = seg.segment(img256)
    cv2.imwrite(a.out, (mask * 255).astype(np.uint8))

    total = 0.0
    for _ in range(a.runs):
        t0 = time.process_time()
        seg.segment(img256)
        total += time.process_time() - t0

    print(f"T_CPU_ms: {total / a.runs * 1000.0:.2f}")
    print(f"S_disk_bytes: {os.path.getsize(a.params)}")


if __name__ == "__main__":
    main()

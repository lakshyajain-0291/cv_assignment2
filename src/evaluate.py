"""Evaluate the trained model on validation, DUTS-TE and the captured photos,
and measure the single-thread CPU time exactly as Section 2b specifies.
"""
import os, sys, json, glob, argparse, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fastenv  # noqa: F401
import numpy as np
import cv2

import data
from pipeline import Segmenter, load_params
from metrics import confusion, miou_from_confusion, bf1, quality, ces

_E = {}


def _init(params):
    _E["seg"] = Segmenter(params)


def _one(args):
    img, msk = args
    pred = _E["seg"].segment(img)
    return confusion(pred, msk), bf1(pred, msk)


def eval_arrays(params, imgs, msks, idx, workers=8):
    """Quality metrics only. Parallel across images purely to save wall-clock;
    timing is measured separately and strictly single-threaded."""
    from multiprocessing import Pool
    payload = [(np.ascontiguousarray(imgs[i]), np.asarray(msks[i])) for i in idx]
    cm = np.zeros((2, 2), np.int64); bs = []
    with Pool(workers, initializer=_init, initargs=(params,)) as pool:
        for c, b in pool.imap(_one, payload, chunksize=16):
            cm += c; bs.append(b)
    miou, iou_bg, iou_ob = miou_from_confusion(cm)
    mb = float(np.mean(bs))
    return dict(n=len(idx), miou=miou, iou_bg=iou_bg, iou_obj=iou_ob,
                bf1=mb, Q=quality(miou, mb))


def time_pipeline(params, imgs, idx):
    """Section 2b: one image at a time, single thread, time.process_time()."""
    seg = Segmenter(params)
    seg.segment(np.ascontiguousarray(imgs[idx[0]]))      # warm-up, not counted
    total = 0.0
    per = []
    for i in idx:
        x = np.ascontiguousarray(imgs[i])
        t0 = time.process_time()
        seg.segment(x)
        dt = time.process_time() - t0
        total += dt; per.append(dt * 1000.0)
    return total / len(idx) * 1000.0, np.array(per)


def load_captured():
    ims, mks, names = [], [], []
    for p in sorted(glob.glob("captured/images_256/*.png")):
        b = os.path.basename(p)
        mp = os.path.join("captured/masks_256", b)
        if not os.path.exists(mp):
            continue
        ims.append(cv2.imread(p, cv2.IMREAD_COLOR))
        mks.append((cv2.imread(mp, cv2.IMREAD_GRAYSCALE) >= 128).astype(np.uint8))
        names.append(os.path.splitext(b)[0])
    return ims, mks, names


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default="params/model.npz")
    ap.add_argument("--n_time", type=int, default=0,
                    help="images to time; 0 = all of DUTS-TE")
    ap.add_argument("--out", default="out/results.json")
    a = ap.parse_args()

    params = load_params(a.params)
    sdisk = os.path.getsize(a.params)
    res = {"S_disk_bytes": sdisk, "params": a.params}

    vi, vm = data.load_cache("val")
    ti, tm = data.load_cache("test")

    print("validation split ...", flush=True)
    res["val"] = eval_arrays(params, vi, vm, range(len(vi)))
    print("   ", res["val"], flush=True)

    print("DUTS-TE ...", flush=True)
    res["test"] = eval_arrays(params, ti, tm, range(len(ti)))
    print("   ", res["test"], flush=True)

    ims, mks, names = load_captured()
    if ims:
        print(f"captured photos ({len(ims)}) ...", flush=True)
        arr_i = np.stack(ims); arr_m = np.stack(mks)
        res["captured"] = eval_arrays(params, arr_i, arr_m, range(len(ims)), workers=4)
        per = []
        seg = Segmenter(params)
        for im, mk, nm in zip(ims, mks, names):
            p = seg.segment(np.ascontiguousarray(im))
            cm = confusion(p, mk)
            mi, _, _ = miou_from_confusion(cm)
            per.append(dict(name=nm, miou=mi, bf1=bf1(p, mk),
                            obj_frac=float(mk.mean())))
        res["captured_per_image"] = per
        print("   ", res["captured"], flush=True)

    n_time = a.n_time if a.n_time > 0 else len(ti)
    print(f"timing on {n_time} DUTS-TE images (single thread) ...", flush=True)
    tcpu, per_img = time_pipeline(params, ti, range(n_time))
    res["T_CPU_ms"] = tcpu
    res["T_CPU_p50"] = float(np.percentile(per_img, 50))
    res["T_CPU_p99"] = float(np.percentile(per_img, 99))
    res["T_CPU_max"] = float(per_img.max())
    res["n_timed"] = n_time

    for split in ("val", "test", "captured"):
        if split in res:
            res[split]["CES"] = ces(res[split]["Q"], tcpu, sdisk)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=2)
    print(json.dumps({k: v for k, v in res.items()
                      if k != "captured_per_image"}, indent=2))

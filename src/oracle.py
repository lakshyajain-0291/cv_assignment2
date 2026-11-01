"""Diagnostic: how good could the graph cut possibly be at a given resolution?

Labels each region with its true majority label and measures the result. This
upper-bounds any region-level method at that clustering resolution, separating
error caused by the clustering granularity from error caused by the region
model and the cut.
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fastenv  # noqa: F401
import numpy as np, cv2
import data, regions as R
from pipeline import Segmenter
from metrics import confusion, miou_from_confusion, bf1, quality

_W = {}


def _init(cfg):
    _W["seg"] = Segmenter(dict(cfg, w=np.zeros(R.N_FEAT, np.float32), b=0.0,
                               prior=np.zeros((R.PRIOR_G, R.PRIOR_G), np.float32),
                               llr=np.zeros(R.LLR_BINS, np.float32),
                               lam=1.0, sigma_mult=1.0))
    _W["res"] = cfg["work_res"]


def _one(args):
    img, msk = args
    seg, res = _W["seg"], _W["res"]
    d = seg.regions(img)
    ms = cv2.resize(msk.astype(np.float32), (res, res),
                    interpolation=cv2.INTER_AREA).ravel()
    frac = np.bincount(d["labels"], weights=ms, minlength=d["nreg"]) / d["sizes"]
    lab = (frac >= 0.5).astype(np.uint8)
    pred = cv2.resize(lab[d["labels"]].reshape(res, res), (256, 256),
                      interpolation=cv2.INTER_NEAREST)
    return confusion(pred, msk), bf1(pred, msk)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    a = ap.parse_args()
    vi, vm = data.load_cache("val")
    payload = [(np.ascontiguousarray(vi[i]), np.asarray(vm[i])) for i in range(a.n)]
    from multiprocessing import Pool
    print(f"{'config':<28} {'mIoU':>7} {'BF1':>7} {'Q':>7}   <- region-oracle ceiling")
    for res, grid, it, comp in [(96,12,5,10.0),(128,16,4,10.0),(160,20,4,10.0),
                                (128,16,4,20.0),(128,16,4,5.0),(192,24,4,10.0)]:
        cfg = dict(work_res=res, grid=grid, kmeans_iters=it, compactness=comp)
        cm = np.zeros((2,2), np.int64); bs=[]
        with Pool(8, initializer=_init, initargs=(cfg,)) as p:
            for c,b in p.imap(_one, payload, chunksize=16):
                cm+=c; bs.append(b)
        miou,_,_ = miou_from_confusion(cm); mb=float(np.mean(bs))
        print(f"res={res} grid={grid}({grid*grid:3d}) m={comp:<4} "
              f"{miou:7.4f} {mb:7.4f} {quality(miou,mb):7.4f}")

"""Train the region model: colour likelihoods, position prior, logistic
regression weights, and the graph-cut hyper-parameters.

All three learned objects are estimated on the training split only; lambda and
sigma_mult are selected on the validation split. DUTS-TE is never touched here.
"""
import os, sys, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fastenv                      # noqa: F401  (pins threads first)
import numpy as np
import cv2

import data
import regions as R
from pipeline import Segmenter, save_params
from metrics import confusion, miou_from_confusion, bf1, quality

PRIOR_G = R.PRIOR_G
LLR_BINS = R.LLR_BINS


# ---------------------------------------------------------------- stage 1 --
def colour_and_position_stats(imgs, msks, res, idx):
    """P(object | image position) and the colour log-likelihood ratio table.

    Both are pixel-level statistics, so no clustering is needed here.
    """
    prior = np.zeros((PRIOR_G, PRIOR_G), dtype=np.float64)
    obj_h = np.zeros(LLR_BINS, dtype=np.float64)
    bg_h = np.zeros(LLR_BINS, dtype=np.float64)
    for n, i in enumerate(idx):
        img = np.ascontiguousarray(imgs[i])
        msk = msks[i].astype(np.float32)
        prior += cv2.resize(msk, (PRIOR_G, PRIOR_G), interpolation=cv2.INTER_AREA)
        small = cv2.resize(img, (res, res), interpolation=cv2.INTER_AREA)
        lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
        ms = cv2.resize(msk, (res, res), interpolation=cv2.INTER_AREA) >= 0.5
        b = R.colour_bin(lab.reshape(-1, 3))
        obj_h += np.bincount(b[ms.ravel()], minlength=LLR_BINS)
        bg_h += np.bincount(b[~ms.ravel()], minlength=LLR_BINS)
        if (n + 1) % 500 == 0:
            print(f"  stats {n+1}/{len(idx)}", flush=True)
    prior /= len(idx)
    a = 1.0
    llr = (np.log((obj_h + a) / (obj_h.sum() + a * LLR_BINS))
           - np.log((bg_h + a) / (bg_h.sum() + a * LLR_BINS)))
    return prior.astype(np.float32), llr.astype(np.float32)


# ---------------------------------------------------------------- stage 2 --
_W = {}


def _init_worker(cfg, prior, llr):
    _W["seg"] = Segmenter(dict(cfg, w=np.zeros(R.N_EXP, np.float32), b=0.0,
                               prior=prior, llr=llr))
    _W["res"] = cfg["work_res"]


def _extract_one(args):
    """Region features plus the soft object fraction of each region."""
    img, msk = args
    seg = _W["seg"]; res = _W["res"]
    d = seg.regions(img)
    ms = cv2.resize(msk.astype(np.float32), (res, res),
                    interpolation=cv2.INTER_AREA).ravel()
    y = (np.bincount(d["labels"], weights=ms, minlength=d["nreg"])
         / d["sizes"]).astype(np.float32)
    return R.expand(d["feats"]), y


def extract_dataset(imgs, msks, idx, cfg, prior, llr, workers=8):
    from multiprocessing import Pool
    payload = [(np.ascontiguousarray(imgs[i]), np.asarray(msks[i])) for i in idx]
    Xs, Ys = [], []
    with Pool(workers, initializer=_init_worker,
              initargs=(cfg, prior, llr)) as pool:
        for n, (f, y) in enumerate(pool.imap(_extract_one, payload, chunksize=16)):
            Xs.append(f); Ys.append(y)
            if (n + 1) % 500 == 0:
                print(f"  feats {n+1}/{len(idx)}", flush=True)
    return np.concatenate(Xs), np.concatenate(Ys)


# ---------------------------------------------------------------- stage 3 --
def fit_logreg(X, y, iters=600, lr=0.5, l2=1e-4):
    """Logistic regression on soft targets, trained from scratch with Adam.

    Targets are the fraction of each region that is object, so this minimises
    cross-entropy against a soft label rather than a hard 0/1 one -- regions
    straddling the object boundary then pull the weights only partially.
    Features are standardised for conditioning and the standardisation is
    folded back into (w, b) afterwards, so inference stays a single dot
    product with no extra cost inside the timed path.
    """
    mu = X.mean(0)
    sd = X.std(0) + 1e-8
    Z = (X - mu) / sd
    n, f = Z.shape
    w = np.zeros(f, dtype=np.float64)
    b = 0.0
    mw = np.zeros(f); vw = np.zeros(f); mb = vb = 0.0
    b1, b2, eps = 0.9, 0.999, 1e-8
    Z = Z.astype(np.float32); y = y.astype(np.float32)
    for t in range(1, iters + 1):
        p = 1.0 / (1.0 + np.exp(-(Z @ w + b)))
        diff = p - y
        gw = Z.T @ diff / n + l2 * w
        gb = diff.mean()
        mw = b1 * mw + (1 - b1) * gw; vw = b2 * vw + (1 - b2) * gw * gw
        mb = b1 * mb + (1 - b1) * gb; vb = b2 * vb + (1 - b2) * gb * gb
        w -= lr * (mw / (1 - b1 ** t)) / (np.sqrt(vw / (1 - b2 ** t)) + eps)
        b -= lr * (mb / (1 - b1 ** t)) / (np.sqrt(vb / (1 - b2 ** t)) + eps)
        if t % 200 == 0:
            ll = -(y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9)).mean()
            print(f"  logreg iter {t}: loss {ll:.4f}", flush=True)
    # Fold standardisation into the weights.
    w_f = (w / sd).astype(np.float32)
    b_f = float(b - np.sum(mu * w / sd))
    return w_f, b_f


# ---------------------------------------------------------------- stage 4 --
_E = {}


def _init_eval(params):
    _E["seg"] = Segmenter(params)


def _eval_one(args):
    img, msk = args
    pred = _E["seg"].segment(img)
    return confusion(pred, msk), bf1(pred, msk)


def evaluate(params, imgs, msks, idx, workers=8):
    from multiprocessing import Pool
    payload = [(np.ascontiguousarray(imgs[i]), np.asarray(msks[i])) for i in idx]
    cm = np.zeros((2, 2), dtype=np.int64)
    bs = []
    with Pool(workers, initializer=_init_eval, initargs=(params,)) as pool:
        for c, b in pool.imap(_eval_one, payload, chunksize=16):
            cm += c; bs.append(b)
    miou, iou_bg, iou_ob = miou_from_confusion(cm)
    mb = float(np.mean(bs))
    return dict(miou=miou, iou_bg=iou_bg, iou_obj=iou_ob, bf1=mb,
                Q=quality(miou, mb), n=len(idx))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, default=128)
    ap.add_argument("--grid", type=int, default=16)
    ap.add_argument("--iters", type=int, default=4)
    ap.add_argument("--compactness", type=float, default=10.0)
    ap.add_argument("--n_stats", type=int, default=3000)
    ap.add_argument("--n_feat", type=int, default=3000)
    ap.add_argument("--n_tune", type=int, default=400)
    ap.add_argument("--max_regions", type=int, default=300000)
    ap.add_argument("--out", default="params/model.npz")
    a = ap.parse_args()

    cfg = dict(work_res=a.res, grid=a.grid, kmeans_iters=a.iters,
               compactness=a.compactness, lam=1.0, sigma_mult=1.0)
    fi, fm = data.load_cache("fit")
    vi, vm = data.load_cache("val")
    t0 = time.time()

    print(f"[1/4] colour + position statistics on {a.n_stats} training images")
    prior, llr = colour_and_position_stats(fi, fm, a.res,
                                           range(min(a.n_stats, len(fi))))
    print(f"      prior mean {prior.mean():.3f} | llr range "
          f"[{llr.min():.2f}, {llr.max():.2f}]  ({time.time()-t0:.0f}s)")

    print(f"[2/4] region features on {a.n_feat} training images")
    X, y = extract_dataset(fi, fm, range(min(a.n_feat, len(fi))), cfg, prior, llr)
    print(f"      {X.shape[0]} regions, mean object fraction {y.mean():.3f}"
          f"  ({time.time()-t0:.0f}s)")

    print("[3/4] logistic regression on a quadratic basis")
    if X.shape[0] > a.max_regions:
        sel = np.random.RandomState(0).choice(X.shape[0], a.max_regions, False)
        X, y = X[sel], y[sel]
        print(f"      subsampled to {X.shape[0]} regions")
    w, b = fit_logreg(X, y)
    print(f"      {w.size} weights, |w| in [{np.abs(w).min():.3g}, "
          f"{np.abs(w).max():.3g}], bias {b:+.3f}")

    print(f"[4/4] tuning lambda, sigma_mult on {a.n_tune} validation images")
    tune_idx = list(range(min(a.n_tune, len(vi))))
    best = None
    for lam in [1.0, 2.0, 4.0, 8.0, 16.0]:
        for sm in [0.05, 0.1, 0.25, 0.5]:
            p = dict(cfg, lam=lam, sigma_mult=sm, w=w, b=b, prior=prior, llr=llr)
            m = evaluate(p, vi, vm, tune_idx)
            tag = "  "
            if best is None or m["Q"] > best[0]["Q"]:
                best = (m, lam, sm); tag = " *"
            print(f"   {tag} lam={lam:<5} sigma_mult={sm:<4} "
                  f"mIoU {m['miou']:.4f} BF1 {m['bf1']:.4f} Q {m['Q']:.4f}",
                  flush=True)
    m, lam, sm = best
    print(f"      chosen lambda={lam} sigma_mult={sm} (Q={m['Q']:.4f})")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    save_params(a.out, w=w.astype(np.float32), b=np.float32(b),
                llr=llr.astype(np.float32), prior=prior.astype(np.float32),
                lam=np.float32(lam), sigma_mult=np.float32(sm),
                work_res=np.int32(a.res), grid=np.int32(a.grid),
                kmeans_iters=np.int32(a.iters),
                compactness=np.float32(a.compactness))
    print(f"saved {a.out}  ({os.path.getsize(a.out)} bytes)  "
          f"total {time.time()-t0:.0f}s")

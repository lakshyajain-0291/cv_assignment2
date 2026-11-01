"""Re-tune only lambda and sigma_mult for an already-trained region model.

Kept separate from train.py so the hyper-parameter search can be extended
without repeating feature extraction or refitting the logistic regression.
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fastenv  # noqa: F401
import numpy as np
import data
from pipeline import load_params, save_params
from train import evaluate

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default="params/model.npz")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--lams", default="8,16,24,32,48")
    ap.add_argument("--sms", default="0.15,0.25,0.4,0.6")
    a = ap.parse_args()

    p = load_params(a.params)
    vi, vm = data.load_cache("val")
    idx = list(range(min(a.n, len(vi))))
    best = None
    for lam in [float(x) for x in a.lams.split(",")]:
        for sm in [float(x) for x in a.sms.split(",")]:
            q = dict(p); q["lam"] = lam; q["sigma_mult"] = sm
            m = evaluate(q, vi, vm, idx)
            star = "  "
            if best is None or m["Q"] > best[0]["Q"]:
                best = (m, lam, sm); star = " *"
            print(f" {star} lam={lam:<5} sm={sm:<5} mIoU {m['miou']:.4f} "
                  f"BF1 {m['bf1']:.4f} Q {m['Q']:.4f}", flush=True)
    m, lam, sm = best
    print(f"chosen lambda={lam} sigma_mult={sm}  Q={m['Q']:.4f}")
    p["lam"] = np.float32(lam); p["sigma_mult"] = np.float32(sm)
    save_params(a.params, **p)
    print(f"updated {a.params} ({os.path.getsize(a.params)} bytes)")

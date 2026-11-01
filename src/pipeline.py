"""The Track-A segmentation pipeline: cluster -> region model -> graph cut.

`Segmenter.segment` is the part that is timed. It takes a 256x256 BGR image
already in memory and returns a 256x256 array of 0s and 1s. Everything inside
it counts towards the budget, including the downscale and the colour
conversion, so the only work done per image is work that earns its keep.

Speed strategy
--------------
Clustering and max-flow over all 65,536 pixels cannot fit in 50 ms in Python.
Instead the image is downscaled to WORK_RES (default 112), clustered into a few
hundred compact regions, and the graph cut runs on the *region* graph -- a few
hundred nodes rather than 65k. The resulting region labelling is then expanded
back to 256x256. WORK_RES controls the quality/time trade-off directly: it sets
both how much pixel work the clustering does and how finely region boundaries
can follow real object edges once upsampled.
"""
import numpy as np
import cv2

from kmeans import KMeansGrid
from maxflow import MaxFlow
import regions as R

CAP_SCALE = 512.0      # capacities are integers; this is the fixed-point scale


class Segmenter:
    def __init__(self, params):
        p = params
        self.work_res = int(p["work_res"])
        self.grid = int(p["grid"])
        self.lam = float(p["lam"])
        self.sigma_mult = float(p["sigma_mult"])
        self.w = np.asarray(p["w"], dtype=np.float32)
        self.b = float(p["b"])
        self.llr = np.asarray(p["llr"], dtype=np.float32)
        self.prior = np.asarray(p["prior"], dtype=np.float32)
        self.km = KMeansGrid(self.work_res, self.grid,
                             compactness=float(p["compactness"]),
                             iters=int(p["kmeans_iters"]))

    # -- stage 1+2: downscale, colour convert, cluster, describe regions ------
    def regions(self, bgr256):
        small = cv2.resize(bgr256, (self.work_res, self.work_res),
                           interpolation=cv2.INTER_AREA)
        lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
        labels = self.km.fit(lab)
        return R.describe(lab, labels, self.work_res, self.prior, self.llr)

    # -- stage 3: learned probability that each region is object -------------
    def region_prob(self, feats):
        z = R.expand(feats) @ self.w + self.b
        p = 1.0 / (1.0 + np.exp(-z))
        return np.clip(p, 0.001, 0.999)      # clipping required by the brief

    # -- stage 4: build the s-t graph and cut it -----------------------------
    def cut(self, d, prob):
        M = d["nreg"]
        sizes = d["sizes"]
        src, snk = M, M + 1
        g = MaxFlow(M + 2)

        # Terminal capacities. Cutting p->sink leaves p attached to the source,
        # i.e. labels it object, and pays -log P_p; cutting source->p pays
        # -log(1 - P_p). Both are weighted by the region's pixel count so that
        # the unary term is a sum over pixels, commensurate with the pairwise
        # term which is a sum over boundary pixel pairs.
        cap_src = (-np.log(1.0 - prob) * sizes * CAP_SCALE).astype(np.int64)
        cap_snk = (-np.log(prob) * sizes * CAP_SCALE).astype(np.int64)
        add = g.add_edge
        for i in range(M):
            add(src, i, int(cap_src[i]))
            add(i, snk, int(cap_snk[i]))

        # Pairwise term: similar neighbours are expensive to separate.
        pi, pj, blen, d_lab = d["pi"], d["pj"], d["blen"], d["d_lab"]
        if pi.size:
            d2 = d_lab.astype(np.float64) ** 2
            sigma2 = d2.mean() * self.sigma_mult + 1e-6
            w = np.exp(-d2 / (2.0 * sigma2)) * (self.lam * blen * CAP_SCALE)
            w = w.astype(np.int64)
            for k in range(pi.size):
                c = int(w[k])
                if c > 0:
                    add(int(pi[k]), int(pj[k]), c, c)   # symmetric, both ways

        g.max_flow(src, snk)
        side = g.min_cut_source_side(src)
        return np.frombuffer(bytes(side[:M]), dtype=np.uint8)

    # -- the timed entry point ----------------------------------------------
    def segment(self, bgr256):
        d = self.regions(bgr256)
        prob = self.region_prob(d["feats"])
        is_obj = self.cut(d, prob)
        small_mask = is_obj[d["labels"]].reshape(self.work_res, self.work_res)
        return cv2.resize(small_mask, (256, 256),
                          interpolation=cv2.INTER_NEAREST)


def load_params(path):
    z = np.load(path)
    return {k: z[k] for k in z.files}


def save_params(path, **kw):
    np.savez(path, **kw)      # plain .npz, not compressed (per the rules)

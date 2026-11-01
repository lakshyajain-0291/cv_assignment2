"""Region adjacency and region descriptors, built with vectorised NumPy only.

Everything here runs inside the timed budget, so there are no Python loops over
pixels: region statistics come from `np.bincount` and adjacency comes from
comparing the label image against its own one-pixel shifts.
"""
import numpy as np

N_FEAT = 17
PRIOR_G = 16          # resolution of the learned position prior
LLR_BITS = 3          # 2^3 bins per Lab channel -> 512-entry colour table
LLR_BINS = (1 << LLR_BITS) ** 3


def colour_bin(lab_flat):
    """Quantise Lab (uint8 range, float array (N,3)) to a 512-bin index."""
    sh = 8 - LLR_BITS
    q = (lab_flat.astype(np.int32) >> sh)
    np.clip(q, 0, (1 << LLR_BITS) - 1, out=q)
    return (q[:, 0] << (2 * LLR_BITS)) | (q[:, 1] << LLR_BITS) | q[:, 2]


def adjacency(labels2d, nreg):
    """4-connected region adjacency and shared border lengths.

    Two regions touch if any of their pixels are directly above, below, left or
    right of each other; the shared border length is the number of such
    touching pixel pairs. We obtain this by comparing the label image with its
    own one-pixel shifts and histogramming the unordered (p, q) pairs.
    """
    a = np.concatenate([labels2d[:, :-1].ravel(), labels2d[:-1, :].ravel()])
    b = np.concatenate([labels2d[:, 1:].ravel(), labels2d[1:, :].ravel()])
    m = a != b
    a, b = a[m], b[m]
    lo = np.minimum(a, b).astype(np.int64)
    hi = np.maximum(a, b).astype(np.int64)
    uniq, cnt = np.unique(lo * nreg + hi, return_counts=True)
    return ((uniq // nreg).astype(np.int32),
            (uniq % nreg).astype(np.int32),
            cnt.astype(np.float64))


def describe(lab, labels, res, prior_map, llr_table):
    """Per-region features plus the adjacency structure of the region graph.

    lab        : (res, res, 3) float32 Lab image
    labels     : (res*res,) int32 raw cluster ids (may contain empty clusters)
    prior_map  : (PRIOR_G, PRIOR_G) learned P(object | position)
    llr_table  : (LLR_BINS,) log P(colour|object) - log P(colour|background)
    """
    N = res * res
    flat = lab.reshape(N, 3)

    # --- compact away empty clusters so region ids are contiguous ---
    uniq, labels_c = np.unique(labels, return_inverse=True)
    M = uniq.size
    labels_c = labels_c.astype(np.int32)
    sizes = np.bincount(labels_c, minlength=M).astype(np.float64)

    mean_lab = np.empty((M, 3), dtype=np.float64)
    for c in range(3):
        mean_lab[:, c] = np.bincount(labels_c, weights=flat[:, c], minlength=M)
    mean_lab /= sizes[:, None]

    yy, xx = np.divmod(np.arange(N, dtype=np.int32), res)
    cy = (np.bincount(labels_c, weights=yy.astype(np.float64), minlength=M)
          / sizes) / (res - 1)
    cx = (np.bincount(labels_c, weights=xx.astype(np.float64), minlength=M)
          / sizes) / (res - 1)

    # --- how much of each region sits on the image border ---
    on_border = np.zeros(N, dtype=np.float64)
    ob2 = on_border.reshape(res, res)
    ob2[0, :] = ob2[-1, :] = ob2[:, 0] = ob2[:, -1] = 1.0
    border_ratio = np.bincount(labels_c, weights=on_border, minlength=M) / sizes

    # --- learned cues looked up per pixel, then averaged per region ---
    prior_pix = prior_map[(yy * PRIOR_G) // res, (xx * PRIOR_G) // res]
    mean_prior = np.bincount(labels_c, weights=prior_pix.astype(np.float64),
                             minlength=M) / sizes
    llr_pix = llr_table[colour_bin(flat)]
    mean_llr = np.bincount(labels_c, weights=llr_pix.astype(np.float64),
                           minlength=M) / sizes

    # --- simple contrast cues ---
    global_mean = flat.mean(0)
    border_mean = flat[ob2.ravel() > 0].mean(0)
    cc_global = np.linalg.norm(mean_lab - global_mean, axis=1) / 255.0
    cc_border = np.linalg.norm(mean_lab - border_mean, axis=1) / 255.0

    # --- adjacency and local contrast ---
    pi, pj, blen = adjacency(labels_c.reshape(res, res), M)
    d_lab = np.linalg.norm(mean_lab[pi] - mean_lab[pj], axis=1)
    wsum = (np.bincount(pi, weights=blen, minlength=M)
            + np.bincount(pj, weights=blen, minlength=M))
    dsum = (np.bincount(pi, weights=blen * d_lab, minlength=M)
            + np.bincount(pj, weights=blen * d_lab, minlength=M))
    local_contrast = dsum / np.maximum(wsum, 1e-6) / 255.0

    # --- spatially weighted colour contrast (region-contrast cue) ---
    # A region whose colour differs from the rest of the image -- especially
    # from nearby regions -- is likely to be the object. M is a few hundred, so
    # the full MxM pairwise matrix is cheap to form.
    # Pairwise distances via the ||a||^2 - 2a.b + ||b||^2 expansion, so the
    # heavy part is a small GEMM rather than an (M, M, 3) broadcast.
    cxy = np.stack([cy, cx], 1)
    nl = (mean_lab ** 2).sum(1)
    d2col = nl[:, None] + nl[None, :] - 2.0 * (mean_lab @ mean_lab.T)
    np.maximum(d2col, 0.0, out=d2col)
    dcol = np.sqrt(d2col)
    ns = (cxy ** 2).sum(1)
    d2spa = ns[:, None] + ns[None, :] - 2.0 * (cxy @ cxy.T)
    np.maximum(d2spa, 0.0, out=d2spa)
    area_w = (sizes / N)[None, :]
    rc_near = (np.exp(-d2spa / (2 * 0.25 ** 2)) * area_w * dcol).sum(1) / 255.0
    rc_far = (area_w * dcol).sum(1) / 255.0

    # --- graph diffusion of backgroundness ---
    # Border regions are usually background, and that evidence should spread to
    # regions joined to them through colour-similar neighbours. A few rounds of
    # similarity-weighted averaging against the border seed approximate a
    # random walk on the region graph, a far stronger background cue than
    # border contact on its own.
    if pi.size:
        s2 = d_lab ** 2
        wsim = np.exp(-s2 / (2.0 * (s2.mean() + 1e-6))) * blen
        deg = (np.bincount(pi, weights=wsim, minlength=M)
               + np.bincount(pj, weights=wsim, minlength=M) + 1e-6)

        def diffuse(seed, alpha=0.7, iters=5):
            x = seed.copy()
            for _ in range(iters):
                acc = (np.bincount(pi, weights=wsim * x[pj], minlength=M)
                       + np.bincount(pj, weights=wsim * x[pi], minlength=M))
                x = (1 - alpha) * seed + alpha * (acc / deg)
            return x
        bg_diff = diffuse(border_ratio)
        llr_diff = diffuse(mean_llr)
    else:
        bg_diff, llr_diff = border_ratio.copy(), mean_llr.copy()

    # --- within-region colour spread (texture / flatness proxy) ---
    sq = np.empty((M, 3), dtype=np.float64)
    for c in range(3):
        sq[:, c] = np.bincount(labels_c, weights=flat[:, c].astype(np.float64) ** 2,
                               minlength=M)
    var = np.maximum(sq / sizes[:, None] - mean_lab ** 2, 0.0)
    col_std = np.sqrt(var.sum(1)) / 255.0

    feats = np.empty((M, N_FEAT), dtype=np.float32)
    feats[:, 0] = mean_llr
    feats[:, 1] = mean_prior
    feats[:, 2] = border_ratio
    feats[:, 3] = sizes / N
    feats[:, 4] = np.abs(cx - 0.5)
    feats[:, 5] = np.abs(cy - 0.5)
    feats[:, 6] = cc_global
    feats[:, 7] = cc_border
    feats[:, 8] = local_contrast
    feats[:, 9] = mean_lab[:, 0] / 255.0
    feats[:, 10] = (mean_lab[:, 1] - 128.0) / 128.0
    feats[:, 11] = (mean_lab[:, 2] - 128.0) / 128.0
    feats[:, 12] = rc_near
    feats[:, 13] = rc_far
    feats[:, 14] = bg_diff
    feats[:, 15] = llr_diff
    feats[:, 16] = col_std

    return dict(feats=feats, sizes=sizes.astype(np.float32),
                mean_lab=mean_lab.astype(np.float32), nreg=M,
                labels=labels_c, pi=pi, pj=pj,
                blen=blen.astype(np.float32), d_lab=d_lab.astype(np.float32))


# --- quadratic basis expansion ------------------------------------------
# The region model is still a logistic regression, but fitted on a quadratic
# basis: the raw cues plus every pairwise product (and square). This lets the
# model express interactions -- "high colour contrast *and* low background
# diffusion" -- that a purely linear decision surface over these 17 cues
# cannot, while staying a convex problem we train ourselves. The expansion
# costs one multiply per term over a few hundred regions, so it is negligible
# inside the time budget, and the extra weights add ~600 bytes to the file.
_IU, _JU = np.triu_indices(N_FEAT)
N_EXP = N_FEAT + _IU.size


def expand(feats):
    """(M, N_FEAT) -> (M, N_EXP) quadratic basis."""
    return np.concatenate([feats, feats[:, _IU] * feats[:, _JU]], axis=1)

"""Grid-constrained K-Means over (Lab, x, y) -- written from scratch.

Plain Lloyd's algorithm costs O(N*K) per assignment sweep, which for N~12.5k
pixels and K~200 clusters is ~2.5M distance evaluations -- far too slow inside
a 50 ms budget. We use the locality insight behind SLIC: centres start on a
regular grid, and a pixel is only compared against the grid centres nearest to
it. Assignment then costs O(C*N) with C a small constant, independent of K.

Two further choices make this cheap enough that clustering stops dominating the
runtime:

* **Four candidates, not nine.** The four nearest grid centres to a pixel are
  fully determined by which quadrant of its cell the pixel falls in, so we take
  the pixel's own cell plus the two edge-adjacent and one corner-adjacent cell
  on that side. This halves the assignment work versus scanning the full 3x3
  neighbourhood, and a centre two cells away essentially never wins once the
  spatial term is weighted by `compactness`.
* **Dot-product distance.** Since ||f - c||^2 = ||f||^2 - 2 f.c + ||c||^2 and
  ||f||^2 is the same for every candidate, the argmin only needs
  ||c||^2 - 2 f.c, which drops one array operation per candidate.

This is still Lloyd's algorithm -- assign, then recompute means. Only the
candidate set per pixel is restricted, which is also what keeps the clusters
spatially compact, as the graph cut downstream treats each cluster as a single
connected node.
"""
import numpy as np


def _quadrant_candidates(res, grid):
    """For each pixel, the 4 nearest grid-cell centres. Shape (4, res*res).

    Depends only on (res, grid) so it is built once and reused for every image;
    it is never recomputed inside the timed path.
    """
    yy, xx = np.meshgrid(np.arange(res), np.arange(res), indexing="ij")
    step = res / grid
    gy = np.minimum((yy / step).astype(np.int32), grid - 1)
    gx = np.minimum((xx / step).astype(np.int32), grid - 1)
    # Which side of its own cell centre the pixel lies on.
    dy = np.where(yy < (gy + 0.5) * step, -1, 1)
    dx = np.where(xx < (gx + 0.5) * step, -1, 1)
    ny = np.clip(gy + dy, 0, grid - 1)
    nx = np.clip(gx + dx, 0, grid - 1)
    cands = np.empty((4, res * res), dtype=np.int32)
    cands[0] = (gy * grid + gx).ravel()
    cands[1] = (ny * grid + gx).ravel()
    cands[2] = (gy * grid + nx).ravel()
    cands[3] = (ny * grid + nx).ravel()
    return cands


class KMeansGrid:
    """Re-usable grid-constrained K-Means for a fixed working resolution."""

    def __init__(self, res, grid, compactness=10.0, iters=5):
        self.res = res
        self.grid = grid
        self.K = grid * grid
        self.compactness = compactness
        self.iters = iters
        self.cands = _quadrant_candidates(res, grid)
        # Position weight: scales (y, x) so `compactness` trades colour
        # distance against spatial distance, as in SLIC's m/S factor.
        self.wpos = compactness / (res / grid)
        step = res / grid
        c = (np.arange(grid) + 0.5) * step
        cy, cx = np.meshgrid(c, c, indexing="ij")
        self._init_xy = np.stack([cy.ravel(), cx.ravel()], 1).astype(np.float32)
        yy, xx = np.meshgrid(np.arange(res, dtype=np.float32),
                             np.arange(res, dtype=np.float32), indexing="ij")
        self._pix_xy = np.stack([yy.ravel(), xx.ravel()], 1) * self.wpos
        self._init_idx = (self._init_xy[:, 0].astype(np.int32) * res
                          + self._init_xy[:, 1].astype(np.int32))

    def fit(self, lab):
        """Cluster one image.

        lab : (res, res, 3) float32 CIE-Lab image.
        returns: labels (res*res,) int32 in [0, K)
        """
        N = self.res * self.res
        feats = np.empty((N, 5), dtype=np.float32)
        feats[:, :3] = lab.reshape(N, 3)
        feats[:, 3:] = self._pix_xy

        centres = np.empty((self.K, 5), dtype=np.float32)
        centres[:, :3] = feats[self._init_idx, :3]
        centres[:, 3:] = self._init_xy * self.wpos

        cands = self.cands
        labels = cands[0].copy()
        best = np.empty(N, dtype=np.float32)
        score = np.empty(N, dtype=np.float32)
        for it in range(self.iters):
            # ||c||^2 term; the ||f||^2 term is constant across candidates.
            cnorm = np.einsum("ij,ij->i", centres, centres)
            for k in range(4):
                ck = centres[cands[k]]
                np.einsum("ij,ij->i", feats, ck, out=score)
                score *= -2.0
                score += cnorm[cands[k]]
                if k == 0:
                    best[:] = score
                    labels[:] = cands[0]
                else:
                    better = score < best
                    np.copyto(best, score, where=better)
                    np.copyto(labels, cands[k], where=better)
            if it == self.iters - 1:
                break
            cnt = np.bincount(labels, minlength=self.K).astype(np.float32)
            nz = cnt > 0
            sums = np.empty((self.K, 5), dtype=np.float32)
            for d in range(5):
                sums[:, d] = np.bincount(labels, weights=feats[:, d],
                                         minlength=self.K)
            centres[nz] = sums[nz] / cnt[nz, None]   # empty clusters unchanged
        return labels

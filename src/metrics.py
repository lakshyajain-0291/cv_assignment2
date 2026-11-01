"""mIoU and Boundary-F1, implemented from scratch and fully vectorised.

The 5x5 tolerance check in BF1 is a binary dilation, but cv2.dilate is not
permitted. Convolving the binary boundary map with a 5x5 box of ones using
cv2.filter2D (which *is* permitted) counts how many boundary pixels fall in
each window; testing that count > 0 is exactly dilation by a 5x5 structuring
element. This replaces the per-pixel Python loop the assignment warns about.
"""
import cv2
import numpy as np

_BOX5 = np.ones((5, 5), dtype=np.float32)


def confusion(pred, gt):
    """2x2 confusion counts [[bg_bg, bg_ob], [ob_bg, ob_ob]] for one image."""
    p = pred.astype(np.uint8).ravel()
    g = gt.astype(np.uint8).ravel()
    return np.bincount(g * 2 + p, minlength=4).reshape(2, 2).astype(np.int64)


def miou_from_confusion(cm):
    """cm[g][p]. IoU per class, averaged over background and object."""
    ious = []
    for c in (0, 1):
        inter = cm[c, c]
        union = cm[c, :].sum() + cm[:, c].sum() - inter
        ious.append(inter / union if union > 0 else 1.0)
    return float(np.mean(ious)), float(ious[0]), float(ious[1])


def boundary(mask):
    """Object pixels having a 4-neighbour that is background.

    Pixels outside the image count as background, so a zero-padded shift is the
    right comparison and object pixels on the image edge are boundary pixels.
    """
    m = mask.astype(np.uint8)
    p = np.zeros((m.shape[0] + 2, m.shape[1] + 2), dtype=np.uint8)
    p[1:-1, 1:-1] = m
    nb = (p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:])
    return (m & ~nb).astype(np.uint8)


def _dilate5(b):
    """5x5 binary dilation via a box filter (cv2.dilate is not allowed)."""
    s = cv2.filter2D(b.astype(np.float32), -1, _BOX5,
                     borderType=cv2.BORDER_CONSTANT)
    return s > 0.5


def bf1(pred, gt):
    """Boundary F1 for one image, with a 5x5 matching tolerance."""
    pb = boundary(pred)
    gb = boundary(gt)
    np_, ng = int(pb.sum()), int(gb.sum())
    if np_ == 0 and ng == 0:
        return 1.0          # both masks empty of boundary -> perfect agreement
    if np_ == 0 or ng == 0:
        return 0.0          # only one has a boundary -> no match possible
    prec = float((pb.astype(bool) & _dilate5(gb)).sum()) / np_
    rec = float((gb.astype(bool) & _dilate5(pb)).sum()) / ng
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def quality(miou, bf1_mean):
    return 0.70 * miou + 0.30 * bf1_mean


def ces(q, t_cpu_ms, s_disk_bytes):
    if t_cpu_ms > 50.0:
        return 0.0
    return q * (1.0 - 0.20 * (t_cpu_ms / 50.0)
                - 0.10 * min(1.0, s_disk_bytes / 4194304.0))

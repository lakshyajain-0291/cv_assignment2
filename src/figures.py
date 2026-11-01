"""Report figures: captured-photo overlays, qualitative DUTS examples, and the
pipeline's intermediate stages.
"""
import os, sys, glob, argparse, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fastenv  # noqa: F401
import numpy as np
import cv2

import data
from pipeline import Segmenter, load_params
from metrics import confusion, miou_from_confusion, bf1

GREEN = np.array([80, 220, 80], np.float32)    # ground truth
RED = np.array([70, 70, 240], np.float32)      # prediction


def overlay(img, mask, colour, alpha=0.45):
    out = img.astype(np.float32).copy()
    m = (mask > 0)[:, :, None]
    out = np.where(m, (1 - alpha) * out + alpha * colour, out)
    return np.clip(out, 0, 255).astype(np.uint8)


def label_strip(w, text, h=22):
    s = np.full((h, w, 3), 245, np.uint8)
    cv2.putText(s, text, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (20, 20, 20), 1)
    return s


def panel(img, gt, pred, title):
    """photo | ground truth | prediction, as semi-transparent overlays."""
    cols = [img, overlay(img, gt, GREEN), overlay(img, pred, RED)]
    heads = ["photo", "true mask", "predicted mask"]
    tiles = [np.vstack([label_strip(256, f"{title} - {h}"), c])
             for c, h in zip(cols, heads)]
    return np.hstack(tiles)



def pack(rows, per_row=2):
    """Pack single-photo panels into a landscape grid.

    One panel is 768x278 (photo | truth | prediction). Stacking eight of them
    gives a 768x2224 strip that is far taller than a LaTeX page can hold, so we
    place `per_row` panels side by side instead and the figure stays wide.
    """
    if not rows:
        return None
    w = rows[0].shape[1]
    out = []
    for i in range(0, len(rows), per_row):
        chunk = list(rows[i:i + per_row])
        while len(chunk) < per_row:
            chunk.append(np.full_like(rows[0], 255))
        out.append(np.hstack(chunk))
    return np.vstack(out)


def captured_figures(seg, outdir):
    os.makedirs(outdir, exist_ok=True)
    rows, meta = [], []
    for p in sorted(glob.glob("captured/images_256/*.png")):
        b = os.path.basename(p)
        mp = os.path.join("captured/masks_256", b)
        if not os.path.exists(mp):
            continue
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        gt = (cv2.imread(mp, cv2.IMREAD_GRAYSCALE) >= 128).astype(np.uint8)
        pred = seg.segment(np.ascontiguousarray(img))
        name = os.path.splitext(b)[0]
        row = panel(img, gt, pred, name)
        cv2.imwrite(os.path.join(outdir, f"cap_{name}.png"), row)
        rows.append(row)
        mi, _, _ = miou_from_confusion(confusion(pred, gt))
        meta.append(dict(name=name, miou=float(mi), bf1=float(bf1(pred, gt)),
                         obj_frac=float(gt.mean())))
    if rows:
        half = (len(rows) + 1) // 2
        for k, part in enumerate((rows[:half], rows[half:]), start=1):
            g = pack(part, per_row=2)
            if g is not None:
                cv2.imwrite(os.path.join(outdir, f"captured_grid{k}.png"), g)
    return meta


def duts_examples(seg, outdir, n_good=3, n_bad=3):
    """Best and worst validation cases by IoU, for the qualitative figure."""
    os.makedirs(outdir, exist_ok=True)
    vi, vm = data.load_cache("val")
    scored = []
    for i in range(300):
        img = np.ascontiguousarray(vi[i]); gt = np.asarray(vm[i])
        pred = seg.segment(img)
        mi, _, _ = miou_from_confusion(confusion(pred, gt))
        scored.append((mi, i))
    scored.sort()
    picks = [("hard", scored[k][1]) for k in range(n_bad)] + \
            [("easy", scored[-1 - k][1]) for k in range(n_good)]
    rows = []
    for tag, i in picks:
        img = np.ascontiguousarray(vi[i]); gt = np.asarray(vm[i])
        rows.append(panel(img, gt, seg.segment(img), f"{tag} #{i}"))
    cv2.imwrite(os.path.join(outdir, "duts_examples.png"), pack(rows, per_row=2))


def stages_figure(seg, outdir):
    """Photo -> superpixels -> region probability -> cut -> final mask."""
    os.makedirs(outdir, exist_ok=True)
    vi, vm = data.load_cache("val")
    img = np.ascontiguousarray(vi[7]); gt = np.asarray(vm[7])
    res = seg.work_res
    d = seg.regions(img)
    lab2d = d["labels"].reshape(res, res)
    # superpixel boundaries
    bnd = np.zeros((res, res), bool)
    bnd[:, :-1] |= lab2d[:, :-1] != lab2d[:, 1:]
    bnd[:-1, :] |= lab2d[:-1, :] != lab2d[1:, :]
    small = cv2.resize(img, (res, res), interpolation=cv2.INTER_AREA)
    sp = small.copy(); sp[bnd] = (0, 255, 255)
    sp = cv2.resize(sp, (256, 256), interpolation=cv2.INTER_NEAREST)
    # region probability map
    import regions as R
    prob = seg.region_prob(d["feats"])
    pm = (prob[d["labels"]].reshape(res, res) * 255).astype(np.uint8)
    pm = cv2.applyColorMap(cv2.resize(pm, (256, 256),
                                      interpolation=cv2.INTER_NEAREST),
                           cv2.COLORMAP_VIRIDIS)
    # cut result at working resolution
    is_obj = seg.cut(d, prob)
    cm = cv2.resize((is_obj[d["labels"]].reshape(res, res) * 255).astype(np.uint8),
                    (256, 256), interpolation=cv2.INTER_NEAREST)
    cm = cv2.cvtColor(cm, cv2.COLOR_GRAY2BGR)
    final = overlay(img, seg.segment(img), RED)
    gtv = overlay(img, gt, GREEN)
    tiles, heads = [img, sp, pm, cm, final, gtv], \
        ["1. input 256x256", f"2. {d['nreg']} regions", "3. region P(object)",
         "4. after min-cut", "5. final mask", "ground truth"]
    out = np.hstack([np.vstack([label_strip(256, h), t])
                     for t, h in zip(tiles, heads)])
    cv2.imwrite(os.path.join(outdir, "stages.png"), out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default="params/model.npz")
    ap.add_argument("--outdir", default="out/figs")
    a = ap.parse_args()
    seg = Segmenter(load_params(a.params))
    stages_figure(seg, a.outdir)
    duts_examples(seg, a.outdir)
    meta = captured_figures(seg, a.outdir)
    json.dump(meta, open(os.path.join(a.outdir, "captured_meta.json"), "w"), indent=2)
    print(f"figures -> {a.outdir}")
    for m in meta:
        print(f"  {m['name']:16s} mIoU {m['miou']:.3f}  BF1 {m['bf1']:.3f}")

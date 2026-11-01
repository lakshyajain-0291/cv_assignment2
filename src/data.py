"""DUTS loading, the fixed resize rule, and an on-disk 256x256 cache.

Resizing is not part of the timed pipeline, so this stage is allowed to use
multiple processes. Everything downstream reads the cached uint8 arrays, which
keeps training and evaluation from re-decoding JPEGs.
"""
import os
import numpy as np
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
CACHE = os.path.join(DATA, "cache")
SIZE = 256
SEED = 1088


def resize_image(bgr):
    """Section 1b: photos to 256x256 with INTER_AREA (squashing is intended)."""
    return cv2.resize(bgr, (SIZE, SIZE), interpolation=cv2.INTER_AREA)


def resize_mask(gray):
    """Section 1b: masks with INTER_NEAREST; object iff value >= 128."""
    m = cv2.resize(gray, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
    return (m >= 128).astype(np.uint8)


def splits():
    """Deterministic 80/20 split of DUTS-TR into train and validation."""
    tr_dir = os.path.join(DATA, "DUTS-TR", "DUTS-TR-Image")
    names = sorted(n[:-4] for n in os.listdir(tr_dir) if n.endswith(".jpg"))
    rng = np.random.RandomState(SEED)
    idx = rng.permutation(len(names))
    n_val = int(round(0.20 * len(names)))
    val = [names[i] for i in idx[:n_val]]
    train = [names[i] for i in idx[n_val:]]
    return train, val


def test_names():
    te_dir = os.path.join(DATA, "DUTS-TE", "DUTS-TE-Image")
    return sorted(n[:-4] for n in os.listdir(te_dir) if n.endswith(".jpg"))


def _pair_paths(name, split):
    sub = "DUTS-TR" if split == "tr" else "DUTS-TE"
    return (os.path.join(DATA, sub, f"{sub}-Image", name + ".jpg"),
            os.path.join(DATA, sub, f"{sub}-Mask", name + ".png"))


def _load_one(args):
    name, split = args
    ip, mp = _pair_paths(name, split)
    img = cv2.imread(ip, cv2.IMREAD_COLOR)
    msk = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
    if img is None or msk is None:
        return None
    return resize_image(img), resize_mask(msk)


def build_cache(tag, names, split, workers=8):
    """Write <tag>_img.npy (N,256,256,3) uint8 and <tag>_msk.npy (N,256,256)."""
    os.makedirs(CACHE, exist_ok=True)
    ip = os.path.join(CACHE, f"{tag}_img.npy")
    mp = os.path.join(CACHE, f"{tag}_msk.npy")
    if os.path.exists(ip) and os.path.exists(mp):
        return ip, mp
    n = len(names)
    imgs = np.lib.format.open_memmap(ip, mode="w+", dtype=np.uint8,
                                     shape=(n, SIZE, SIZE, 3))
    msks = np.lib.format.open_memmap(mp, mode="w+", dtype=np.uint8,
                                     shape=(n, SIZE, SIZE))
    from multiprocessing import Pool
    with Pool(workers) as pool:
        for i, r in enumerate(pool.imap(_load_one,
                                        [(nm, split) for nm in names],
                                        chunksize=32)):
            if r is not None:
                imgs[i], msks[i] = r
            if (i + 1) % 1000 == 0:
                print(f"  {tag}: {i+1}/{n}", flush=True)
    imgs.flush(); msks.flush()
    return ip, mp


def load_cache(tag):
    return (np.load(os.path.join(CACHE, f"{tag}_img.npy"), mmap_mode="r"),
            np.load(os.path.join(CACHE, f"{tag}_msk.npy"), mmap_mode="r"))

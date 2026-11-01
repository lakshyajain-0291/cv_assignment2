# CSL7360 Assignment 2 — Cost-Efficient Segmentation (Track A)

Lakshya Jain (B23CS1032). Classical pipeline: grid-constrained K-Means → learned
region model → min-cut via Dinic's algorithm. No pretrained weights, no library
code for segmentation, superpixels, clustering, max-flow, morphology or metrics.

## Results (DUTS-TE, 5,019 photos)

| | mIoU | BF1 | Q | CES |
|---|---|---|---|---|
| validation (2,111) | 0.7305 | 0.4660 | 0.6512 | 0.5896 |
| **DUTS-TE (5,019)** | 0.6115 | 0.3737 | 0.5402 | **0.4891** |
| captured (8) | 0.7351 | 0.4024 | 0.6353 | 0.5752 |

`T_CPU = 23.62 ms` (limit 50) · `S_disk = 6,238 bytes` · CPU: Intel Core i5-1340P

## Required entry point

```bash
python predict.py --image photo.jpg --out mask.png
# T_CPU_ms: 24.10   (varies with the photo; 23.62 is the DUTS-TE mean)
# S_disk_bytes: 6238
```

## Layout

```
predict.py              required CLI
params/model.npz        all learned values (6.2 KB)
src/
  fastenv.py            single-thread pinning (imported before numpy/cv2)
  kmeans.py             grid-constrained K-Means, from scratch
  maxflow.py            Dinic's max-flow / min-cut, from scratch
  regions.py            region adjacency + 17 cues + quadratic basis
  pipeline.py           the timed segment() path
  metrics.py            mIoU and Boundary-F1, from scratch
  data.py               DUTS loading, resize rule, 256x256 cache
  train.py              colour/position stats, logistic regression, tuning
  retune.py             extend the lambda/sigma grid on an existing model
  evaluate.py           val + DUTS-TE + captured metrics, and timing
  oracle.py             region-oracle ceiling diagnostic
  figures.py            report figures
  annotate.py           polygon mask annotator for the captured photos
captured/
  original/ images_256/ masks_256/
report/report.tex       report source -> report.pdf
finalize.sh             rebuild metrics, figures and PDF
```

## Reproducing from scratch

```bash
python src/build_cache.py                   # resize DUTS into a cache
python src/train.py --out params/model.npz  # stats, features, logreg, tuning
python src/retune.py                        # wider lambda/sigma grid
./finalize.sh                               # evaluate, figures, PDF
```

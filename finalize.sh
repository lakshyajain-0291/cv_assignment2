#!/usr/bin/env bash
# Regenerate everything that depends on the captured masks, then rebuild the PDF.
set -e
cd "$(dirname "$0")"
PY=.venv/bin/python
echo "[1/4] evaluating (val + DUTS-TE + captured) and timing ..."
$PY src/evaluate.py --params params/model.npz --out out/results.json
echo "[2/4] figures ..."
$PY src/figures.py --params params/model.npz --outdir out/figs
echo "[3/4] LaTeX numbers ..."
$PY src/make_numbers.py > /dev/null
echo "[4/4] compiling report ..."
cd report && pdflatex -interaction=nonstopmode report.tex > /dev/null 2>&1 \
          && pdflatex -interaction=nonstopmode report.tex > /dev/null 2>&1
echo "done -> report/report.pdf"

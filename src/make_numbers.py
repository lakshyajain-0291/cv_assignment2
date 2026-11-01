"""Emit LaTeX macros from results.json so the report never hard-codes a number."""
import json, os, sys, glob

R = json.load(open("out/results.json"))
cap_meta = []
p = "out/figs/captured_meta.json"
if os.path.exists(p):
    cap_meta = json.load(open(p))

def f(x, n=4):
    return f"{x:.{n}f}"

L = []
def mac(name, val):
    L.append(f"\\newcommand{{\\{name}}}{{{val}}}")

sp = json.load(open("data/cache/splits.json"))
mac("NtrainSplit", f"{sp['n_train_split']:,}".replace(",", "{,}"))
mac("NvalSplit", f"{sp['n_val_split']:,}".replace(",", "{,}"))
mac("Nfit", f"{sp['n_fit']:,}".replace(",", "{,}"))
mac("Ntest", f"{sp['n_test']:,}".replace(",", "{,}"))

for tag, key in (("val", "val"), ("te", "test"), ("cap", "captured")):
    d = R.get(key)
    if not d:
        for s in ("MIoU", "IoUbg", "IoUob", "BF", "Q", "CES"):
            mac(tag + s, "--")
        mac("N" + tag, "--")
        continue
    mac(tag + "MIoU", f(d["miou"]))
    mac(tag + "IoUbg", f(d["iou_bg"]))
    mac(tag + "IoUob", f(d["iou_obj"]))
    mac(tag + "BF", f(d["bf1"]))
    mac(tag + "Q", f(d["Q"]))
    mac(tag + "CES", f(d.get("CES", 0.0)))
    mac("N" + tag, str(d["n"]))

# penalty terms, derived rather than hard-coded so they cannot drift
_tp = 0.20 * R["T_CPU_ms"] / 50.0
_sp = 0.10 * min(1.0, R["S_disk_bytes"] / 4194304.0)
mac("timePen", f(_tp, 4))
mac("sizePen", f(_sp, 5))
mac("cesMult", f(1.0 - _tp - _sp, 4))
mac("timePenPct", f(100 * _tp, 1))

if os.path.exists("out/phases.json"):
    ph = json.load(open("out/phases.json"))
    mac("phaseMean", f(ph["phase_mean"], 1))
    mac("phasePfive", f(ph["phase_p5"], 0))
    mac("phasePnine", f(ph["phase_p95"], 0))

mac("Tcpu", f(R["T_CPU_ms"], 2))
mac("Tpfifty", f(R.get("T_CPU_p50", 0), 2))
mac("Tpninenine", f(R.get("T_CPU_p99", 0), 2))
mac("Tmax", f(R.get("T_CPU_max", 0), 2))
mac("Ntimed", f"{R['n_timed']:,}".replace(",", "{,}"))
mac("Sdisk", f"{R['S_disk_bytes']:,}".replace(",", "{,}"))
mac("SdiskMiB", f(R["S_disk_bytes"] / 1048576.0, 3))

import numpy as np
z = np.load("params/model.npz")
mac("workRes", str(int(z["work_res"])))
mac("gridG", str(int(z["grid"])))
mac("nRegions", str(int(z["grid"]) ** 2))
mac("kmIters", str(int(z["kmeans_iters"])))
mac("lamVal", f(float(z["lam"]), 0))
mac("sigmaMult", f(float(z["sigma_mult"]), 2))
mac("compactness", f(float(z["compactness"]), 0))

# per-captured-photo rows
rows = []
for m in cap_meta:
    rows.append(f"{m['name'].replace('_',chr(92)+'_')} & {m['miou']:.3f} & "
                f"{m['bf1']:.3f} & {100*m['obj_frac']:.1f}\\% \\\\")
open("report/captured_rows.tex", "w").write("\n".join(rows) if rows else
    "\\multicolumn{4}{c}{\\emph{masks not yet annotated}} \\\\")

open("report/numbers.tex", "w").write("\n".join(L) + "\n")
print("\n".join(L))

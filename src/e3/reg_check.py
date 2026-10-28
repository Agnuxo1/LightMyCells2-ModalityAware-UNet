"""Protocol §8.3 — per-study spatial registration of TL↔fluorescence.

With the baseline: phase correlation (skimage) between the prediction and the nucleus GT (or mitochondria
if there is no nucleus) at the middle plane, up to N FOV per study (all folds: the shift
is a property of the acquisition, not of the model). A median shift |d| > 2 px with low
dispersion indicates systematic misalignment of the study.
Usage: python reg_check.py --index index.json --weights unet_l1.pt --per-study 12 --out reg.json
"""
import argparse
import json
import random
import statistics as stt
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from skimage.registration import phase_cross_correlation

from lmc_common import pnorm, read_ome
from lmc_model import UNet, predict_full

ap = argparse.ArgumentParser()
ap.add_argument("--index", required=True)
ap.add_argument("--weights", required=True)
ap.add_argument("--per-study", type=int, default=12)
ap.add_argument("--out", required=True)
ap.add_argument("--val-fold", type=int, default=0)
a = ap.parse_args()
dev = "cuda"
m = UNet().to(dev).eval()
m.load_state_dict(torch.load(a.weights, map_location="cpu", weights_only=True)["model"])
idx = json.loads(Path(a.index).read_text())
by = defaultdict(list)
for k, e in sorted(idx.items()):
    by[e["study"]].append((k, e))
rng = random.Random(2026)
res = {}
for st, items in sorted(by.items()):
    shifts, unseen = [], []
    for k, e in rng.sample(items, min(a.per_study, len(items))):
        org = "Nucleus" if "Nucleus" in e["orgs"] else ("Mitochondria" if "Mitochondria" in e["orgs"] else None)
        if not org:
            continue
        mod = sorted(e["tl"])[0]
        planes = e["tl"][mod]
        tl, px = read_ome(planes[len(planes) // 2])
        gt = read_ome(e["orgs"][org])[0]
        if gt.shape != tl.shape:
            continue
        pred = predict_full(m, tl, px, mod, dev)[["Nucleus", "Mitochondria", "Tubulin", "Actin"].index(org)]
        sh, err, _ = phase_cross_correlation(pnorm(gt, sub=1), pnorm(pred, sub=1), upsample_factor=10)
        shifts.append([float(sh[0]), float(sh[1])])
        unseen.append(e["fold"] in ("ood", a.val_fold))
    if shifts:
        dy = [s[0] for s in shifts]
        dx = [s[1] for s in shifts]
        med = (stt.median(dy), stt.median(dx))
        res[str(st)] = {"n": len(shifts), "med_dy": med[0], "med_dx": med[1],
                        "mad": stt.median([abs(y - med[0]) + abs(x - med[1]) for y, x in shifts]),
                        "flag": bool(np.hypot(*med) > 2),
                        "frac_no_visto_en_train": sum(unseen) / len(unseen)}
        print(st, res[str(st)], flush=True)
Path(a.out).write_text(json.dumps(res, indent=1))
print("estudios marcados (|d|>2 px):", [s for s, r in res.items() if r["flag"]])

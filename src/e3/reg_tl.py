"""Protocol §8.3 — TL↔fluorescence registration on RAW DATA (no model).

Corrects reg_check.py: that one measured the shift between the model PREDICTION and the GT, and a model
trained on misaligned data reproduces that same misalignment (bias towards zero). Here the transmitted
light is compared directly with the fluorescence, on gradients, at 0.2 µm/pixel, with no model.

Null control: each TL is also paired with the GT of ANOTHER specimen of the same organelle. If the
true registration error is not clearly smaller than the null, the test measures nothing.

Usage: python reg_tl.py --index index.json --per-study 12 --out resultados/reg_tl.json
"""
import argparse
import json
import random
import statistics as stt
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import ndimage
from skimage.filters import sobel
from skimage.registration import phase_cross_correlation

from lmc_common import pnorm, read_ome
from lmc_model import rescale

SEED = 2026
MAX_SHIFT = 2.0  # pixels at 0.2 µm (0.4 µm)


def feat(img, px):
    x = pnorm(rescale(img.astype(np.float32), px), sub=1)
    f = sobel(ndimage.gaussian_filter(x, 2.0))
    h, w = f.shape
    return f[int(.15 * h):int(.85 * h), int(.15 * w):int(.85 * w)]


def center(a, h, w):
    y, x = (a.shape[0] - h) // 2, (a.shape[1] - w) // 2
    return a[y:y + h, x:x + w]


def pair(gt_f, tl_f):
    # Pixel rounding when moving to 0.2 µm: centred crop to a common frame
    h, w = min(gt_f.shape[0], tl_f.shape[0]), min(gt_f.shape[1], tl_f.shape[1])
    gt_f, tl_f = center(gt_f, h, w), center(tl_f, h, w)
    sh, err, _ = phase_cross_correlation(gt_f, tl_f, upsample_factor=10, normalization=None)
    return float(sh[0]), float(sh[1]), float(err)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--per-study", type=int, default=12)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = random.Random(SEED)
    idx = json.loads(Path(a.index).read_text())

    # One (TL, GT) pair per FOV: nucleus preferred, otherwise mitochondria
    items = []
    for k, e in sorted(idx.items()):
        org = "Nucleus" if "Nucleus" in e["orgs"] else ("Mitochondria" if "Mitochondria" in e["orgs"] else None)
        if org is None:
            continue
        mod = sorted(e["tl"])[0]
        planes = e["tl"][mod]
        tl, tpx = read_ome(planes[len(planes) // 2])
        gt, gpx = read_ome(e["orgs"][org])
        items.append({"key": k, "study": e["study"], "org": org, "tl": feat(tl, tpx or 0.1),
                      "gt": feat(gt, gpx or tpx or 0.1)})
    print("pares crudos cargados:", len(items), flush=True)

    by_study = defaultdict(list)
    for it in items:
        by_study[it["study"]].append(it)
    res = {}
    for st, its in sorted(by_study.items()):
        sample = rng.sample(its, min(a.per_study, len(its)))
        true, null = [], []
        for it in sample:
            dy, dx, err = pair(it["gt"], it["tl"])
            true.append((dy, dx, err))
            others = [o for o in items if o["org"] == it["org"] and o["study"] != st]
            o = rng.choice(others)
            _, _, errn = pair(it["gt"], o["tl"])  # TL of another specimen against this one's GT
            null.append(errn)
        dys = [t[0] for t in true]
        dxs = [t[1] for t in true]
        med = (stt.median(dys), stt.median(dxs))
        dist = [np.hypot(y - med[0], x - med[1]) for y, x in zip(dys, dxs)]
        consist = sum(d <= MAX_SHIFT for d in dist) / len(dist)
        res[str(st)] = {
            "n": len(sample), "orgs": sorted({it["org"] for it in sample}),
            "med_dy_px": round(med[0], 2), "med_dx_px": round(med[1], 2),
            "frac_consistentes_2px": round(consist, 2),
            "err_verdadero": round(stt.median([t[2] for t in true]), 4),
            "err_nulo": round(stt.median(null), 4),
            "flag_desalineado": bool(np.hypot(*med) > MAX_SHIFT and consist >= 0.5),
            "registro_medible": bool(stt.median(null) - stt.median([t[2] for t in true]) > 0.05),
        }
        print(st, res[str(st)], flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print("marcados desalineados:", [s for s, r in res.items() if r["flag_desalineado"]])
    print("not measurable (null error ~ true error):", [s for s, r in res.items() if not r["registro_medible"]])


if __name__ == "__main__":
    main()

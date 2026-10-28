"""Agrega la validación cruzada (pliegues 0-4): media y sd entre pliegues por split × orgánulo.

Uso: python summarize_cv.py <dir_con_fold*/metrics_rows.csv>
Por pliegue se promedian las filas (FOV × modalidad × plano × orgánulo); luego se resume entre pliegues.
Para 'ood' cada pliegue evalúa el mismo banco fijo con un modelo distinto: la sd mide la variabilidad del entrenamiento.
"""
import csv
import json
import statistics as st
import sys
from pathlib import Path

root = Path(sys.argv[1])
KEYS = ["ssim", "pcc", "ssim_tl_trivial", "pcc_tl_trivial"]
groups = {}
for f in sorted(root.glob("fold*/metrics_rows.csv")):
    fold = int(f.parent.name[4:])
    for r in csv.DictReader(open(f)):
        groups.setdefault((r["split"], r["org"], fold), []).append(r)

per = {}
for (split, org, fold), rs in groups.items():
    per.setdefault((split, org), {})[fold] = {
        **{k: st.mean(float(x[k]) for x in rs) for k in KEYS}, "n_filas": len(rs)}

out = {}
for (split, org), d in sorted(per.items()):
    summ = {}
    for k in KEYS:
        vals = [d[f][k] for f in sorted(d)]
        summ[k] = {"media": round(st.mean(vals), 4),
                   "sd": round(st.stdev(vals), 4) if len(vals) > 1 else None,
                   "por_pliegue": {str(f): round(d[f][k], 4) for f in sorted(d)}}
    out[f"{split}|{org}"] = {"pliegues": sorted(d), **summ}

(root / "cv_resumen.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
for k, v in out.items():
    print(f"{k:24} pliegues={v['pliegues']} ssim={v['ssim']['media']}±{v['ssim']['sd']} "
          f"pcc={v['pcc']['media']}±{v['pcc']['sd']} trivial_ssim={v['ssim_tl_trivial']['media']}")

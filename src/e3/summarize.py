"""Summary of metrics_rows.csv: mean and n per split × organelle, and per stratum (site, modality, z)."""
import csv
import json
import sys
from collections import defaultdict
from statistics import mean

rows = list(csv.DictReader(open(sys.argv[1])))
out = {}
for keys in [("split", "org"), ("split", "org", "site"), ("split", "org", "mod"), ("split", "org", "zpos")]:
    g = defaultdict(list)
    for r in rows:
        nz = int(r["nz"])
        r["zpos"] = "mid" if int(r["z"]) == nz // 2 else ("first" if int(r["z"]) == 0 else "last")
        g[tuple(r[k] for k in keys)].append(r)
    out["|".join(keys)] = {
        "|".join(k): {"n": len(v), "n_fov": len({x["fov"] for x in v}),
                      **{m: round(mean(float(x[m]) for x in v), 4)
                         for m in ["ssim", "pcc", "ssim_raw", "ssim_tl_trivial", "pcc_tl_trivial"]}}
        for k, v in sorted(g.items())}
json.dump(out, open(sys.argv[2], "w"), indent=1, ensure_ascii=False)
for k, v in out["split|org"].items():
    print(k.ljust(20), v)

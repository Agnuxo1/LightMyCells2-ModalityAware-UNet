"""Per-study table (ed.1, S-BIAD1047): REMBI + counts per modality/organelle.

Usage: python3 study_table.py <meta_dir> <output.csv>
Reads study_manifest.json and Study_N.json (FileLists). Stdlib only.
"""
import collections
import csv
import json
import re
import sys
from pathlib import Path

meta = Path(sys.argv[1])
out = Path(sys.argv[2])
d = json.loads((meta / "study_manifest.json").read_text())


def flat(s):
    o = []
    for x in s.get("subsections", []):
        o += x if isinstance(x, list) else [x]
    return o


def attr(s, n):
    return [a["value"] for a in s.get("attributes", []) if a.get("name") == n]


subs = flat(d["section"])
by_type_title = collections.defaultdict(dict)
for s in subs:
    t = attr(s, "Title")
    if t:
        by_type_title[s.get("type")][t[0].strip()] = s

FNAME = re.compile(r"image_(\d+)_(BF|PC|DIC|Nucleus|Mitochondria|Tubulin|Actin)(?:_z(\d+))?\.ome\.tiff$")
rows = []
for sc in [s for s in subs if s.get("type") == "Study Component"]:
    st = re.search(r"Study_\d+", attr(sc, "File List")[0]).group(0)
    assoc = {a["name"]: a["value"].strip() for a in flat(sc)[0].get("attributes", [])}
    bio = by_type_title["Biosample"].get(assoc.get("Biosample", ""), {})
    spe = by_type_title["Specimen"].get(assoc.get("Specimen", ""), {})
    acq = by_type_title["Image acquisition"].get(assoc.get("Image acquisition", ""), {})
    files = json.loads((meta / f"{st}.json").read_text())
    cnt = collections.Counter()
    imgs = set()
    zs = collections.Counter()
    for f in files:
        m = FNAME.search(f["path"])
        if not m:
            cnt["other"] += 1
            continue
        imgs.add(m.group(1))
        cnt[m.group(2)] += 1
        if m.group(3) is not None:
            zs[m.group(2)] += 1
    instr = " ".join(attr(acq, "Imaging instrument"))
    params = " ".join(attr(acq, "Image acquisition parameters"))
    px = re.search(r"pixel size\s*[=:]?\s*([\d.,]+)\s*(nm|µm|um)", params, re.I)
    rows.append({
        "study": st,
        "name": attr(sc, "Name")[0].strip(),
        "biosample": assoc.get("Biosample", ""),
        "organism": "; ".join(attr(bio, "Organism")),
        "entity": "; ".join(attr(bio, "Biological entity"))[:120],
        "specimen": assoc.get("Specimen", ""),
        "acquisition": assoc.get("Image acquisition", ""),
        "methods": "; ".join(attr(acq, "Imaging method")),
        "instrument": instr.replace("\n", " ")[:200],
        "pixel": (px.group(1) + px.group(2)) if px else "",
        "n_fov": len(imgs),
        **{k: cnt.get(k, 0) for k in ["BF", "PC", "DIC", "Nucleus", "Mitochondria", "Tubulin", "Actin", "other"]},
        "n_files": len(files),
        "bytes": sum(f.get("size", 0) for f in files),
    })

rows.sort(key=lambda r: int(r["study"].split("_")[1]))
with out.open("w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(f"{len(rows)} estudios -> {out}")

"""E2 — frozen validation splits for Light My Cells 2 (design C, decided by JEV 0.93).

Usage: python3 make_splits.py <meta_dir> <output_dir>

Design (fixed before seeing results):
  * Unit = field (FOV) = (study, image_id). A FOV is never split.
  * Biological group = connected component of studies sharing a specimen (normalised REMBI
    title). No group is split across folds, not even the giant specimens.
  * Amendment 1 (25-10): the giant studies (25, 26, 27) are NOT chunked; each group goes whole to
    one fold. v1 chunked them and caused study leakage (found by Opus statistical review).
  * Fixed OOD bank = small complete sites covering BF/PC/DIC and nucleus/mito/tubulin; never
    trained on during model selection (but used in the final model).
  * K=5 folds on the rest, deterministic greedy assignment (seed 2026) balancing
    counts per organelle and modality.
  * Actin: additionally, 2-fold leave-one-specimen-out (JEV: "both", 0.45).
  * Ed.2 extension: keep v2; a new biological group goes whole to fold
    int(sha256(norm(specimen)+"|2026"), 16) % 5. No giant is chunked. See RULE_ED2.
"""
import collections
import datetime
import hashlib
import json
import random
import re
import sys
from pathlib import Path

SEED = 2026
K = 5
GIANT_FOV = 150
ORGS = ["Nucleus", "Mitochondria", "Tubulin", "Actin"]
MODS = ["BF", "PC", "DIC"]
# INFERRED acquisition site (not provided as a field): from study name and instrument.
SITES = {
    "U2OS_LeicaDMi8": [1], "CRSA_FV3000": [2], "Genethon_SP8": [3], "Roussy_DMI8": [4],
    "AMU_AxioZ1": [5, 6, 7], "NikonTiE": list(range(8, 16)), "ZeissAO": [16, 17],
    "NikonTi2": list(range(18, 25)), "IGH": [25, 26, 27, 29, 30], "HomeMade_Scan": [28],
}
OOD_SITES = ["ZeissAO", "HomeMade_Scan", "U2OS_LeicaDMi8"]
RULE_ED2 = ("Preserve v2 group assignments, never v1. A new biological group is assigned whole by "
            "int(sha256(norm(specimen)+'|2026'),16) % 5; never split giant studies or shared "
            "specimens. Existing group membership overrides the new-group hash; OOD bank unchanged.")
FNAME = re.compile(r"Study_(\d+)/image_(\d+)_(BF|PC|DIC|Nucleus|Mitochondria|Tubulin|Actin)(?:_z(\d+))?\.ome\.tiff$")


def norm(s):
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def flat(s):
    o = []
    for x in s.get("subsections", []):
        o += x if isinstance(x, list) else [x]
    return o


def attr(s, n):
    return [a["value"] for a in s.get("attributes", []) if a.get("name") == n]


def main(meta, outdir):
    site_of = {st: name for name, sts in SITES.items() for st in sts}
    d = json.loads((meta / "study_manifest.json").read_text())
    spec_of = {}
    for sc in [s for s in flat(d["section"]) if s.get("type") == "Study Component"]:
        st = int(re.search(r"Study_(\d+)", attr(sc, "File List")[0]).group(1))
        assoc = {a["name"]: a["value"] for a in flat(sc)[0].get("attributes", [])}
        spec_of[st] = norm(assoc["Specimen"])
    assert sorted(spec_of) == list(range(1, 31)), sorted(spec_of)
    assert sorted(site_of) == list(range(1, 31))

    # FOV -> modalities (no. of planes) and available organelles
    fov = collections.defaultdict(lambda: {"mods": collections.Counter(), "orgs": set()})
    for st in spec_of:
        for f in json.loads((meta / f"Study_{st}.json").read_text()):
            m = FNAME.search(f["path"])
            if not m:
                continue
            key = f"S{int(m.group(1))}_{int(m.group(2))}"
            if m.group(3) in MODS:
                fov[key]["mods"][m.group(3)] += 1
            else:
                fov[key]["orgs"].add(m.group(3))
    fov = {k: v for k, v in fov.items() if v["orgs"] and v["mods"]}

    # Groups = connected components by shared specimen (union-find over studies)
    parent = {st: st for st in spec_of}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    by_spec = collections.defaultdict(list)
    for st, sp in spec_of.items():
        by_spec[sp].append(st)
    for sts in by_spec.values():
        for st in sts[1:]:
            parent[find(st)] = find(sts[0])
    comp = {st: find(st) for st in spec_of}
    # A group cannot span OOD/non-OOD sites (if it did, the whole group goes to OOD)
    comp_sites = collections.defaultdict(set)
    for st in spec_of:
        comp_sites[comp[st]].add(site_of[st])

    assign = {}
    units = collections.defaultdict(list)  # assignable unit -> FOVs
    for k in sorted(fov, key=lambda k: (int(k.split("_")[0][1:]), int(k.split("_")[1]))):
        st = int(k.split("_")[0][1:])
        if comp_sites[comp[st]] & set(OOD_SITES):
            assign[k] = "ood"
            continue
        units[f"G{comp[st]}"].append(k)
    rng = random.Random(SEED)
    # Amendment 1 (25-10, JEV C 0.68): giants are NOT chunked; each group (study/specimen)
    # goes whole to one fold. The v1 chunking put the same study in training and validation.
    giant_blocks = []
    order = rng.sample(range(K), K)
    def vec(ks):
        c = collections.Counter()
        for k in ks:
            for o in fov[k]["orgs"]:
                c[o] += 1
            for m in fov[k]["mods"]:
                c[m] += 1
        return c

    fold_load = [vec([k for k, f in assign.items() if f == i]) for i in range(K)]
    total = vec(list(fov))
    keys = ORGS + MODS
    # Greedy: large units first (seeded tie-break); minimises relative deviation
    rest = sorted(units.items(), key=lambda u: (-len(u[1]), rng.random()))
    for name, ks in rest:
        v = vec(ks)
        best = None
        for i in range(K):
            cost = 0.0
            for key in keys:
                if total[key] == 0:
                    continue
                loads = [fold_load[j][key] + (v[key] if j == i else 0) for j in range(K)]
                cost += sum((x / total[key] - 1 / K) ** 2 for x in loads)
            if best is None or cost < best[0] - 1e-12:
                best = (cost, i)
        for k in ks:
            assign[k] = best[1]
        fold_load[best[1]] += v

    # Actin LOSO: one fold per specimen with actin
    act_specs = sorted({spec_of[int(k.split("_")[0][1:])] for k in fov if "Actin" in fov[k]["orgs"]})
    actin_loso = {k: act_specs.index(spec_of[int(k.split("_")[0][1:])])
                  for k in fov if "Actin" in fov[k]["orgs"]}

    # Integrity checks
    assert set(assign) == set(fov)
    giants = set()  # amendment 1: there are no longer any chunked giants
    spec_folds = collections.defaultdict(set)
    for k, f in assign.items():
        spec_folds[f"G{comp[int(k.split('_')[0][1:])]}"].add(f)
    for g, fs in spec_folds.items():
        assert len(fs) == 1, f"group {g} split across {fs}"  # no study or specimen leakage

    # Per-fold summary
    summary = {}
    for f in list(range(K)) + ["ood"]:
        ks = [k for k, v in assign.items() if v == f]
        c = vec(ks)
        summary[str(f)] = {"fov": len(ks), **{x: c[x] for x in keys},
                           "studies": sorted({int(k.split("_")[0][1:]) for k in ks})}
    body = {
        "version": "v2-enmienda1-ed1", "seed": SEED, "k": K, "design": "C-v2 StratifiedGroupKFold por grupo (enmienda 1, JEV 0.68)",
        "unit": "FOV = S<study>_<image_id>", "giant_threshold_fov": GIANT_FOV,
        "sites_inferred": SITES, "ood_sites": OOD_SITES, "rule_ed2": RULE_ED2,
        "specimen_of_study": {str(k): v for k, v in sorted(spec_of.items())},
        "group_of_study": {str(k): f"G{v}" for k, v in sorted(comp.items())},
        "assign": dict(sorted(assign.items())), "actin_loso": dict(sorted(actin_loso.items())),
        "actin_loso_specimens": act_specs, "summary": summary,
    }
    canon = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    body["sha256"] = hashlib.sha256(canon.encode()).hexdigest()
    body["created_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "splits.json").write_text(json.dumps(body, indent=1, ensure_ascii=False))
    print("sha256", body["sha256"])
    for f, s in summary.items():
        print(f.rjust(4), json.dumps(s, ensure_ascii=False))
    print("actin_loso", act_specs, collections.Counter(actin_loso.values()))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))

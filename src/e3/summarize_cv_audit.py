"""Audit CV coverage and aggregate z -> modality -> FOV, with equal-FOV/equal-study estimands.

Leaves the original row-weighted summary untouched. OOD is the same bank evaluated
by five models; it is never counted as five independent sets of specimens.
Usage: python summarize_cv_audit.py RESULTS --index index.json --out cv_audit.json
"""
import argparse
import collections
import csv
import hashlib
import json
import math
import statistics as st
from pathlib import Path

KEYS = ("ssim", "pcc", "ssim_raw", "ssim_tl_trivial", "pcc_tl_trivial")
ORGS = ("Nucleus", "Mitochondria", "Tubulin", "Actin")


def mean_values(items):
    out = {}
    for k in KEYS:
        vals = [x[k] for x in items if x[k] is not None and math.isfinite(x[k])]
        out[k] = st.mean(vals) if vals else None
    return out


def expected_rows(index, fold):
    out = set()
    for fov, e in index.items():
        if e["fold"] not in (fold, "ood"):
            continue
        split = "ood" if e["fold"] == "ood" else "val"
        for mod, planes in e["tl"].items():
            for z in sorted({0, len(planes) // 2, len(planes) - 1}):
                for org in e["orgs"]:
                    out.add((split, fov, mod, z, org))
    return out


def aggregate(root, index, constant_gt=()):
    observed = collections.defaultdict(list)
    coverage, files, metric_warnings = {}, {}, []
    for fold in range(5):
        p = root / f"fold{fold}" / "metrics_rows.csv"
        expected = expected_rows(index, fold)
        seen, duplicates, unexpected, invalid = set(), [], [], []
        rows = list(csv.DictReader(p.open(encoding="utf-8", newline=""))) if p.exists() else []
        for r in rows:
            try:
                key = (r["split"], r["fov"], r["mod"], int(r["z"]), r["org"])
                vals = {k: float(r[k]) for k in KEYS}
                if (r["fov"], r["org"]) in constant_gt:
                    metric_warnings.append({"fold":fold,"key":key,"reason":"constant reference verified in source TIFF; all reference-based metrics undefined"})
                    vals = {k:float("nan") for k in KEYS}
                for k, v in vals.items():
                    if not math.isfinite(v):
                        metric_warnings.append({"fold":fold,"key":key,"metric":k,"value":str(v)})
                        vals[k] = None
                e = index[r["fov"]]
                if int(r["study"]) != int(e["study"]) or r["site"] != e["site"]:
                    raise ValueError("metadata mismatch")
                if int(r["nz"]) != len(e["tl"][r["mod"]]):
                    raise ValueError("z-stack length mismatch")
            except (ValueError, KeyError, TypeError) as exc:
                invalid.append(str(exc))
                continue
            if key in seen:
                duplicates.append(key)
                continue
            seen.add(key)
            if key not in expected:
                unexpected.append(key)
                continue
            observed[(r["split"], r["org"], fold, r["fov"], r["mod"])].append(vals)
        missing = expected - seen
        coverage[str(fold)] = {
            "expected_rows": len(expected), "observed_rows": len(rows),
            "missing": len(missing), "unexpected": len(unexpected),
            "duplicates":len(duplicates), "invalid":len(invalid),
            "complete": not (missing or unexpected or duplicates or invalid),
            "missing_examples": sorted(missing)[:8], "invalid_examples": invalid[:8],
        }
        if p.exists():
            files[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()

    # Each modality first gets one vote per FOV, independent of z-stack length.
    fov_mod = {key: mean_values(vals) for key, vals in observed.items()}
    fov_groups = collections.defaultdict(list)
    for (split, org, fold, fov, mod), vals in fov_mod.items():
        fov_groups[(split, org, fold, fov)].append(vals)
    fov_scores = {key: mean_values(vals) for key, vals in fov_groups.items()}
    buckets = collections.defaultdict(dict)
    for (split, org, fold, fov), vals in fov_scores.items():
        buckets[(split, org, fold)][fov] = vals

    def describe(fs):
        by_study = collections.defaultdict(list)
        for fov, vals in fs.items():
            by_study[str(index[fov]["study"])].append(vals)
        return {
            "n_fov": len(fs), "n_studies":len(by_study),
            "valid_fov_per_metric":{k:sum(v[k] is not None for v in fs.values()) for k in KEYS},
            "fov_mean": mean_values(list(fs.values())),
            "study_mean": mean_values([mean_values(v) for v in by_study.values()]),
            "by_study": {s:{"n_fov":len(v), **mean_values(v)} for s,v in sorted(by_study.items())},
        }

    per_fold = {f"{split}|{org}|{fold}":describe(fs) for (split,org,fold),fs in sorted(buckets.items())}
    val, ood = {}, {}
    for org in ORGS:
        fs = {}
        fold_means = []
        ood_models = []
        unique_ood = set()
        for fold in range(5):
            vf = buckets.get(("val",org,fold),{})
            assert not (set(fs) & set(vf)), "FOV appears in multiple validation folds"
            fs.update(vf)
            if vf:
                fold_means.append(mean_values(list(vf.values())))
            of = buckets.get(("ood",org,fold),{})
            if of:
                ood_models.append(describe(of))
                unique_ood.update(of)
        if fs:
            val[org] = {**describe(fs), "n_folds_with_org":len(fold_means),
                        "fold_macro_mean":mean_values(fold_means)}
        if ood_models:
            ood[org] = {"n_unique_fov":len(unique_ood), "n_model_evaluations":len(ood_models),
                        "model_mean":mean_values([d["fov_mean"] for d in ood_models]),
                        "model_sd":{k:st.stdev([d["fov_mean"][k] for d in ood_models if d["fov_mean"][k] is not None]) if sum(d["fov_mean"][k] is not None for d in ood_models)>1 else None for k in KEYS},
                        "interpretation":"Fixed OOD bank; model variability, not five independent samples."}

    # Audit group separation in the actual job index, not just split labels in a document.
    study_folds = collections.defaultdict(set)
    for e in index.values():
        study_folds[str(e["study"])].add(str(e["fold"]))
    split_check = all(len(fs)==1 for fs in study_folds.values())
    complete = all(d["complete"] for d in coverage.values()) and split_check
    return {"status":("complete_with_metric_warnings" if metric_warnings else "complete") if complete else "INCOMPLETE_DO_NOT_REPORT_AS_FINAL",
            "estimand":"Mean z planes within modality, modalities within FOV, then equal FOV; equal-study sensitivity also reported.",
            "coverage":coverage, "metric_warnings":metric_warnings, "study_disjoint":split_check,
            "study_folds":{k:sorted(v) for k,v in study_folds.items()},
            "val":val, "ood":ood, "per_fold":per_fold, "source_sha256":files,
            "limits":["This is local ed.1 UNet-L1 baseline validation, not an official score or ed.2 validation.",
                      "Actin in a held-out study is untrained when all labelled Actin specimens belong to that fold; inspect coverage before claiming learning.",
                      "Inferred acquisition sites are not verified biological independence."]}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root",type=Path)
    ap.add_argument("--index",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--integrity",type=Path,help="Full TIFF scan JSON, used to flag constant reference images")
    a = ap.parse_args()
    raw = a.index.read_bytes()
    const = set()
    if a.integrity:
        import re
        for entry in json.loads(a.integrity.read_text(encoding="utf-8"))["constant_gt"]:
            m=re.search(r"Study_(\d+)/image_(\d+)_(Nucleus|Mitochondria|Tubulin|Actin)\.ome\.tiff$",entry["path"])
            if m: const.add((f"S{int(m[1])}_{int(m[2])}",m[3]))
    out = aggregate(a.root,json.loads(raw),const)
    out["constant_gt_verified"] = sorted(const)
    out["index_sha256"] = hashlib.sha256(raw).hexdigest()
    a.out.write_text(json.dumps(out,indent=2,ensure_ascii=False,allow_nan=False),encoding="utf-8")
    print(json.dumps({"status":out["status"], "coverage":{f:{k:v for k,v in d.items() if not k.endswith("examples")} for f,d in out["coverage"].items()},
                      "val":{o:{"n_fov":d["n_fov"],"ssim":d["fov_mean"]["ssim"],"pcc":d["fov_mean"]["pcc"]} for o,d in out["val"].items()}},indent=2))
    return 0 if out["status"].startswith("complete") else 2


if __name__=="__main__":
    raise SystemExit(main())

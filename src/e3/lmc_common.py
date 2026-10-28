"""Common LMC2 utilities: OME-TIFF reading, normalisation, FOV index."""
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import tifffile

ORGS = ["Nucleus", "Mitochondria", "Tubulin", "Actin"]
MODS = ["BF", "PC", "DIC"]
FNAME = re.compile(r"Study_(\d+)/image_(\d+)_(BF|PC|DIC|Nucleus|Mitochondria|Tubulin|Actin)(?:_z(\d+))?\.ome\.tiff$")
UNIT_TO_UM = {"nm": 1e-3, "µm": 1.0, "um": 1.0, "mm": 1e3}


def read_ome(path):
    """Return (2D float32 image, pixel size in µm or None)."""
    with tifffile.TiffFile(path) as tf:
        arr = tf.asarray()
        px = None
        xml = tf.ome_metadata
        if xml:
            try:
                root = ET.fromstring(xml)
                for el in root.iter():
                    if el.tag.endswith("Pixels") and el.get("PhysicalSizeX"):
                        px = float(el.get("PhysicalSizeX")) * UNIT_TO_UM.get(el.get("PhysicalSizeXUnit", "µm"), 1.0)
                        break
            except ET.ParseError:
                px = None
    arr = np.squeeze(arr)
    if arr.ndim != 2:  # safety check: first page is 2D
        arr = arr.reshape(-1, *arr.shape[-2:])[0]
    return arr.astype(np.float32), px


def read_ome_full(path):
    """For the container: (2D or 3D ZYX float32 array, pixel size in µm or None, channel name or None)."""
    with tifffile.TiffFile(path) as tf:
        arr = np.squeeze(tf.asarray()).astype(np.float32)
        xml = tf.ome_metadata or ""
    px = chan = None
    if xml:
        try:
            root = ET.fromstring(xml)
            for el in root.iter():
                if el.tag.endswith("Pixels") and el.get("PhysicalSizeX") and px is None:
                    px = float(el.get("PhysicalSizeX")) * UNIT_TO_UM.get(el.get("PhysicalSizeXUnit", "µm"), 1.0)
                if el.tag.endswith("Channel") and chan is None:
                    chan = el.get("Name") or el.get("Fluor")
        except ET.ParseError:
            pass
    if arr.ndim > 3:
        arr = arr.reshape(-1, *arr.shape[-2:])
    return arr, px, chan


def modality_from(name):
    n = (name or "").upper()
    for m in ("DIC", "PC", "BF"):
        if m in n:
            return m
    if "PHASE" in n:
        return "PC"
    return None


def pnorm(img, lo=2.0, hi=99.8, sub=4):
    """Percentile scaling (like MONAI's ScaleIntensityRangePercentiles, without clipping)."""
    s = img[::sub, ::sub] if sub > 1 else img
    a, b = np.percentile(s, [lo, hi])
    return (img - a) / max(b - a, 1e-6)


def build_index(meta_dir, data_dir, splits_path):
    """FOV index with complete files (size = FileList)."""
    meta_dir, data_dir = Path(meta_dir), Path(data_dir)
    splits = json.loads(Path(splits_path).read_text())
    site_of = {st: name for name, sts in splits["sites_inferred"].items() for st in sts}
    fovs = {}
    for st in range(1, 31):
        for f in json.loads((meta_dir / f"Study_{st}.json").read_text()):
            m = FNAME.search(f["path"])
            if not m:
                continue
            p = data_dir / f["path"]
            if not p.exists() or p.stat().st_size != f["size"]:
                continue
            key = f"S{int(m.group(1))}_{int(m.group(2))}"
            e = fovs.setdefault(key, {"study": st, "site": site_of[st], "tl": {}, "orgs": {}})
            if m.group(3) in MODS:
                e["tl"].setdefault(m.group(3), []).append((int(m.group(4) or 0), str(p)))
            else:
                e["orgs"][m.group(3)] = str(p)
    out = {}
    for k, e in fovs.items():
        if not e["tl"] or not e["orgs"] or k not in splits["assign"]:
            continue
        e["tl"] = {mod: [p for _, p in sorted(v)] for mod, v in e["tl"].items()}
        e["fold"] = splits["assign"][k]
        out[k] = e
    return out

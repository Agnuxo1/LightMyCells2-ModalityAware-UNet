"""Export small out-of-fold showcase arrays for README figures (CPU, public ed.1 data only).

Each FOV is predicted by the single fold model whose validation fold contains it, so the
model never saw that FOV (or its specimen group) during training. Output: one compressed
.npz per FOV with centre crops of the transmitted-light planes, per-plane predictions and
ground-truth fluorescence. No full-resolution images leave the server.

Usage: python make_showcase_arrays.py <index.json> <model_dir> <out_dir> FOV [FOV ...]
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / "src" / "e3"), str(HERE.parent / "src" / "submission")]

from lmc_common import ORGS, read_ome  # noqa: E402
from lmc_model import UNet, rescale  # noqa: E402
from inference import predict_normalized  # noqa: E402

CROP = 768


def centre(a, size=CROP):
    h, w = a.shape[-2:]
    s = min(size, h, w)
    y, x = (h - s) // 2, (w - s) // 2
    return a[..., y:y + s, x:x + s]


def main():
    index = json.loads(Path(sys.argv[1]).read_text())
    model_dir, out = Path(sys.argv[2]), Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    for key in sys.argv[4:]:
        e = index[key]
        fold = int(e["fold"])
        ck = torch.load(model_dir / f"fold{fold}.pt", map_location="cpu", weights_only=True)
        model = UNet().eval()
        model.load_state_dict(ck["model"])
        mod, paths = next(iter(e["tl"].items()))
        planes, px = [], None
        for p in paths:
            img, px = read_ome(p)
            planes.append(img)
        stack = np.stack(planes)
        # Common stack normalisation, identical to the submitted MIP rule.
        sample = np.concatenate([rescale(p, px)[::4, ::4].ravel() for p in stack])
        lo, hi = np.percentile(sample, [2, 99.8])
        preds = np.stack([predict_normalized([model], (rescale(p, px) - lo) / max(hi - lo, 1e-6), p.shape, mod, "cpu")
                          for p in stack])
        gt = {o: centre(read_ome(e["orgs"][o])[0]) for o in ORGS if o in e["orgs"]}
        np.savez_compressed(out / f"{key}.npz", tl=centre(stack).astype(np.float32), pred=centre(preds).astype(np.float32),
                            gt_names=np.array(list(gt)), gt=np.stack(list(gt.values())).astype(np.float32),
                            meta=json.dumps({"fov": key, "fold": fold, "study": e["study"], "site": e["site"],
                                             "modality": mod, "nz": len(paths), "pixel_um": px, "orgs": ORGS}))
        print(key, fold, mod, stack.shape, flush=True)


if __name__ == "__main__":
    main()

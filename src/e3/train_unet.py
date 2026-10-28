"""E3 — UNet-L1 baseline (TL 1 plane -> 4 fluorescence channels), L1 masked per organelle with GT.

Usage: python train_unet.py --index index.json --val-fold 0 --iters 20000 --out results/
Protocol: seed 2026; selection ONLY on the internal validation; OOD is only reported.
"""
import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from lmc_common import MODS, ORGS, pnorm, read_ome
from lmc_model import PX_TARGET, UNet, predict_full, rescale
from metrics import check_against_skimage, score

SEED = 2026


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


class TrainSet(Dataset):
    def __init__(self, fovs, crop=256, length=10 ** 7):
        self.fovs = fovs
        self.crop = crop
        self.length = length
        cnt = {o: sum(o in e["orgs"] for e in fovs) for o in ORGS}
        # Per-FOV weight: its rarest organelle (balances actin/tubulin against nucleus)
        w = np.array([max(1 / math.sqrt(max(cnt[o], 1)) for o in e["orgs"]) for e in fovs])
        self.p = w / w.sum()

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        rng = np.random.default_rng((SEED * 1_000_003 + i) % 2 ** 32)
        e = self.fovs[rng.choice(len(self.fovs), p=self.p)]
        mod = rng.choice(sorted(e["tl"]))
        tl, px = read_ome(rng.choice(e["tl"][mod]))  # random z plane = simulated defocus
        x = pnorm(rescale(tl, px))
        ys = np.zeros((4,) + x.shape, np.float32)
        mask = np.zeros(4, np.float32)
        for j, o in enumerate(ORGS):
            if o in e["orgs"]:
                y, pyx = read_ome(e["orgs"][o])
                y = rescale(y, pyx or px)
                if y.shape != x.shape:
                    y = F.interpolate(torch.from_numpy(y)[None, None], size=x.shape, mode="bilinear")[0, 0].numpy()
                ys[j] = pnorm(y)
                mask[j] = 1
        c = self.crop
        if min(x.shape) < c:  # reflective padding for small images
            ph, pw = max(0, c - x.shape[0]), max(0, c - x.shape[1])
            x = np.pad(x, ((0, ph), (0, pw)), mode="reflect")
            ys = np.pad(ys, ((0, 0), (0, ph), (0, pw)), mode="reflect")
        r0 = rng.integers(0, x.shape[0] - c + 1)
        c0 = rng.integers(0, x.shape[1] - c + 1)
        x, ys = x[r0:r0 + c, c0:c0 + c], ys[:, r0:r0 + c, c0:c0 + c]
        k = rng.integers(4)
        x, ys = np.rot90(x, k), np.rot90(ys, k, axes=(1, 2))
        if rng.random() < 0.5:
            x, ys = x[:, ::-1], ys[:, :, ::-1]
        onehot = np.zeros((3,) + x.shape, np.float32)
        onehot[MODS.index(mod)] = 1
        inp = np.concatenate([x[None], onehot]).astype(np.float32)
        return torch.from_numpy(inp.copy()), torch.from_numpy(ys.copy()), torch.from_numpy(mask)


def evaluate(model, fovs, dev, tag, writer, max_fov=None):
    model.eval()
    rng = random.Random(SEED)
    items = sorted(fovs, key=lambda e: e["key"])
    if max_fov and len(items) > max_fov:
        items = rng.sample(items, max_fov)
    for e in items:
        gts = {o: read_ome(p)[0] for o, p in e["orgs"].items()}
        for mod, planes in sorted(e["tl"].items()):
            zs = sorted({0, len(planes) // 2, len(planes) - 1})
            for z in zs:
                tl, px = read_ome(planes[z])
                pred = predict_full(model, tl, px, mod, dev)
                for j, o in enumerate(ORGS):
                    if o not in gts:
                        continue
                    gt = gts[o]
                    pj = pred[j] if pred[j].shape == gt.shape else F.interpolate(
                        torch.from_numpy(pred[j])[None, None], size=gt.shape, mode="bilinear")[0, 0].numpy()
                    m = score(pj, gt, dev)
                    tl_r = tl if tl.shape == gt.shape else F.interpolate(
                        torch.from_numpy(tl)[None, None], size=gt.shape, mode="bilinear")[0, 0].numpy()
                    tri_tl = score(tl_r, gt, dev)  # trivial bound: the transmitted light itself
                    writer.writerow({"split": tag, "fov": e["key"], "study": e["study"], "site": e["site"],
                                     "mod": mod, "z": z, "nz": len(planes), "org": o, **m,
                                     "ssim_tl_trivial": tri_tl["ssim"], "pcc_tl_trivial": tri_tl["pcc"]})
    model.train()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--val-fold", type=int, default=0)
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-eval-fov", type=int, default=0)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    set_seed(SEED)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = "cuda"
    print("self-check SSIM/PCC vs skimage: max |diff| =", check_against_skimage(device=dev), flush=True)

    idx = json.loads(Path(a.index).read_text())
    for k, e in idx.items():
        e["key"] = k
    train = [e for e in idx.values() if e["fold"] not in ("ood", a.val_fold)]
    val = [e for e in idx.values() if e["fold"] == a.val_fold]
    ood = [e for e in idx.values() if e["fold"] == "ood"]
    print(f"FOV train={len(train)} val={len(val)} ood={len(ood)}", flush=True)
    assert train and val, "missing data for training/validation"

    model = UNet().to(dev)
    print("parameters:", sum(p.numel() for p in model.parameters()), flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.iters, pct_start=0.05)
    dl = DataLoader(TrainSet(train, length=a.iters * a.batch), batch_size=a.batch, num_workers=a.workers,
                    persistent_workers=a.workers > 0, prefetch_factor=4 if a.workers else None)
    t0 = time.time()
    run = 0.0
    for it, (x, y, m) in enumerate(dl, 1):
        x, y, m = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True), m.to(dev, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p = model(x)
        l1 = (p.float() - y).abs().mean(dim=(2, 3))
        loss = (l1 * m).sum() / m.sum().clamp(min=1)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        run = 0.98 * run + 0.02 * loss.item() if it > 1 else loss.item()
        if it % 200 == 0 or it == a.iters:
            print(f"it {it}/{a.iters} loss {run:.4f} {time.time() - t0:.0f}s", flush=True)
        if it >= a.iters:
            break
    torch.save({"model": model.state_dict(), "px_target": PX_TARGET, "args": vars(a)}, out / "unet_l1.pt")

    fields = ["split", "fov", "study", "site", "mod", "z", "nz", "org", "ssim", "pcc", "ssim_raw",
              "ssim_tl_trivial", "pcc_tl_trivial"]
    with open(out / "metrics_rows.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        evaluate(model, val, dev, "val", w, a.max_eval_fov or None)
        evaluate(model, ood, dev, "ood", w, a.max_eval_fov or None)
    print("evaluation written", time.time() - t0, flush=True)


if __name__ == "__main__":
    main()

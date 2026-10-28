"""Model and preprocessing shared by training (E3) and the container (E8).

Changing anything here changes both: the container must use EXACTLY the training preprocessing.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lmc_common import MODS, pnorm

PX_TARGET = 0.2  # working µm/pixel (fallback if metadata is missing: 0.1 µm)


def rescale(img, px, out_px=PX_TARGET):
    f = (px or 0.1) / out_px
    t = torch.from_numpy(img)[None, None]
    h, w = max(8, round(img.shape[0] * f)), max(8, round(img.shape[1] * f))
    return F.interpolate(t, size=(h, w), mode="area" if f < 1 else "bilinear")[0, 0].numpy()


def block(ci, co):
    return nn.Sequential(nn.Conv2d(ci, co, 3, padding=1), nn.GroupNorm(8, co), nn.SiLU(),
                         nn.Conv2d(co, co, 3, padding=1), nn.GroupNorm(8, co), nn.SiLU())


class UNet(nn.Module):
    def __init__(self, cin=4, cout=4, base=32, depth=4):
        super().__init__()
        ch = [base * 2 ** i for i in range(depth + 1)]
        self.down = nn.ModuleList([block(cin, ch[0])] + [block(ch[i], ch[i + 1]) for i in range(depth)])
        self.up = nn.ModuleList([nn.ConvTranspose2d(ch[i + 1], ch[i], 2, stride=2) for i in reversed(range(depth))])
        self.dec = nn.ModuleList([block(ch[i] * 2, ch[i]) for i in reversed(range(depth))])
        self.head = nn.Conv2d(ch[0], cout, 1)
        self.depth = depth

    def forward(self, x):
        skips = []
        for i, d in enumerate(self.down):
            x = d(x if i == 0 else F.max_pool2d(x, 2))
            skips.append(x)
        x = skips.pop()
        for up, dec in zip(self.up, self.dec):
            x = dec(torch.cat([up(x), skips.pop()], 1))
        return self.head(x)


@torch.no_grad()
def predict_full(model, tl, px, mod, dev):
    """Full image: rescale to PX_TARGET, pad to a multiple of 16, predict and return to the original size."""
    x = pnorm(rescale(tl, px))
    h, w = x.shape
    H, W = math.ceil(h / 16) * 16, math.ceil(w / 16) * 16
    x = np.pad(x, ((0, H - h), (0, W - w)), mode="reflect")
    onehot = np.zeros((3, H, W), np.float32)
    onehot[MODS.index(mod)] = 1
    inp = torch.from_numpy(np.concatenate([x[None], onehot])[None].astype(np.float32)).to(dev)
    cuda = str(dev).startswith("cuda")
    amp = torch.bfloat16 if cuda and torch.cuda.is_bf16_supported() else torch.float16  # T4: fp16
    with torch.autocast("cuda" if cuda else "cpu", dtype=amp, enabled=cuda):
        y = model(inp).float()[:, :, :h, :w]
    y = F.interpolate(y, size=tl.shape, mode="bilinear")
    return y[0].cpu().numpy()

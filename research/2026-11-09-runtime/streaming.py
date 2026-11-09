"""Frozen LMC2 inference with one final GPU-to-host transfer per stack.

Original ensemble, preprocessing and interpolation retained. No model fitting.
"""
import math
import numpy as np
import torch
import torch.nn.functional as F
import inference as baseline


@torch.inference_mode()
def predict_normalized_tensor(models, x, original_shape, mod, device=None):
    device = baseline.DEVICE if device is None else torch.device(device)
    h, w = x.shape
    H, W = math.ceil(h / 16) * 16, math.ceil(w / 16) * 16
    x = np.pad(x, ((0, H-h), (0, W-w)), mode='reflect')
    onehot = np.zeros((3, H, W), np.float32)
    onehot[baseline.MODS.index(mod)] = 1
    inp = torch.from_numpy(np.concatenate((x[None], onehot))[None].astype(np.float32)).to(device)
    cuda = device.type == 'cuda'
    total = None
    with torch.autocast('cuda' if cuda else 'cpu', dtype=torch.float16, enabled=cuda):
        for model in models:
            pred = model(inp).float()[:, :, :h, :w]
            total = pred if total is None else total + pred
    return F.interpolate(total / len(models), size=original_shape, mode='bilinear')[0]


@torch.inference_mode()
def predict_stack(models, stack, px, mod, normalization='common', device=None):
    if stack.ndim != 3 or stack.shape[0] < 1:
        raise ValueError('Expected nonempty ZYX')
    low = high = None
    if len(stack) > 1 and normalization == 'common':
        sample = np.concatenate([baseline.rescale(plane, px)[::4, ::4].ravel() for plane in stack])
        low, high = np.percentile(sample, [2, 99.8])
    out = None
    for plane in stack:
        x = baseline.rescale(plane, px)
        if low is None:
            x = baseline.pnorm(x)
        else:
            x = (x-low) / max(high-low, 1e-6)
        pred = predict_normalized_tensor(models, x, plane.shape, mod, device)
        out = pred if out is None else torch.maximum(out, pred)
    out = out.cpu().numpy()
    if not np.isfinite(out).all():
        raise ValueError('Nonfinite prediction')
    return out

"""Métricas LMC2 (referencia provisional hasta que se publique la oficial de la ed.2).

Principal ("mmv"): como el evaluador público del 2.º puesto de la ed.1 (MMV-Lab/LightMyCells,
evaluate.py): GT y predicción normalizadas cada una por percentiles 2-99,8 sin recorte;
SSIM = skimage.structural_similarity con ventana uniforme 7, data_range = rango del GT normalizado;
PCC = Pearson sobre todos los píxeles.
Variante ("raw"): SSIM sobre la predicción reescalada al rango del GT crudo (sensibilidad).

ssim_torch replica skimage (ventana uniforme 7x7, K1=0.01, K2=0.03, covarianza muestral, recorte
de borde de 3 px, media) en GPU; check_against_skimage() lo verifica.
"""
import numpy as np
import torch
import torch.nn.functional as F

from lmc_common import pnorm

WIN = 7


def ssim_torch(x, y, data_range):
    """x, y: tensores 2D float64 en el mismo dispositivo."""
    x = x[None, None].double()
    y = y[None, None].double()
    np_ = WIN * WIN
    cov_norm = np_ / (np_ - 1)
    k = torch.ones(1, 1, WIN, WIN, dtype=torch.float64, device=x.device) / np_
    ux, uy = F.conv2d(x, k), F.conv2d(y, k)
    uxx, uyy, uxy = F.conv2d(x * x, k), F.conv2d(y * y, k), F.conv2d(x * y, k)
    vx = cov_norm * (uxx - ux * ux)
    vy = cov_norm * (uyy - uy * uy)
    vxy = cov_norm * (uxy - ux * uy)
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    s = ((2 * ux * uy + c1) * (2 * vxy + c2)) / ((ux * ux + uy * uy + c1) * (vx + vy + c2))
    # conv 'valid' ya elimina 3 px por borde = crop(S, pad) de skimage
    return float(s.mean())


def pcc_torch(x, y):
    x = x.double().flatten()
    y = y.double().flatten()
    x = x - x.mean()
    y = y - y.mean()
    den = torch.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / den) if den > 0 else 0.0


def score(pred, gt, device="cuda"):
    """pred, gt: arrays 2D numpy del mismo tamaño. Devuelve dict de métricas."""
    g = torch.from_numpy(pnorm(gt, sub=1).astype(np.float64)).to(device)
    p = torch.from_numpy(pnorm(pred, sub=1).astype(np.float64)).to(device)
    out = {"ssim": ssim_torch(g, p, float(g.max() - g.min())), "pcc": pcc_torch(g, p)}
    gr = torch.from_numpy(gt.astype(np.float64)).to(device)
    pr = p * (gr.max() - gr.min()) + gr.min()
    out["ssim_raw"] = ssim_torch(gr, pr, float(gr.max() - gr.min()))
    return out


def check_against_skimage(n=3, size=300, device="cuda"):
    from scipy.stats import pearsonr
    from skimage.metrics import structural_similarity
    rng = np.random.default_rng(2026)
    worst = 0.0
    for _ in range(n):
        gt = rng.gamma(2.0, 100.0, (size, size + 17)).astype(np.float32)
        pred = (gt + rng.normal(0, 60, gt.shape)).astype(np.float32)
        g, p = pnorm(gt, sub=1), pnorm(pred, sub=1)
        ref_s = structural_similarity(g, p, data_range=g.max() - g.min())
        ref_p = pearsonr(g.flatten(), p.flatten()).statistic
        m = score(pred, gt, device)
        worst = max(worst, abs(m["ssim"] - ref_s), abs(m["pcc"] - ref_p))
    assert worst < 1e-6, f"SSIM/PCC torch difiere de skimage/scipy: {worst}"
    return worst

"""Render the animated README figures from real out-of-fold predictions and archived evidence.

Inputs:
  --showcase DIR   .npz files written by tools/make_showcase_arrays.py (public ed.1 data, CC-BY 4.0)
  --evidence DIR   the repository evidence/ folder
Outputs: assets/*.gif

Every image panel shows data exactly as predicted or acquired; only a per-image display
stretch (1st-99.7th percentile) and organelle pseudo-colours are applied for viewing.
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
from PIL import Image  # noqa: E402

ORGS = ["Nucleus", "Mitochondria", "Tubulin", "Actin"]
COLOR = {"Nucleus": (0.33, 0.62, 1.0), "Mitochondria": (1.0, 0.36, 0.33),
         "Tubulin": (0.35, 0.95, 0.50), "Actin": (1.0, 0.78, 0.22)}
BG, FG, MUTED, ACCENT = "#0d1117", "#e6edf3", "#8b949e", "#58a6ff"
plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": FG, "axes.labelcolor": FG,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": "#30363d",
                     "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG})


def stretch(a, lo=1.0, hi=99.7):
    p1, p2 = np.percentile(a, [lo, hi])
    return np.clip((a - p1) / max(p2 - p1, 1e-6), 0, 1)


def tint(a, org):
    return stretch(a)[..., None] * np.array(COLOR[org])[None, None]


def composite(chans, orgs):
    # Maximum blend keeps each organelle's own colour instead of saturating to white.
    return np.max(np.stack([tint(c, o) for c, o in zip(chans, orgs)]), 0)


def gray(a):
    return np.repeat(stretch(a)[..., None], 3, -1)


def fig_to_img(fig):
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return img


def save_gif(frames, path, ms):
    ims = [Image.fromarray(f) for f in frames]
    # One palette built from a montage of sampled frames, so no frame loses its colours.
    pick = ims[::max(1, len(ims) // 8)][:8]
    mont = Image.new("RGB", (pick[0].width, pick[0].height * len(pick)))
    for i, im in enumerate(pick):
        mont.paste(im, (0, i * im.height))
    pal = mont.quantize(colors=255, method=Image.Quantize.MEDIANCUT)
    q = [im.quantize(palette=pal, dither=Image.Dither.NONE) for im in ims]
    durs = ms if isinstance(ms, list) else [ms] * len(q)
    q[0].save(path, save_all=True, append_images=q[1:], duration=durs, loop=0, optimize=True)
    print(path, len(q), "frames", round(Path(path).stat().st_size / 1e6, 2), "MB")


def load(showcase, key):
    z = np.load(Path(showcase) / f"{key}.npz")
    meta = json.loads(str(z["meta"]))
    return z["tl"], z["pred"], list(z["gt_names"]), z["gt"], meta


def ds(a, f=2):
    return a[..., ::f, ::f]


def panels(imgs, titles, suptitle, foot, size=(12.8, 5.0)):
    fig, axs = plt.subplots(1, len(imgs), figsize=size, dpi=85)
    for ax, im, t in zip(axs, imgs, titles):
        ax.imshow(im, interpolation="lanczos")
        ax.set_title(t, fontsize=15, color=FG, pad=8)
        ax.axis("off")
    fig.suptitle(suptitle, fontsize=17, color=FG, fontweight="bold", y=0.97)
    fig.text(0.5, 0.03, foot, ha="center", fontsize=11, color=MUTED)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.83, bottom=0.08, wspace=0.03)
    return fig_to_img(fig)


def hero(showcase, out):
    tl, pred, names, gt, m = load(showcase, "S24_390")
    zc = tl.shape[0] // 2
    tlz, pz = ds(tl[zc]), ds(pred[zc])
    g = {n: ds(gt[i]) for i, n in enumerate(names)}
    p = {o: pz[ORGS.index(o)] for o in names}
    foot = (f"Out-of-fold example · public LMC ed.1 Study {m['study']} ({m['modality']}, plane {zc + 1}/{m['nz']}) · "
            f"model fold {m['fold']} never saw this specimen group")
    frames, durs = [], []
    stages = [(o, [o]) for o in names]
    prev = None
    for label, orgs in stages:
        cur = (composite([p[o] for o in orgs], orgs), composite([g[o] for o in orgs], orgs))
        if prev is not None:
            for a in np.linspace(0, 1, 5)[1:-1]:
                mix = [prev[i] * (1 - a) + cur[i] * a for i in range(2)]
                frames.append(panels([gray(tlz), *mix], ["Transmitted light (input)", "Prediction", "Ground truth"],
                                     f"Transmitted light → fluorescence · {label}", foot))
                durs.append(90)
        frames.append(panels([gray(tlz), *cur], ["Transmitted light (input)", "Prediction", "Ground truth"],
                             f"Transmitted light → fluorescence · {label}", foot))
        durs.append(1700)
        prev = cur
    save_gif(frames, out / "hero_tl_to_fluorescence.gif", durs)


def wipe(showcase, out, key="S20_308"):
    tl, pred, names, gt, m = load(showcase, key)
    zc = tl.shape[0] // 2
    tlz = gray(ds(tl[zc]))
    pc = composite([ds(pred[zc][ORGS.index(o)]) for o in names], names)
    w = tlz.shape[1]
    frames, durs = [], []
    pos = list(np.linspace(0, 1, 22)) + [1.0] * 6 + list(np.linspace(1, 0, 22)) + [0.0] * 4
    for t in pos:
        x = int(round(t * w))
        img = tlz.copy()
        img[:, :x] = pc[:, :x]
        if 0 < x < w:
            img[:, max(0, x - 2):x + 2] = 1.0
        fig, ax = plt.subplots(figsize=(7.2, 6.6), dpi=85)
        ax.imshow(img, interpolation="lanczos")
        ax.axis("off")
        ax.set_title("In-silico labelling from one transmitted-light image", fontsize=14, pad=10)
        for i, o in enumerate(names):
            fig.text(0.5 + (i - (len(names) - 1) / 2) * 0.3, 0.025, f"● {o}", ha="center", fontsize=12, color=COLOR[o])
        fig.text(0.02, 0.07, f"Study {m['study']} · {m['modality']} · out-of-fold (fold {m['fold']})", fontsize=10,
                 color=MUTED)
        fig.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.1)
        frames.append(fig_to_img(fig))
        durs.append(70)
    durs[27] = 900
    save_gif(frames, out / "wipe_in_silico_labelling.gif", durs)


def zstack(showcase, out, key="S25_1031"):
    tl, pred, names, gt, m = load(showcase, key)
    nz = tl.shape[0]
    orgs = [o for o in ORGS if o in names] or ["Nucleus"]
    comp = [np.stack([pred[z][ORGS.index(o)] for o in orgs]) for z in range(nz)]
    # One display scale for the whole stack, so the running maximum is shown faithfully.
    lo = {o: np.percentile(pred[:, ORGS.index(o)], 1) for o in orgs}
    hi = {o: np.percentile(pred[:, ORGS.index(o)], 99.7) for o in orgs}

    def col(stack):
        return np.max(np.stack([np.clip((stack[i] - lo[o]) / (hi[o] - lo[o]), 0, 1)[..., None] * np.array(COLOR[o])
                                for i, o in enumerate(orgs)]), 0)

    tlo, thi = np.percentile(tl, [1, 99.7])
    frames, durs, run = [], [], None
    for z in range(nz):
        run = comp[z] if run is None else np.maximum(run, comp[z])
        tlz = np.repeat(np.clip((tl[z] - tlo) / (thi - tlo), 0, 1)[..., None], 3, -1)
        fig = plt.figure(figsize=(12.8, 5.6), dpi=85)
        for i, (im, t) in enumerate([(tlz, f"Transmitted-light plane z = {z + 1}/{nz}"),
                                     (col(comp[z]), "Per-plane prediction"),
                                     (col(run), "Running maximum (MIP)")]):
            ax = fig.add_axes([0.01 + i * 0.33, 0.14, 0.31, 0.68])
            ax.imshow(im, interpolation="lanczos")
            ax.set_title(t, fontsize=14)
            ax.axis("off")
        bar = fig.add_axes([0.2, 0.06, 0.6, 0.03])
        bar.barh([0], [z + 1], color=ACCENT, height=1)
        bar.set_xlim(0, nz)
        bar.set_yticks([])
        bar.set_xticks([1, nz // 2, nz])
        fig.suptitle("MIP task: z-stack in, maximum-intensity projection out", fontsize=17, fontweight="bold", y=0.97)
        frames.append(fig_to_img(fig))
        durs.append(220)
    durs[-1] = 2200
    save_gif(frames, out / "zstack_to_mip.gif", durs)


def pipeline(out):
    steps = [("OME-TIFF input", "BF · PC · DIC\n2D plane or z-stack"),
             ("Rescale", "0.2 µm / pixel\nfrom OME metadata"),
             ("Normalise", "percentile p2–p99.8\n(common per stack)"),
             ("Modality code", "+3 one-hot channels\nBF / PC / DIC"),
             ("5 × U-Net", "one per grouped fold\n4 outputs each"),
             ("Ensemble", "mean of 5 models\nfp16 on T4"),
             ("Project", "Best Focus: plane\nMIP: max over z"),
             ("OME-TIFF out", "4 organelles\noriginal size + spacing")]
    frames, durs = [], []
    n = len(steps)
    for k in range(n + 6):
        fig = plt.figure(figsize=(16, 4.4), dpi=85)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, n * 1.6)
        ax.set_ylim(0, 4.2)
        ax.axis("off")
        ax.text(n * 0.8, 3.75, "Inference pipeline (identical code in training, container and validation)",
                ha="center", fontsize=16, fontweight="bold", color=FG)
        for i, (t, s) in enumerate(steps):
            on = i <= k
            x = 0.12 + i * 1.6
            ax.add_patch(FancyBboxPatch((x, 1.0), 1.36, 2.05, boxstyle="round,pad=0.02,rounding_size=0.12",
                                        fc="#1f6feb" if i == k else ("#161b22" if not on else "#0f2a4a"),
                                        ec=ACCENT if on else "#30363d", lw=2))
            ax.text(x + 0.68, 2.62, t, ha="center", va="center", fontsize=11.5, fontweight="bold",
                    color=FG if on else MUTED)
            ax.text(x + 0.68, 1.65, s, ha="center", va="center", fontsize=9.5, color=FG if on else MUTED)
            if i < n - 1:
                ax.annotate("", xy=(x + 1.6, 2.02), xytext=(x + 1.38, 2.02),
                            arrowprops=dict(arrowstyle="-|>", color=ACCENT if i < k else "#30363d", lw=2))
        orgs_on = k >= n - 1
        for j, o in enumerate(ORGS):
            ax.text(n * 0.8 + (j - 1.5) * 2.2, 0.45, f"● {o}", ha="center", fontsize=13,
                    color=COLOR[o] if orgs_on else "#30363d")
        frames.append(fig_to_img(fig))
        durs.append(550)
    durs[-1] = 2500
    save_gif(frames, out / "pipeline.gif", durs)


def fold_stats(evidence):
    d = json.loads((Path(evidence) / "cv_audit_quality.json").read_text())
    res = {}
    for o in ORGS:
        ks = [k for k in d["per_fold"] if k.startswith(f"val|{o}|")]
        get = lambda m: np.array([d["per_fold"][k]["fov_mean"][m] for k in ks], float)  # noqa: E731
        res[o] = {"n_fov": d["val"][o]["n_fov"], "folds": len(ks),
                  **{m: (float(d["val"][o]["fov_mean"][m]), float(np.std(get(m), ddof=1)) if len(ks) > 1 else 0.0)
                     for m in ("ssim", "pcc", "ssim_tl_trivial", "pcc_tl_trivial")}}
    return res


def results(evidence, out):
    st = fold_stats(evidence)
    s = json.loads((Path(evidence) / "ed2_20261027/supervised_results_summary.json").read_text())
    mip = {(r["method"], r["organelle"]): r for r in s["summary"] if r["modality"] == "all" and r["metric"] == "ssim"}
    frames, durs = [], []
    ease = [1 - (1 - t) ** 3 for t in np.linspace(0, 1, 16)]
    x = np.arange(4)
    for e in ease + [1.0] * 4:
        fig, axs = plt.subplots(1, 3, figsize=(14.4, 5.4), dpi=85)
        for ax, (m, base, title) in zip(axs[:2], [("ssim", "ssim_tl_trivial", "SSIM · grouped 5-fold CV (ed.1)"),
                                                  ("pcc", "pcc_tl_trivial", "PCC · grouped 5-fold CV (ed.1)")]):
            v = np.array([st[o][m][0] for o in ORGS])
            sd = np.array([st[o][m][1] for o in ORGS])
            b = np.array([st[o][base][0] for o in ORGS])
            ax.bar(x - 0.2, b * e, 0.38, color="#30363d", label="TL image as prediction (control)")
            ax.bar(x + 0.2, v * e, 0.38, color=[COLOR[o] for o in ORGS], label="Five-fold U-Net")
            if e == 1.0:
                ax.errorbar(x + 0.2, v, yerr=sd, fmt="none", ecolor=FG, capsize=4, lw=1.3)
                for i in range(4):
                    ax.text(x[i] + 0.2, v[i] + sd[i] + 0.02, f"{v[i]:.2f}", ha="center", fontsize=10.5)
            ax.set_xticks(x, [f"{o}\n(n={st[o]['n_fov']})" for o in ORGS], fontsize=10)
            ax.set_ylim(0, 0.98)
            ax.set_title(title, fontsize=13.5)
            ax.spines[["top", "right"]].set_visible(False)
        ax = axs[2]
        for j, (meth, lab, c) in enumerate([("tl_mean", "TL mean projection", "#30363d"),
                                            ("tl_max", "TL max projection", "#484f58"),
                                            ("ed1_ensemble", "Five-fold U-Net, max over z", None)]):
            v = np.array([mip[(meth, o)]["equal_fov"] for o in ORGS])
            ci = np.array([mip[(meth, o)]["ci95_fov_cluster"] for o in ORGS]).T
            ax.bar(x + (j - 1) * 0.27, v * e, 0.26, color=c if c else [COLOR[o] for o in ORGS], label=lab)
            if e == 1.0 and meth == "ed1_ensemble":
                ax.errorbar(x + 0.27, v, yerr=[v - ci[0], ci[1] - v], fmt="none", ecolor=FG, capsize=4, lw=1.3)
                for i in range(4):
                    ax.text(x[i] + 0.27, ci[1][i] + 0.02, f"{v[i]:.2f}", ha="center", fontsize=10.5)
        ax.set_xticks(x, [f"{o}\n(n={mip[('ed1_ensemble', o)]['n_fov']})" for o in ORGS], fontsize=10)
        ax.set_ylim(0, 1.18)
        ax.set_title("SSIM · MIP task · frozen ed.2 hold-out", fontsize=13.5)
        ax.spines[["top", "right"]].set_visible(False)
        axs[0].legend(loc="upper left", fontsize=9.5, frameon=False)
        ax.legend(loc="upper left", fontsize=9.5, frameon=False)
        fig.suptitle("Local validation (provisional ed.1 metric convention; equal-FOV means)", fontsize=15,
                     fontweight="bold", y=0.975)
        fig.text(0.5, 0.01, "Whiskers: SD across validation folds (CV) · 95% FOV-cluster bootstrap CI (hold-out). "
                 "Not official challenge scores.", ha="center", fontsize=10, color=MUTED)
        fig.subplots_adjust(left=0.04, right=0.99, top=0.83, bottom=0.17, wspace=0.18)
        frames.append(fig_to_img(fig))
        durs.append(60)
    durs[-1] = 4000
    save_gif(frames, out / "validation_results.gif", durs)


def grouped_cv(evidence, out):
    d = json.loads((Path(evidence) / "cv_audit_quality.json").read_text())
    sf = {int(k): v[0] for k, v in d["study_folds"].items()}
    cols = {"0": "#1f6feb", "1": "#a371f7", "2": "#3fb950", "3": "#d29922", "4": "#f85149", "ood": "#6e7681"}
    frames, durs = [], []
    for k in ["all", "0", "1", "2", "3", "4", "ood"]:
        fig = plt.figure(figsize=(12.8, 4.4), dpi=85)
        ax = fig.add_axes([0.02, 0.05, 0.96, 0.75])
        ax.set_xlim(0, 15)
        ax.set_ylim(0, 2)
        ax.axis("off")
        for st in range(1, 31):
            f = sf[st]
            r, c = divmod(st - 1, 15)
            hl = k == "all" or f == k
            ax.add_patch(FancyBboxPatch((c + 0.06, 1.06 - r), 0.88, 0.86, boxstyle="round,pad=0.01,rounding_size=0.1",
                                        fc=cols[f] if hl else "#161b22", ec=cols[f], lw=1.6, alpha=1 if hl else 0.6))
            ax.text(c + 0.5, 1.49 - r, f"S{st}", ha="center", va="center", fontsize=11, fontweight="bold",
                    color=FG if hl else MUTED)
        if k == "all":
            msg = "30 public studies → 20 specimen groups → 5 grouped folds + fixed out-of-distribution bank"
        elif k == "ood":
            msg = "Out-of-distribution bank (studies 1, 16, 17, 28): never used for training or model selection"
        else:
            msg = f"Fold {k} held out: its model is trained on the other four folds only"
        fig.suptitle("Leakage-safe validation: whole specimen groups move together", fontsize=16,
                     fontweight="bold", y=0.97)
        fig.text(0.5, 0.83, msg, ha="center", fontsize=12.5, color=FG)
        frames.append(fig_to_img(fig))
        durs.append(1500)
    save_gif(frames, out / "grouped_cv.gif", durs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--showcase", required=True)
    ap.add_argument("--evidence", default=str(Path(__file__).resolve().parents[1] / "evidence"))
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "assets"))
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = {"hero": lambda: hero(a.showcase, out), "wipe": lambda: wipe(a.showcase, out),
            "zstack": lambda: zstack(a.showcase, out), "pipeline": lambda: pipeline(out),
            "results": lambda: results(a.evidence, out), "cv": lambda: grouped_cv(a.evidence, out)}
    for name, fn in jobs.items():
        if not a.only or name in a.only:
            fn()


if __name__ == "__main__":
    main()

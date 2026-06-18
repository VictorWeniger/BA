"""Figur: H0/H1-Wasserstein-Heatmap ResNet-50-Schichten vs. Glasser-Areale.

Liest tda_resnet50_glasser_all_subjects.csv und zeichnet fuer eine Auswahl
ventral-stromiger Areale (frueh -> hoch) die mittlere Wasserstein-Distanz je
Schicht. Schwarzer Rahmen = beste (minimale) Schicht pro Areal.
"""

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LAYERS = ["conv1", "layer1", "layer2", "layer3", "layer4", "fc"]
AREAS = ["V1", "V2", "V3", "V4", "V8", "VMV1", "VVC", "FFC", "PIT", "LO2"]

RES = PROJECT_ROOT / "outputs_resnet50_v2_surface" / "results"
FIG = PROJECT_ROOT.parent / "figures_resnet50_v2"
FIG.mkdir(parents=True, exist_ok=True)


def load_means():
    acc = defaultdict(list)  # (area, layer, hdim) -> [dists]
    with (RES / "tda_resnet50_glasser_all_subjects.csv").open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            acc[(r["area"], r["layer"], int(r["homology_dim"]))].append(
                float(r["wasserstein_dist"]))
    return acc


def heatmap(acc, hdim, fname, title):
    mat = np.full((len(AREAS), len(LAYERS)), np.nan)
    for i, a in enumerate(AREAS):
        for j, l in enumerate(LAYERS):
            vals = acc.get((a, l, hdim), [])
            if vals:
                mat[i, j] = np.mean(vals)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    im = ax.imshow(mat, aspect="auto", cmap="YlOrRd_r",
                   vmin=np.nanmin(mat), vmax=np.nanmax(mat))
    plt.colorbar(im, ax=ax, label=fr"$\varnothing$ Wasserstein-Distanz $W_1$ ($H_{hdim}$)")
    ax.set_xticks(range(len(LAYERS)))
    ax.set_xticklabels(LAYERS, rotation=30)
    ax.set_yticks(range(len(AREAS)))
    ax.set_yticklabels(AREAS)
    ax.set_xlabel("ResNet-50-Schicht")
    ax.set_ylabel("Glasser-Areal (früh oben → hoch unten)")
    ax.set_title(title)
    for i in range(len(AREAS)):
        if np.all(np.isnan(mat[i])):
            continue
        bj = int(np.nanargmin(mat[i]))
        ax.add_patch(plt.Rectangle((bj - 0.5, i - 0.5), 1, 1, fill=False,
                                   edgecolor="black", lw=2))
        for j in range(len(LAYERS)):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i,j]:.1f}", ha="center", va="center",
                        fontsize=7)
    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{fname}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("gespeichert:", FIG / f"{fname}.pdf")


def main():
    acc = load_means()
    heatmap(acc, 0, "v2_tda_h0_heatmap",
            r"Topologische Distanz $H_0$ — ResNet-50 vs. Glasser-Areale (Ø S1–S3)")
    heatmap(acc, 1, "v2_tda_h1_heatmap",
            r"Topologische Distanz $H_1$ — ResNet-50 vs. Glasser-Areale (Ø S1–S3)")
    if any(k[2] == 2 for k in acc):
        heatmap(acc, 2, "v2_tda_h2_heatmap",
                r"Topologische Distanz $H_2$ — ResNet-50 vs. Glasser-Areale (Ø S1–S3)")


if __name__ == "__main__":
    main()

"""TDA-Vergleich: ResNet-50-Schichten vs. fMRI-ROIs.

Modellagnostische Variante von ``compare_tda_resnet18_to_fmri_rois.py``.
Liest die ResNet-50-Cosine-RDMs aus dem ResNet-50-Ausgabeordner und die
fMRI-RDMs aus ``outputs/human`` (modellunabhängig) und berechnet die
Wasserstein-Distanzen der Persistenzdiagramme (H0, H1, optional H2).

Beispiel:
  python scripts/compute_tda_resnet50_fmri.py \
      --geometry-dir outputs_resnet50/geometry \
      --results-dir  outputs_resnet50/results \
      --figures-dir  outputs_resnet50/figures
"""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from persim import wasserstein
from ripser import ripser

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LAYERS = ["conv1", "layer1", "layer2", "layer3", "layer4", "fc"]
ROIS = ["V1", "hV4", "LOC", "IT"]
ROI_COLORS = {"V1": "#e41a1c", "hV4": "#ff7f00", "LOC": "#4daf4a", "IT": "#377eb8"}


def normalize_rdm(rdm):
    rdm = rdm.copy()
    np.fill_diagonal(rdm, 0.0)
    vmax = rdm.max()
    if vmax > 0:
        rdm = rdm / vmax
    return rdm


def diagram_from_rdm(rdm, maxdim):
    return ripser(normalize_rdm(rdm), distance_matrix=True, maxdim=maxdim)["dgms"]


def wasserstein_safe(a, b):
    if len(a) == 0 and len(b) == 0:
        return 0.0
    return float(wasserstein(a, b))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="resnet50")
    parser.add_argument("--subjects", nargs="+", default=["S1", "S2", "S3"])
    parser.add_argument("--rois", nargs="+", default=ROIS)
    parser.add_argument("--maxdim", type=int, default=2)
    parser.add_argument("--geometry-dir", default="outputs_resnet50/geometry")
    parser.add_argument("--human-dir", default="outputs/human")
    parser.add_argument("--results-dir", default="outputs_resnet50/results")
    parser.add_argument("--figures-dir", default="outputs_resnet50/figures")
    args = parser.parse_args()

    geom_dir = PROJECT_ROOT / args.geometry_dir
    human_dir = PROJECT_ROOT / args.human_dir
    results_dir = PROJECT_ROOT / args.results_dir
    figures_dir = PROJECT_ROOT / args.figures_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    model = args.model
    output_csv = results_dir / f"tda_{model}_all_subjects_roi_summary.csv"
    all_rows = []

    for subject in args.subjects:
        print(f"\n=== {subject} ===")
        ann_diagrams = {}
        for layer in LAYERS:
            rdm_file = geom_dir / f"ann_{model}_{subject}_matched_{layer}_cosine.npy"
            if not rdm_file.exists():
                print(f"  FEHLER: {rdm_file} fehlt")
                continue
            ann_diagrams[layer] = diagram_from_rdm(np.load(rdm_file), args.maxdim)
            print(f"  ANN {layer}: Diagramm OK")

        for roi in args.rois:
            rdm_file = human_dir / f"processed_fmri_{subject}_{roi}_distance.npy"
            if not rdm_file.exists():
                print(f"  FEHLER: {rdm_file} fehlt, ROI {roi} übersprungen")
                continue
            fmri_diagrams = diagram_from_rdm(np.load(rdm_file), args.maxdim)
            print(f"  fMRI {roi}: Diagramm OK")

            for layer in LAYERS:
                if layer not in ann_diagrams:
                    continue
                for hdim in range(args.maxdim + 1):
                    dist = wasserstein_safe(ann_diagrams[layer][hdim], fmri_diagrams[hdim])
                    all_rows.append({
                        "subject": subject, "roi": roi, "layer": layer,
                        "homology_dim": hdim, "wasserstein_dist": round(dist, 6),
                    })

    fieldnames = ["subject", "roi", "layer", "homology_dim", "wasserstein_dist"]
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nGespeichert: {output_csv}  ({len(all_rows)} Zeilen)")

    # Heatmaps (Mittel über Subjects) für H0/H1/H2
    for hdim in range(args.maxdim + 1):
        rows_h = [r for r in all_rows if r["homology_dim"] == hdim]
        if not rows_h:
            continue
        mat = np.full((len(args.rois), len(LAYERS)), np.nan)
        for i, roi in enumerate(args.rois):
            for j, layer in enumerate(LAYERS):
                vals = [r["wasserstein_dist"] for r in rows_h
                        if r["roi"] == roi and r["layer"] == layer]
                if vals:
                    mat[i, j] = np.mean(vals)
        fig, ax = plt.subplots(figsize=(7, 3.5))
        im = ax.imshow(mat, aspect="auto", cmap="YlOrRd_r",
                       vmin=np.nanmin(mat), vmax=np.nanmax(mat))
        plt.colorbar(im, ax=ax, label="Ø Wasserstein-Distanz")
        ax.set_xticks(range(len(LAYERS)))
        ax.set_xticklabels(LAYERS, rotation=30)
        ax.set_yticks(range(len(args.rois)))
        ax.set_yticklabels(args.rois)
        ax.set_title(f"Topologische Ähnlichkeit H{hdim} — {model} Ø {args.subjects}")
        for i in range(len(args.rois)):
            for j in range(len(LAYERS)):
                if not np.isnan(mat[i, j]):
                    ax.text(j, i, f"{mat[i,j]:.3f}", ha="center", va="center", fontsize=7.5)
        plt.tight_layout()
        out = figures_dir / f"tda_{model}_fmri_heatmap_H{hdim}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Heatmap: {out}")


if __name__ == "__main__":
    main()

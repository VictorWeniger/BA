"""
TDA-Vergleich: ResNet18-Schichten vs. fMRI-ROIs.

Methode: Vietoris-Rips-Persistenz auf normierten Cosine-Distanzmatrizen
(RDMs, direkt aus den RSA-Analyseergebnissen). Diese Variante ist
skaleninvariant und direkt vergleichbar mit den RSA-Ergebnissen.

Für jede Kombination (Subject × ROI × Layer):
  1. Die bereits berechneten Cosine-RDMs werden auf [0,1] normiert.
  2. Mit ripser (distance_matrix=True) werden Persistenzdiagramme berechnet.
  3. Wasserstein-Distanz zwischen ANN- und fMRI-Diagramm (H0, H1, H2).

Kleinere Wasserstein-Distanz = ähnlichere topologische Struktur.

Verwendung:
  venv_cortex/bin/python scripts/compare_tda_resnet18_to_fmri_rois.py \
      --subjects S1 S2 S3 \
      --rois V1 hV4 LOC IT \
      --output outputs/results/tda_resnet18_all_subjects_roi_summary.csv
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
GEOMETRY_DIR = PROJECT_ROOT / "outputs" / "geometry"
HUMAN_DIR    = PROJECT_ROOT / "outputs" / "human"
RESULTS_DIR  = PROJECT_ROOT / "outputs" / "results"
FIGURES_DIR  = PROJECT_ROOT / "outputs" / "figures"

LAYERS = ["conv1", "layer1", "layer2", "layer3", "layer4", "fc"]
ROIS   = ["V1", "hV4", "LOC", "IT"]


def normalize_rdm(rdm):
    """Normiert RDM auf [0,1] für skaleninvarianten TDA-Vergleich.

    Persistente Homologie hängt von Distanzschwellen ab. Ohne Normierung wären
    Wasserstein-Distanzen zwischen Diagrammen schwer interpretierbar, weil eine
    Matrix einfach größere Zahlenbereiche haben könnte als eine andere.
    """
    rdm = rdm.copy()
    np.fill_diagonal(rdm, 0.0)
    vmax = rdm.max()
    if vmax > 0:
        rdm = rdm / vmax
    return rdm


def compute_diagram_from_rdm(rdm, maxdim=2):
    """Vietoris-Rips-Persistenz direkt auf normierter Distanzmatrix.

    `distance_matrix=True` sagt ripser, dass die Eingabe keine Punktkoordinaten
    sind, sondern bereits paarweise Distanzen zwischen den 100 Stimuli.
    """
    rdm_norm = normalize_rdm(rdm)
    return ripser(rdm_norm, distance_matrix=True, maxdim=maxdim)["dgms"]


def wasserstein_safe(dgm_a, dgm_b):
    """Wasserstein-Distanz; gibt 0 zurück wenn beide Diagramme leer sind.

    Das verhindert Sonderfallfehler, falls in einer Homologiedimension keine
    Merkmale gefunden werden.
    """
    if len(dgm_a) == 0 and len(dgm_b) == 0:
        return 0.0
    return float(wasserstein(dgm_a, dgm_b))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects",  nargs="+", default=["S1", "S2", "S3"])
    parser.add_argument("--rois",      nargs="+", default=ROIS)
    parser.add_argument("--maxdim",    type=int,  default=2,
                        help="Maximale Homologie-Dimension (0=H0, 1=H1, 2=H2)")
    parser.add_argument("--output",    default=None)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    output_csv = Path(args.output) if args.output else \
        RESULTS_DIR / "tda_resnet18_all_subjects_roi_summary.csv"

    all_rows = []

    for subject in args.subjects:
        print(f"\n=== {subject} ===")

        # ANN-Diagramme einmalig pro Subject berechnen (aus Cosine-RDMs).
        # Diese Diagramme können für alle ROIs desselben Subjects wiederverwendet
        # werden, weil die ANN-Seite nicht von der ROI abhängt.
        ann_diagrams = {}
        for layer in LAYERS:
            rdm_file = GEOMETRY_DIR / f"ann_resnet18_{subject}_matched_{layer}_cosine.npy"
            if not rdm_file.exists():
                print(f"  FEHLER: {rdm_file} nicht gefunden")
                continue
            rdm_ann = np.load(rdm_file)
            ann_diagrams[layer] = compute_diagram_from_rdm(rdm_ann, args.maxdim)
            print(f"  ANN {layer}: RDM {rdm_ann.shape} → normiert → Diagramm OK")

        for roi in args.rois:
            rdm_file = HUMAN_DIR / f"processed_fmri_{subject}_{roi}_distance.npy"
            if not rdm_file.exists():
                print(f"  FEHLER: {rdm_file} nicht gefunden, ROI {roi} übersprungen")
                continue

            # fMRI-Diagramm ist ROI-spezifisch: V1/hV4/LOC/IT haben jeweils
            # eigene RDMs und damit eigene topologische Signaturen.
            rdm_fmri = np.load(rdm_file)
            fmri_diagrams = compute_diagram_from_rdm(rdm_fmri, args.maxdim)
            print(f"  fMRI {roi}: RDM {rdm_fmri.shape} → normiert → Diagramm OK")

            for layer in LAYERS:
                if layer not in ann_diagrams:
                    continue
                for hdim in range(args.maxdim + 1):
                    # H0: Komponenten/Cluster; H1: Schleifen/Zyklen.
                    # Kleinere Wasserstein-Distanz = ähnlichere Topologie.
                    dist = wasserstein_safe(
                        ann_diagrams[layer][hdim],
                        fmri_diagrams[hdim],
                    )
                    row = {
                        "subject":          subject,
                        "roi":              roi,
                        "layer":            layer,
                        "homology_dim":     hdim,
                        "wasserstein_dist": round(dist, 6),
                    }
                    all_rows.append(row)
                    print(f"    {layer} vs {roi} H{hdim}: W = {dist:.6f}")

    # CSV schreiben
    fieldnames = ["subject", "roi", "layer", "homology_dim", "wasserstein_dist"]
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nGespeichert: {output_csv}  ({len(all_rows)} Zeilen)")

    # -----------------------------------------------------------------------
    # Visualisierung
    # -----------------------------------------------------------------------
    _plot_results(all_rows, args.subjects, args.rois, args.maxdim)


def _plot_results(all_rows, subjects, rois, maxdim):
    """Wasserstein-Distanz pro ROI als Linienplot (Layer auf x-Achse)."""
    import matplotlib.cm as cm

    roi_colors = {"V1": "#e41a1c", "hV4": "#ff7f00", "LOC": "#4daf4a", "IT": "#377eb8"}

    for hdim in range(maxdim + 1):
        rows_h = [r for r in all_rows if r["homology_dim"] == hdim]
        if not rows_h:
            continue

        fig, axes = plt.subplots(1, len(subjects), figsize=(5 * len(subjects), 4),
                                  sharey=False)
        if len(subjects) == 1:
            axes = [axes]

        for ax, subject in zip(axes, subjects):
            for roi in rois:
                vals = [r["wasserstein_dist"] for r in rows_h
                        if r["subject"] == subject and r["roi"] == roi]
                if vals:
                    ax.plot(LAYERS, vals, marker="o",
                            color=roi_colors.get(roi, "gray"),
                            label=roi, linewidth=1.8)

            ax.set_title(subject, fontsize=12)
            ax.set_xlabel("ResNet18 Layer")
            ax.set_ylabel("Wasserstein-Distanz" if ax == axes[0] else "")
            ax.grid(alpha=0.25)
            ax.tick_params(axis="x", rotation=30)
            if ax == axes[-1]:
                ax.legend(fontsize=9, loc="upper right")

        fig.suptitle(f"Topologische Ähnlichkeit (H{hdim}) — ResNet18 vs. fMRI-ROIs",
                     fontsize=13)
        plt.tight_layout()
        out = FIGURES_DIR / f"tda_resnet18_fmri_wasserstein_H{hdim}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Abbildung: {out}")

    # Heatmap: gemittelt über Subjects, für H0, H1 und H2
    for hdim in range(maxdim + 1):
        rows_h = [r for r in all_rows if r["homology_dim"] == hdim]
        if not rows_h:
            continue

        # Mittelwert über Subjects
        mat = np.zeros((len(rois), len(LAYERS)))
        for i, roi in enumerate(rois):
            for j, layer in enumerate(LAYERS):
                vals = [r["wasserstein_dist"] for r in rows_h
                        if r["roi"] == roi and r["layer"] == layer]
                mat[i, j] = np.mean(vals) if vals else np.nan

        fig, ax = plt.subplots(figsize=(7, 3.5))
        im = ax.imshow(mat, aspect="auto", cmap="YlOrRd_r",
                       vmin=np.nanmin(mat), vmax=np.nanmax(mat))
        plt.colorbar(im, ax=ax, label="Ø Wasserstein-Distanz")
        ax.set_xticks(range(len(LAYERS)))
        ax.set_xticklabels(LAYERS, rotation=30)
        ax.set_yticks(range(len(rois)))
        ax.set_yticklabels(rois)
        ax.set_title(f"Topologische Ähnlichkeit H{hdim} — Ø über {subjects}")

        # Werte eintragen
        for i in range(len(rois)):
            for j in range(len(LAYERS)):
                if not np.isnan(mat[i, j]):
                    ax.text(j, i, f"{mat[i,j]:.3f}", ha="center", va="center",
                            fontsize=7.5, color="black")

        plt.tight_layout()
        out = FIGURES_DIR / f"tda_resnet18_fmri_heatmap_H{hdim}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Heatmap: {out}")


if __name__ == "__main__":
    main()

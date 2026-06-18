"""Geometrischer Vergleich: ANN-Schichten vs. fMRI-ROI-RDM.

Für jede Schicht werden berechnet:
  - Spearman-ρ: Rangkorrelation der paarweisen Distanzen (RSA)
  - p-Wert: Permutationstest (N=1000 Shuffles der fMRI-Paarwerte)
  - CKA: Centered Kernel Alignment zwischen ANN- und fMRI-Kernel

Output: CSV mit einer Zeile pro Schicht.

Hinweis: Einige Config-Felder heißen aus der alten Pipeline noch `human`.
In der aktuellen BA-Analyse zeigt `config["human"]["distance_matrix"]` auf
eine fMRI-RDM einer ROI, z.B. V1 oder LOC.
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.stats_utils import kernel_cka, permutation_test_spearman, spearman_corr, upper_triangle_values
from src.utils import ensure_dir, load_config, project_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dummy_resnet18.json")
    parser.add_argument("--n-permutations", type=int, default=1000)
    args = parser.parse_args()

    config = load_config(args.config)
    experiment_name = config["experiment_name"]
    metric = config["geometry"]["metric"]

    # --- fMRI-Referenzmatrix laden ---
    # Historischer Config-Name: "human". Aktuell ist das eine fMRI-RDM einer ROI.
    human_matrix = np.load(project_path(config["human"]["distance_matrix"]))
    # Oberes Dreieck: n*(n-1)/2 Werte — jedes Paar einmal
    human_values = upper_triangle_values(human_matrix)

    # Für CKA: Distanz → Ähnlichkeit umkehren, Diagonale auf 1 setzen.
    # So liegen ANN und fMRI als vergleichbare 100×100-Ähnlichkeitsmatrizen vor.
    K_human = 1.0 - human_matrix
    np.fill_diagonal(K_human, 1.0)

    geometry_dir = ensure_dir(config["geometry"]["output_dir"])
    results_dir = ensure_dir(config["results"]["output_dir"])

    # Fester Seed für Reproduzierbarkeit des Permutationstests
    rng = np.random.default_rng(0)
    rows = []

    for layer_name in config["model"]["layers"]:
        # --- ANN-RDM laden ---
        # Diese Matrix wurde in compute_geometry.py aus den ResNet-Features
        # derselben 100 Stimuli berechnet.
        ann_matrix = np.load(geometry_dir / f"{experiment_name}_{layer_name}_{metric}.npy")
        ann_values = upper_triangle_values(ann_matrix)

        # --- RSA: Spearman-Rangkorrelation ---
        # Fragt: Stimmt die Rangordnung der paarweisen Distanzen überein?
        rho = spearman_corr(ann_values, human_values)

        # --- Permutationstest ---
        # fMRI-Paarwerte werden 1000× zufällig umsortiert.
        # p-Wert = Anteil der Permutationen mit |ρ| ≥ beobachtetem |ρ|
        p_val = permutation_test_spearman(
            ann_values, human_values, n_permutations=args.n_permutations, rng=rng
        )

        # --- CKA ---
        # Kosinus-Ähnlichkeit als ANN-Kernel: K_DNN = 1 - Kosinus-Distanz
        # Vergleicht direkt die Ähnlichkeitsstruktur (nicht nur Ränge wie RSA)
        K_dnn = 1.0 - ann_matrix
        np.fill_diagonal(K_dnn, 1.0)
        cka = kernel_cka(K_dnn, K_human)

        rows.append({
            "layer": layer_name,
            "metric": metric,
            "spearman_r": round(rho, 6),
            "spearman_p": round(p_val, 6),
            "cka": round(cka, 6) if not np.isnan(cka) else "nan",
        })
        print(f"{layer_name}: Spearman ρ={rho:.4f} (p={p_val:.4f}), CKA={cka:.4f}")

    # --- Ergebnisse als CSV speichern ---
    output_csv = results_dir / f"{experiment_name}_geometric_alignment.csv"
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["layer", "metric", "spearman_r", "spearman_p", "cka"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Gespeichert: {output_csv}")


if __name__ == "__main__":
    main()

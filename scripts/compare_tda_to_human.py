"""Schritt 4b: Topologischer Vergleich DNN-Schichten vs. menschliche Referenz.

Für jede Kombination aus Schicht × Preprocessing-Modus × Homologiedimension
wird die 2-Wasserstein-Distanz zwischen dem DNN-Persistenzdiagramm und dem
menschlichen Referenz-Persistenzdiagramm berechnet.

Kleinere Wasserstein-Distanz = ähnlichere topologische Struktur.
"""

import argparse
import csv
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Kein Display nötig — verhindert macOS GUI-Hang
import matplotlib.pyplot as plt
import numpy as np
from persim import wasserstein
from ripser import ripser

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.utils import ensure_dir, load_config, project_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dummy_resnet18.json")
    args = parser.parse_args()

    config = load_config(args.config)
    experiment_name = config["experiment_name"]
    tda_config = config["tda"]
    maxdim = tda_config["homology_maxdim"]  # 1 → H0 und H1
    modes = tda_config["modes"]

    tda_dir = ensure_dir(tda_config["output_dir"])
    results_dir = ensure_dir(config["results"]["output_dir"])
    figure_dir = ensure_dir(config["figures"]["output_dir"])

    # --- Human-Referenz-Persistenzdiagramm berechnen ---
    # Die THINGS-Distanzmatrix wird direkt als Distanzmatrix für Ripser verwendet.
    # distance_matrix=True: Ripser interpretiert Input als vorberechnete Distanzen
    # (kein erneutes Berechnen paarweiser Abstände nötig)
    human_matrix = np.load(project_path(config["human"]["distance_matrix"]))
    human_diagrams = ripser(human_matrix, distance_matrix=True, maxdim=maxdim)["dgms"]

    all_rows = []  # Alle Ergebnisse für das kombinierte CSV

    for mode in modes:
        for hdim in range(maxdim + 1):  # H0, H1 (ggf. H2)
            human_diagram = human_diagrams[hdim]
            rows = []  # Ergebnisse für diesen Modus × Dimension (für den Plot)

            for layer_name in config["model"]["layers"]:
                # DNN-Persistenzdiagramm laden (aus compute_tda.py)
                path = tda_dir / f"{experiment_name}_{layer_name}_{mode}.pkl"
                with open(path, "rb") as f:
                    payload = pickle.load(f)

                layer_diagram = payload["diagrams"][hdim]

                # 2-Wasserstein-Distanz: optimaler Transport zwischen den
                # Persistenzpunkten beider Diagramme (inkl. Diagonale als Fallback)
                distance = float(wasserstein(layer_diagram, human_diagram))

                row = {
                    "layer": layer_name,
                    "tda_mode": mode,
                    "homology_dimension": hdim,
                    "wasserstein_distance_to_human": round(distance, 6),
                }
                rows.append(row)
                all_rows.append(row)
                print(f"{layer_name} ({mode}, H{hdim}): Wasserstein = {distance:.4f}")

            # --- Plot: Wasserstein-Distanz pro Schicht ---
            plt.figure(figsize=(7, 4))
            plt.plot(
                [r["layer"] for r in rows],
                [r["wasserstein_distance_to_human"] for r in rows],
                marker="o",
            )
            plt.xlabel("Schicht")
            plt.ylabel(f"Wasserstein-Distanz zu Human H{hdim}")
            plt.title(f"Topologische Übereinstimmung ({mode}, H{hdim})")
            plt.grid(alpha=0.25)
            plt.tight_layout()
            plt.savefig(
                figure_dir / f"{experiment_name}_topological_alignment_{mode}_H{hdim}.png",
                dpi=180,
            )
            plt.close()

    # --- Kombiniertes CSV: alle Modi × Dimensionen × Schichten ---
    output_csv = results_dir / f"{experiment_name}_topological_alignment_all.csv"
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["layer", "tda_mode", "homology_dimension", "wasserstein_distance_to_human"],
        )
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nGespeichert: {output_csv}")


if __name__ == "__main__":
    main()

"""Schritt 2a: Repräsentationale Distanzmatrizen (RDMs) berechnen.

Für jede Schicht wird eine n×n Distanzmatrix erstellt, die für jedes
Stimuluspaar den Abstand im Aktivierungsraum angibt.
Standard-Metrik: Kosinus-Distanz (nach Z-Score).
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.utils import ensure_dir, load_config


def pairwise_cosine_distance(features):
    """Kosinus-Distanzmatrix: D_ij = 1 - cos(x_i, x_j).

    Schritte:
    1. Z-Score — Skalierungsunterschiede zwischen Kanälen ausgleichen
    2. L2-Normierung — jeden Vektor auf Einheitslänge bringen
    3. Matrixmultiplikation — alle paarweisen Kosinus-Ähnlichkeiten auf einmal
    4. 1 - Ähnlichkeit = Distanz

    Kosinus-Distanz ∈ [0, 2], bei normalisierten Aktivierungen meist [0, 1].
    Invariant gegenüber isotroper Skalierung — nur die Richtung der Vektoren zählt.
    """
    # Z-Score: mean=0, std=1 pro Feature-Spalte
    means = features.mean(axis=0, keepdims=True)
    stds = features.std(axis=0, keepdims=True)
    stds[stds == 0] = 1.0
    features = (features - means) / stds

    # L2-Normierung: jeden Stimulusvektor auf Länge 1 bringen
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized = features / norms

    # Alle paarweisen Ähnlichkeiten: (n, d) @ (d, n) = (n, n)
    similarity = normalized @ normalized.T
    return 1.0 - similarity


def pairwise_euclidean_distance(features):
    """Euklidische Distanzmatrix: D_ij = ||x_i - x_j||.

    Effiziente Berechnung über die Binomialformel:
    ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2 * x_i · x_j
    """
    squared = np.sum(features**2, axis=1, keepdims=True)
    distances_sq = squared + squared.T - 2 * (features @ features.T)
    return np.sqrt(np.maximum(distances_sq, 0.0))  # maximum(.,0) vermeidet numerische Negativwerte


def compute_distance(features, metric):
    """Distanzmatrix nach gewählter Metrik berechnen."""
    if metric == "cosine":
        return pairwise_cosine_distance(features)
    if metric == "euclidean":
        return pairwise_euclidean_distance(features)
    raise ValueError(f"Unbekannte Metrik: {metric}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/minimal_resnet18.json")
    args = parser.parse_args()

    config = load_config(args.config)
    experiment_name = config["experiment_name"]
    metric = config["geometry"]["metric"]

    feature_dir = ensure_dir(config["features"]["output_dir"])
    geometry_dir = ensure_dir(config["geometry"]["output_dir"])

    for layer_name in config["model"]["layers"]:
        # Features laden: (n_stimuli, n_channels)
        features = np.load(feature_dir / f"{experiment_name}_{layer_name}.npy")
        # RDM berechnen: (n_stimuli, n_stimuli)
        distances = compute_distance(features, metric)

        matrix_path = geometry_dir / f"{experiment_name}_{layer_name}_{metric}.npy"
        np.save(matrix_path, distances)

        print(f"RDM gespeichert für {layer_name}: {distances.shape}")


if __name__ == "__main__":
    main()

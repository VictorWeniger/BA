"""Schritt 3: Topologische Datenanalyse — Persistenzdiagramme pro Schicht.

Für jede Schicht × Preprocessing-Modus wird ein Vietoris-Rips-Komplex
berechnet und die persistente Homologie für H0 (Zusammenhangskomponenten)
und H1 (Schleifen) bestimmt.

Preprocessing-Modi:
  pca50    — Z-Score + PCA auf 50 Dimensionen (Galella et al. 2025)
  pca10    — Z-Score + PCA auf 10 Dimensionen
  original — nur Z-Score, keine Dimensionsreduktion
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
from ripser import ripser

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.pca_utils import pca_numpy, standardize
from src.utils import ensure_dir, load_config


# Verfügbare PCA-Modi mit ihrer Zieldimension
_PCA_MODES = {
    "pca3":  3,
    "pca10": 10,
    "pca50": 50,
}


def make_tda_input(features, mode, max_points):
    """Features für Ripser vorbereiten.

    Warum Begrenzung auf max_points: Ripser berechnet alle paarweisen Abstände
    und hat O(n²) Speicherbedarf — bei 1854 Stimuli zu langsam/speicherintensiv.

    Warum Dimensionsreduktion: Ripser mit 2048 Dimensionen ist sehr langsam.
    PCA auf 50 Dimensionen behält den Großteil der Varianz und ist handhabbar.
    """
    subset = features[:max_points]  # Erste max_points Stimuli nehmen

    if mode in _PCA_MODES:
        # Z-Score ist in pca_numpy() enthalten
        return pca_numpy(subset, _PCA_MODES[mode])
    if mode == "original":
        # Z-Score ohne Dimensionsreduktion
        return standardize(subset)

    raise ValueError(f"Unbekannter TDA-Modus: {mode}. Erlaubt: {list(_PCA_MODES) + ['original']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/minimal_resnet18.json")
    args = parser.parse_args()

    config = load_config(args.config)
    experiment_name = config["experiment_name"]
    tda_config = config["tda"]

    feature_dir = ensure_dir(config["features"]["output_dir"])
    tda_dir = ensure_dir(tda_config["output_dir"])

    for layer_name in config["model"]["layers"]:
        features = np.load(feature_dir / f"{experiment_name}_{layer_name}.npy")

        for mode in tda_config["modes"]:
            # Features vorverarbeiten
            tda_input = make_tda_input(features, mode, tda_config["max_points"])

            # Vietoris-Rips-Persistenz berechnen
            # maxdim=1: H0 (Komponenten) und H1 (Schleifen) — H2 zu langsam
            # Ergebnis: dict mit "dgms" = Liste von Persistenzdiagrammen [H0, H1, ...]
            diagrams = ripser(tda_input, maxdim=tda_config["homology_maxdim"])["dgms"]

            # Als pickle speichern (Persistenzdiagramme sind Listen von Arrays
            # mit variabler Länge — .npy nicht geeignet)
            output_base = f"{experiment_name}_{layer_name}_{mode}"
            with open(tda_dir / f"{output_base}.pkl", "wb") as f:
                pickle.dump({"layer": layer_name, "mode": mode, "diagrams": diagrams}, f)

            print(f"TDA gespeichert: {layer_name} ({mode})")


if __name__ == "__main__":
    main()

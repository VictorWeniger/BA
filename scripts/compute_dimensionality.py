"""Schritt 2b: Effektive (ED) und Intrinsische Dimensionalität (ID) pro Schicht.

ED  — lineares Maß: Participation Ratio der Eigenwerte der Kovarianzmatrix
      (Galella et al. 2025, Gl. 1). Fragt: Wie viele Dimensionen werden aktiv genutzt?

ID  — nichtlineares Maß: MLE-Schätzer (Levina & Bickel 2004 / Galella et al. 2025, Gl. 2).
      Fragt: Auf welcher Dimension liegt die Datenmannigfaltigkeit tatsächlich?
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.pca_utils import standardize
from src.utils import ensure_dir, load_config


def effective_dimensionality(features):
    """ED = (Σλ_i)² / Σλ_i² — Participation Ratio der Eigenwerte (Galella Gl. 1).

    Interpretation: Wenn alle Eigenwerte gleich groß wären (Varianz gleichmäßig
    verteilt), wäre ED = d (volle Dimensionalität). Wenn nur ein Eigenwert dominiert,
    wäre ED ≈ 1. ED misst also wie gleichmäßig der Repräsentationsraum genutzt wird.

    Wichtig: Kein vollständiger Z-Score vor ED — die Varianzunterschiede zwischen
    Dimensionen sind die eigentliche Information. Nur Zentrierung (mean=0).
    """
    centered = features - features.mean(axis=0)
    cov = (centered.T @ centered) / len(features)
    eigenvalues = np.linalg.eigvalsh(cov)       # Alle Eigenwerte der Kovarianzmatrix
    eigenvalues = eigenvalues[eigenvalues > 0]  # Numerisch negative Eigenwerte ignorieren
    return float((eigenvalues.sum() ** 2) / (eigenvalues ** 2).sum())


def _knn_distances(features, k):
    """k nächste Nachbarn für jeden Punkt berechnen (Brute-Force, reines NumPy).

    Gibt eine (n, k) Matrix zurück: Zeile i enthält die Abstände zu den
    k nächsten Nachbarn von Punkt i, aufsteigend sortiert.
    """
    # Paarweise quadratische Distanzen via Binomialformel
    sq = np.sum(features ** 2, axis=1)
    dist_sq = sq[:, None] + sq[None, :] - 2.0 * (features @ features.T)
    # Diagonale auf inf setzen — Abstand zu sich selbst ausschließen
    np.fill_diagonal(dist_sq, np.inf)
    dist = np.sqrt(np.maximum(dist_sq, 0.0))
    # Jede Zeile sortieren und die ersten k Spalten behalten
    sorted_dist = np.sort(dist, axis=1)[:, :k]
    return sorted_dist


def intrinsic_dimensionality(features, k=20):
    """ID via MLE-Schätzer (Levina & Bickel 2004), Galella Gl. 2.

    Idee: In d Dimensionen wächst die Anzahl der Punkte in einem Ball mit
    Radius r wie r^d. Der MLE schätzt d aus den Verhältnissen der
    Nachbarabstände — ohne Linearitätsannahme.

    Formel: ID = [ 1/(n*(k-1)) * Σ_i Σ_{j=1}^{k-1} log(T_k(x_i)/T_j(x_i)) ]^{-1}
    T_j(x_i) = Abstand von x_i zu seinem j-ten nächsten Nachbarn.

    Z-Score vor ID notwendig: Nearest-Neighbor-Distanzen sollen nicht von
    der Skalierung einzelner Features abhängen.
    """
    knn = _knn_distances(features, k)     # (n, k): Abstände zu den k Nachbarn
    T_k = knn[:, -1]                      # Abstand zum k-ten (weitesten) Nachbarn
    T_j = knn[:, :-1]                     # Abstände zu Nachbarn 1..k-1
    # log(0) vermeiden — Nullabstände auf NaN setzen
    T_j = np.where(T_j > 0, T_j, np.nan)
    T_k_col = T_k[:, None]
    log_ratios = np.log(T_k_col / T_j)   # (n, k-1): Log-Verhältnisse
    mean_log_ratio = np.nanmean(log_ratios)
    if mean_log_ratio <= 0:
        return np.nan
    return float(1.0 / mean_log_ratio)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/resnet50.json")
    args = parser.parse_args()

    config = load_config(args.config)
    experiment_name = config["experiment_name"]
    dim_config = config.get("dimensionality", {})
    k = dim_config.get("id_k_neighbors", 20)          # Anzahl Nachbarn für ID
    n_subsample = dim_config.get("n_pca_subsample", 50)  # Zufällige Feature-Dimensionen für ID

    feature_dir = ensure_dir(config["features"]["output_dir"])
    out_dir = ensure_dir(dim_config.get("output_dir", "outputs/dimensionality"))

    rows = []
    for layer_name in config["model"]["layers"]:
        features = np.load(feature_dir / f"{experiment_name}_{layer_name}.npy")
        # Z-Score vor allen Dimensionalitätsmaßen
        features = standardize(features)

        # ED: auf allen Dimensionen, ohne weitere Vorverarbeitung
        ed = effective_dimensionality(features)

        # ID: auf Subsample von n_subsample Dimensionen (Galella: 50 GAP-Features)
        # Subsample reduziert Rechenzeit — brute-force kNN skaliert O(n² * d)
        if features.shape[1] > n_subsample:
            rng = np.random.default_rng(0)
            idx = rng.choice(features.shape[1], size=n_subsample, replace=False)
            features_sub = features[:, idx]
        else:
            features_sub = features

        id_val = intrinsic_dimensionality(features_sub, k=k)

        rows.append({
            "layer": layer_name,
            "effective_dimensionality": round(ed, 4),
            "intrinsic_dimensionality": round(id_val, 4) if not np.isnan(id_val) else "nan",
        })
        print(f"{layer_name}: ED={ed:.3f}, ID={id_val:.3f}")

    out_csv = out_dir / f"{experiment_name}_dimensionality.csv"
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["layer", "effective_dimensionality", "intrinsic_dimensionality"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Gespeichert: {out_csv}")


if __name__ == "__main__":
    main()

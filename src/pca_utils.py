import numpy as np


def standardize(features):
    """Z-Score-Normalisierung: jede Feature-Spalte auf mean=0, std=1 bringen.

    Warum: Verhindert, dass Features mit großer Skala (z.B. 0–1000) die
    Distanzberechnungen dominieren gegenüber Features mit kleiner Skala (0–1).
    Spalten mit std=0 (konstante Features) werden nicht dividiert.
    """
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True)
    std[std == 0] = 1.0  # Konstante Features unverändert lassen
    return (features - mean) / std


def pca_numpy(features, n_components):
    """PCA mit reinem NumPy — kein scikit-learn nötig.

    Schritte:
    1. Z-Score (standardize) — alle Dimensionen gleichgewichtig behandeln
    2. Zentrierung — Mittelwert auf 0 (für SVD notwendig)
    3. SVD — Singulärwertzerlegung liefert Hauptkomponenten
    4. Projektion auf die n_components stärksten Richtungen

    Verwendung im Projekt: TDA-Preprocessing (pca50, pca10 Modi),
    damit Ripser nicht mit 2048 Dimensionen umgehen muss.
    """
    scaled = standardize(features)
    centered = scaled - scaled.mean(axis=0, keepdims=True)
    # vt enthält die Hauptkomponenten als Zeilenvektoren (absteigend nach Varianz)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    n_components = min(n_components, vt.shape[0])
    components = vt[:n_components].T  # (d, n_components)
    return centered @ components       # (n, n_components)

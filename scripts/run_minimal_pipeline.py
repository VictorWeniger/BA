"""Orchestrierung: Alle Pipeline-Schritte der Reihe nach ausführen.

Aufruf:
    python run_minimal_pipeline.py --config configs/resnet50.json
    python run_minimal_pipeline.py --config configs/resnet50.json --generate-dummy

Reihenfolge:
    1. extract_resnet18_features   → .npy Aktivierungsmatrizen pro Schicht
    2. compute_dimensionality      → ED und ID pro Schicht (CSV)
    3. compute_geometry            → RDMs pro Schicht (.npy)
    4. compute_tda                 → Persistenzdiagramme pro Schicht × Modus (.pkl)
    5. visualize_features          → PCA/t-SNE-Plots
    6. visualize_rdms              → RDM-Heatmaps (PNG)
    7. compute_layer_cka           → Layer×Layer CKA-Matrix (CSV + PNG)
    8. compare_geometry_to_human   → Spearman-ρ, p-Wert, CKA vs. THINGS (CSV)
    9. compare_tda_to_human        → Wasserstein-Distanzen vs. THINGS (CSV + Plots)
"""

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_step(command):
    """Einen Pipeline-Schritt als Subprocess ausführen und bei Fehler abbrechen."""
    print("\n" + "=" * 80)
    print(" ".join(command))
    print("=" * 80)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/minimal_resnet18.json",
                        help="Pfad zur Experiment-Config (JSON)")
    parser.add_argument("--generate-dummy", action="store_true",
                        help="Dummy-Stimuli generieren (wenn keine echten Bilder vorhanden)")
    args = parser.parse_args()

    python = sys.executable  # Gleiche Python-Umgebung für alle Schritte verwenden

    # Optionaler Dummy-Modus: Testbilder erzeugen statt echte THINGS-Bilder zu laden
    if args.generate_dummy:
        run_step([python, "scripts/generate_dummy_stimuli.py"])

    # Schritt 1: Features extrahieren
    run_step([python, "scripts/extract_resnet18_features.py", "--config", args.config])

    # Schritt 2: Dimensionalität (ED + ID)
    run_step([python, "scripts/compute_dimensionality.py", "--config", args.config])

    # Schritt 3: RDMs (Kosinus-Distanzmatrizen)
    run_step([python, "scripts/compute_geometry.py", "--config", args.config])

    # Schritt 4: TDA (Persistenzdiagramme für alle Modi)
    run_step([python, "scripts/compute_tda.py", "--config", args.config])

    # Schritt 5–7: Visualisierungen
    run_step([python, "scripts/visualize_features.py",  "--config", args.config])
    run_step([python, "scripts/visualize_rdms.py",      "--config", args.config])
    run_step([python, "scripts/compute_layer_cka.py",   "--config", args.config])

    # Schritt 8–9: Vergleiche gegen Human-Referenz
    run_step([python, "scripts/compare_geometry_to_human.py", "--config", args.config])
    run_step([python, "scripts/compare_tda_to_human.py",      "--config", args.config])

    print("\nPipeline abgeschlossen.")


if __name__ == "__main__":
    main()

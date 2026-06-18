import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.html_plots import write_3d_scatter_html
from src.pca_utils import pca_numpy
from src.utils import ensure_dir, load_config, read_stimuli_csv


def plot_pca_2d(coords, labels, title, output_path):
    unique = sorted(set(labels))
    label_to_color = {label: idx for idx, label in enumerate(unique)}
    colors = [label_to_color[label] for label in labels]

    plt.figure(figsize=(6, 5))
    scatter = plt.scatter(coords[:, 0], coords[:, 1], c=colors, cmap="tab20", s=18, alpha=0.8)
    plt.title(title)
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/minimal_resnet18.json")
    parser.add_argument("--label-column", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    experiment_name = config["experiment_name"]
    figure_dir = ensure_dir(config["figures"]["output_dir"])
    feature_dir = ensure_dir(config["features"]["output_dir"])

    label_column = args.label_column or config["data"]["concept_column"]
    rows = read_stimuli_csv(config["data"]["stimuli_csv"])
    labels = [row[label_column] for row in rows]

    for layer_name in config["model"]["layers"]:
        feature_path = feature_dir / f"{experiment_name}_{layer_name}.npy"
        features = np.load(feature_path)

        coords_2d = pca_numpy(features, 2)
        plot_pca_2d(
            coords_2d,
            labels,
            f"2D-PCA: {layer_name}",
            figure_dir / f"{experiment_name}_{layer_name}_pca2d.png",
        )

        coords_3d = pca_numpy(features, 3)
        write_3d_scatter_html(
            coords_3d,
            labels,
            f"3D-PCA: {layer_name}",
            figure_dir / f"{experiment_name}_{layer_name}_pca3d.html",
        )

        print(f"Saved PCA plots for {layer_name}")


if __name__ == "__main__":
    main()

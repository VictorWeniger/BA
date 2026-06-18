"""
Compute pairwise linear CKA between all layer pairs.

Saves layer x layer CKA matrix as CSV and PNG heatmap (no GUI required).
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.png_utils import matrix_to_rgb, write_png
from src.stats_utils import linear_cka
from src.utils import ensure_dir, load_config

_BLUES = [
    (240, 248, 255), (198, 219, 239), (158, 202, 225),
    (107, 174, 214), ( 49, 130, 189), (  8,  81, 156),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/resnet50.json")
    args = parser.parse_args()

    config = load_config(args.config)
    experiment_name = config["experiment_name"]
    layers = config["model"]["layers"]

    feature_dir = ensure_dir(config["features"]["output_dir"])
    results_dir = ensure_dir(config["results"]["output_dir"])
    figure_dir = ensure_dir(config["figures"]["output_dir"])

    features = {l: np.load(feature_dir / f"{experiment_name}_{l}.npy") for l in layers}

    n = len(layers)
    cka_matrix = np.zeros((n, n))
    rows = []

    for i, l1 in enumerate(layers):
        for j, l2 in enumerate(layers):
            val = linear_cka(features[l1], features[l2])
            cka_matrix[i, j] = 0.0 if np.isnan(val) else float(val)
            if j >= i:
                rows.append({"layer_1": l1, "layer_2": l2, "linear_cka": round(float(val), 6)})
        print(f"  {l1}: done")

    out_csv = results_dir / f"{experiment_name}_layer_cka.csv"
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["layer_1", "layer_2", "linear_cka"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {out_csv}")

    rgb = matrix_to_rgb(cka_matrix, vmin=0, vmax=1, cmap=_BLUES)
    cell = 60
    rgb_scaled = np.repeat(np.repeat(rgb, cell, axis=0), cell, axis=1)
    out_fig = figure_dir / f"{experiment_name}_layer_cka_heatmap.png"
    write_png(out_fig, rgb_scaled)
    print(f"Saved {out_fig}")


if __name__ == "__main__":
    main()

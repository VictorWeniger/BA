"""
Visualize RDMs as heatmaps for all layers.

Uses only numpy + stdlib (no GUI required) via src.png_utils.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.png_utils import matrix_to_rgb, tile_images, write_png
from src.utils import ensure_dir, load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/resnet50.json")
    args = parser.parse_args()

    config = load_config(args.config)
    experiment_name = config["experiment_name"]
    metric = config["geometry"]["metric"]
    layers = config["model"]["layers"]

    geometry_dir = ensure_dir(config["geometry"]["output_dir"])
    figure_dir = ensure_dir(config["figures"]["output_dir"])

    matrices = {l: np.load(geometry_dir / f"{experiment_name}_{l}_{metric}.npy") for l in layers}
    vmax = max(m.max() for m in matrices.values())

    tiles = []
    for layer_name, matrix in matrices.items():
        rgb = matrix_to_rgb(matrix, vmin=0, vmax=vmax)
        # Scale to 200×200
        scale = 200 / rgb.shape[0]
        h_out, w_out = int(rgb.shape[0] * scale), int(rgb.shape[1] * scale)
        rgb_scaled = np.kron(rgb, np.ones((max(1, h_out // rgb.shape[0]),
                                           max(1, w_out // rgb.shape[1]), 1), dtype=np.uint8))[:h_out, :w_out]
        tiles.append(rgb_scaled)

        out = figure_dir / f"{experiment_name}_{layer_name}_rdm.png"
        write_png(out, rgb_scaled)
        print(f"Saved {out}")

    overview = tile_images(tiles, ncols=3)
    out = figure_dir / f"{experiment_name}_rdms_overview.png"
    write_png(out, overview)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.utils import ensure_dir, load_config, read_stimuli_csv, write_csv


FEATURE_DIMS = {
    "conv1": 64,
    "layer1": 64,
    "layer2": 128,
    "layer3": 256,
    "layer4": 512,
    "fc": 1000,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dummy_resnet18.json")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    config = load_config(args.config)
    rows = read_stimuli_csv(config["data"]["stimuli_csv"])
    if not rows:
        raise RuntimeError("No stimuli rows found. Run generate_dummy_stimuli.py first.")

    concepts = sorted({row[config["data"]["concept_column"]] for row in rows})
    concept_to_index = {concept: idx for idx, concept in enumerate(concepts)}

    output_dir = ensure_dir(config["features"]["output_dir"])
    experiment_name = config["experiment_name"]

    for layer_position, layer_name in enumerate(config["model"]["layers"]):
        dim = FEATURE_DIMS.get(layer_name, 128)
        centers = rng.normal(0, 1 + 0.2 * layer_position, size=(len(concepts), dim))
        noise_scale = max(0.25, 1.2 - 0.15 * layer_position)

        features = []
        for row in rows:
            concept_index = concept_to_index[row[config["data"]["concept_column"]]]
            feature = centers[concept_index] + rng.normal(0, noise_scale, size=dim)
            features.append(feature.astype(np.float32))

        matrix = np.vstack(features)
        output_path = output_dir / f"{experiment_name}_{layer_name}.npy"
        np.save(output_path, matrix)
        print(f"Saved dummy features for {layer_name}: {matrix.shape}")

    write_csv(
        rows,
        output_dir / f"{experiment_name}_image_order.csv",
        fieldnames=list(rows[0].keys()),
    )


if __name__ == "__main__":
    main()

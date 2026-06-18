import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.utils import load_config, project_path, read_stimuli_csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dummy_resnet18.json")
    args = parser.parse_args()

    config = load_config(args.config)
    rows = read_stimuli_csv(config["data"]["stimuli_csv"])
    concept_column = config["data"]["concept_column"]
    category_column = config["data"]["category_column"]

    n = len(rows)
    matrix = np.zeros((n, n), dtype=np.float32)

    for i, row_i in enumerate(rows):
        for j, row_j in enumerate(rows):
            same_concept = row_i[concept_column] == row_j[concept_column]
            same_category = row_i[category_column] == row_j[category_column]

            if i == j:
                distance = 0.0
            elif same_concept:
                distance = 0.1
            elif same_category:
                distance = 0.45
            else:
                distance = 1.0

            matrix[i, j] = distance

    output_path = project_path(config["human"]["distance_matrix"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, matrix)

    print(f"Saved dummy human matrix: {matrix.shape} -> {output_path}")


if __name__ == "__main__":
    main()

import argparse
import csv
import json
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.utils import project_path, write_csv


def load_json(path):
    """Config relativ zum Projektordner laden."""
    with open(project_path(path), "r", encoding="utf-8") as f:
        return json.load(f)


def read_csv_rows(path):
    """CSV vollständig als Liste von Dict-Zeilen laden."""
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def find_column_index(csv_path, column_name):
    """Spaltenposition einer ROI in einer ROI-CSV finden."""
    with open(csv_path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split(",")
    if column_name not in header:
        raise KeyError(f"Column '{column_name}' not found in {csv_path}")
    return header.index(column_name)


def load_roi_mask(subject_dir, roi_type, roi_name):
    """
    Load one ROI column as a boolean mask.

    This currently targets volume-space ROI CSVs. Those files have one row per
    volume beta feature, matching betas/vol/<subject>_betas_vol.npy.
    """

    roi_path = subject_dir / "rois" / roi_type / f"{subject_dir.name}_{roi_type}.csv"
    column_index = find_column_index(roi_path, roi_name)
    mask = np.loadtxt(
        roi_path,
        delimiter=",",
        skiprows=1,
        usecols=[column_index],
        dtype=np.uint8,
    ).astype(bool)
    return mask


def pairwise_cosine_distance(features):
    """
    Cosine-RDM mit Feature-Z-Score.

    Diese Variante ist die allgemeine Fallback-Variante: Erst werden alle
    Feature-Spalten standardisiert, dann werden Stimulusvektoren L2-normalisiert.
    Wenn vorher schon per Session z-standardisiert wurde, nutzt der Code unten
    stattdessen `pairwise_cosine_distance_l2only`.
    """
    means = features.mean(axis=0, keepdims=True)
    stds = features.std(axis=0, keepdims=True)
    stds[stds == 0] = 1.0
    features = (features - means) / stds
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized = features / norms
    similarity = normalized @ normalized.T
    return (1.0 - similarity).astype(np.float32)


def pairwise_cosine_distance_l2only(features):
    """Cosine distance with L2 normalization only (no z-scoring).
    Used after per-session z-scoring, which already removes voxel-wise mean/variance."""
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized = features / norms
    similarity = normalized @ normalized.T
    return (1.0 - similarity).astype(np.float32)


def per_session_zscore(features, rows):
    """Z-score betas across conditions (stimuli) within each session separately.
    This removes session-specific scanner drift and physiological noise before aggregation."""
    sessions = [r["Session"] for r in rows]
    unique_sessions = sorted(set(sessions))
    features_out = features.copy()
    for sess in unique_sessions:
        idx = [i for i, s in enumerate(sessions) if s == sess]
        subset = features_out[idx, :]
        means = subset.mean(axis=0, keepdims=True)
        stds = subset.std(axis=0, keepdims=True)
        stds[stds == 0] = 1.0
        features_out[idx, :] = (subset - means) / stds
    return features_out


def filter_rows(rows, trial_filter):
    """Trial-Zeilen nach Split, Dataset und optional max_trials filtern."""
    indices = list(range(len(rows)))

    split = trial_filter.get("split")
    if split:
        indices = [idx for idx in indices if rows[idx].get("Split") == split]

    dataset = trial_filter.get("dataset")
    if dataset:
        indices = [idx for idx in indices if rows[idx].get("Dataset") == dataset]

    max_trials = trial_filter.get("max_trials")
    if max_trials:
        indices = indices[: int(max_trials)]

    return indices


def aggregate_features(features, rows, group_by, method="mean"):
    """
    Aggregate repeated rows, e.g. multiple trials of the same Stimulus or Concept.
    """

    groups = OrderedDict()
    for idx, row in enumerate(rows):
        key = row[group_by]
        groups.setdefault(key, []).append(idx)

    aggregated_features = []
    aggregated_rows = []

    for key, indices in groups.items():
        # Alle Wiederholungen desselben Stimulus werden zu einem stabileren
        # Stimulus-Aktivierungsvektor gemittelt.
        group_features = features[indices]
        if method != "mean":
            raise ValueError(f"Unsupported aggregation method: {method}")

        aggregated_features.append(group_features.mean(axis=0))

        base_row = dict(rows[indices[0]])
        base_row["AggregatedBy"] = group_by
        base_row["AggregatedKey"] = key
        base_row["NAggregatedTrials"] = str(len(indices))
        aggregated_rows.append(base_row)

    return np.vstack(aggregated_features).astype(np.float32), aggregated_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/processed_fmri_s1_loc.json")
    parser.add_argument("--subject", default=None)
    parser.add_argument("--roi", default=None)
    parser.add_argument("--roi-type", default=None)
    parser.add_argument("--max-trials", type=int, default=None)
    parser.add_argument("--experiment-name", default=None)
    args = parser.parse_args()

    # Config laden und optionale CLI-Argumente darüberlegen. Dadurch kann man
    # dasselbe Template für S1/S2/S3 und verschiedene ROIs wiederverwenden.
    config = load_json(args.config)
    if args.subject:
        config["subject"] = args.subject
    if args.roi:
        config.setdefault("roi", {})["name"] = args.roi
    if args.roi_type:
        config.setdefault("roi", {})["type"] = args.roi_type
    if args.max_trials is not None:
        config.setdefault("trial_filter", {})["max_trials"] = args.max_trials
    if args.experiment_name:
        config["experiment_name"] = args.experiment_name

    processed_root = Path(config["processed_root"])
    subject = config["subject"]
    subject_dir = processed_root / subject

    stimuli_path = subject_dir / "stimuli" / f"{subject}_stimuli.csv"
    # Stimulus-CSV und Beta-Matrix sind zeilenweise gekoppelt:
    # rows[i] beschreibt betas[i].
    rows = read_csv_rows(stimuli_path)
    row_indices = filter_rows(rows, config.get("trial_filter", {}))
    selected_rows = [rows[idx] for idx in row_indices]

    if not selected_rows:
        raise RuntimeError("No trials selected. Check trial_filter in config.")

    if config["space"] != "vol":
        raise NotImplementedError("This importer currently supports space='vol'.")

    betas_path = subject_dir / "betas" / "vol" / f"{subject}_betas_vol.npy"
    # mmap_mode vermeidet, dass die große Beta-Matrix vollständig in den RAM
    # geladen wird, bevor die ROI-Spalten ausgewählt werden.
    betas = np.load(betas_path, mmap_mode="r")

    roi_config = config.get("roi")
    if roi_config:
        roi_mask = load_roi_mask(subject_dir, roi_config["type"], roi_config["name"])
        if roi_mask.shape[0] != betas.shape[1]:
            raise ValueError(
                f"ROI mask length {roi_mask.shape[0]} does not match beta features {betas.shape[1]}"
            )
        if roi_mask.sum() == 0:
            raise ValueError(f"ROI {roi_config['name']} is empty for {subject}.")

        # Erst relevante Trials, dann relevante ROI-Voxel auswählen.
        # Ergebnis bleibt eine Matrix: Trials × ROI-Voxel.
        features = betas[np.array(row_indices)][:, roi_mask]
        feature_label = f"{subject}_{config['space']}_{roi_config['type']}_{roi_config['name']}"
    else:
        features = betas[np.array(row_indices)]
        feature_label = f"{subject}_{config['space']}_wholebrain"

    features = np.asarray(features, dtype=np.float32)
    print(f"Selected fMRI features {feature_label}: {features.shape}")

    # Per-session z-scoring must happen before aggregation (Santi/Galella et al.)
    # Aktueller BA-Workflow: pro Session z-standardisieren, bevor über
    # Wiederholungen gemittelt wird. Das reduziert session-spezifische Drifts.
    use_session_zscore = config.get("normalization", {}).get("per_session_zscore", False)
    if use_session_zscore:
        features = per_session_zscore(features, selected_rows)
        print(f"Applied per-session z-scoring across {len(set(r['Session'] for r in selected_rows))} sessions.")

    aggregation = config.get("aggregation", {})
    group_by = aggregation.get("group_by")
    if group_by:
        # Nach dem Session-Z-Score werden die 12 Wiederholungen pro Stimulus
        # gemittelt. Die Voxel-Dimension bleibt erhalten.
        features, selected_rows = aggregate_features(
            features,
            selected_rows,
            group_by=group_by,
            method=aggregation.get("method", "mean"),
        )
        print(f"Aggregated by {group_by}: {features.shape}")

    experiment_name = config.get("experiment_name", feature_label)
    default_stimuli = f"data/{experiment_name}_stimuli.csv"
    default_features = f"outputs/human/{experiment_name}_features.npy"
    default_distance = f"outputs/human/{experiment_name}_distance.npy"

    output_features = project_path(config.get("outputs", {}).get("features_npy", default_features))
    if args.experiment_name:
        output_features = project_path(default_features)
    output_features.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_features, features)

    # RDM berechnen. Wenn bereits per Session z-standardisiert wurde, wird hier
    # nur noch die Richtung der Stimulusvektoren über Cosine/L2 verglichen.
    if use_session_zscore:
        distance = pairwise_cosine_distance_l2only(features)
    else:
        distance = pairwise_cosine_distance(features)
    output_distance = project_path(config.get("outputs", {}).get("distance_matrix", default_distance))
    if args.experiment_name:
        output_distance = project_path(default_distance)
    output_distance.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_distance, distance)

    output_stimuli = project_path(config.get("outputs", {}).get("stimuli_csv", default_stimuli))
    if args.experiment_name:
        output_stimuli = project_path(default_stimuli)
    fieldnames = list(selected_rows[0].keys())
    write_csv(selected_rows, output_stimuli, fieldnames)

    print(f"Saved stimuli: {output_stimuli}")
    print(f"Saved features: {output_features}")
    print(f"Saved distance matrix: {output_distance} {distance.shape}")


if __name__ == "__main__":
    main()

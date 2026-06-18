import argparse
import csv
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.utils import project_path, write_csv


def read_rows(path):
    """fMRI-Stimulusliste relativ zum Projektordner laden."""
    with open(project_path(path), "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fmri-stimuli", default="data/processed_fmri_S1_LOC_stimuli.csv")
    parser.add_argument("--image-root", default="/Volumes/Sonstige Backups/Data/object_images")
    parser.add_argument("--output", default="data/ann_S1_LOC_matched_stimuli.csv")
    args = parser.parse_args()

    # Diese CSV kommt aus import_processed_fmri.py und enthält die bereits
    # aggregierten 100 Stimuli in der Reihenfolge der fMRI-RDM.
    rows = read_rows(args.fmri_stimuli)
    image_root = Path(args.image_root)

    output_rows = []
    missing = []

    for row in rows:
        stimulus = row["Stimulus"]
        concept = row["Concept"]
        # THINGS-Bildpfade sind nach Konzeptordnern organisiert:
        # object_images/<Concept>/<Stimulus>
        image_path = image_root / concept / stimulus

        if not image_path.exists():
            # Fehlende Bilder sind kritisch: Wenn wir einfach überspringen
            # würden, hätten ANN- und fMRI-RDM unterschiedliche Stimulusmengen.
            missing.append(str(image_path))
            continue

        output_rows.append(
            {
                "image_id": row.get("AggregatedKey") or stimulus,
                "concept": concept,
                "category": row.get("Dataset", "THINGS"),
                "image_path": str(image_path),
                "fmri_stimulus": stimulus,
                "fmri_subject": row.get("Subject", ""),
                "n_aggregated_trials": row.get("NAggregatedTrials", "1"),
            }
        )

    if missing:
        preview = "\n".join(missing[:10])
        raise FileNotFoundError(
            f"{len(missing)} image files are missing. First examples:\n{preview}"
        )

    write_csv(
        output_rows,
        args.output,
        fieldnames=[
            "image_id",
            "concept",
            "category",
            "image_path",
            "fmri_stimulus",
            "fmri_subject",
            "n_aggregated_trials",
        ],
    )

    print(f"Saved {len(output_rows)} matched ANN stimuli -> {project_path(args.output)}")


if __name__ == "__main__":
    main()

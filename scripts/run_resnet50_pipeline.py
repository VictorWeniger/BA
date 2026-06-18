"""Vollständige ResNet-50-Pipeline: ANN-Schichten vs. fMRI-ROIs.

Dieses Skript ist die ResNet-50-Entsprechung zu
``compare_resnet18_to_fmri_rois.py``. Es führt für jeden Probanden die
gesamte geometrische Analysekette aus und schreibt alle Ergebnisse in einen
eigenen Ausgabeordner (Standard: ``outputs_resnet50/``), damit der
ResNet-18-Lauf unverändert reproduzierbar bleibt.

Schritte pro Proband (S1, S2, S3):
  1. Gematchte Stimulusliste erzeugen: Die fMRI-Stimulusreihenfolge wird
     übernommen und der Bildpfad auf ``<images-root>/object_images/<concept>/<id>``
     umgeschrieben. Dadurch ist die Pipeline unabhängig von der externen
     Festplatte und lokal wie auf dem Cluster lauffähig.
  2. ResNet-50-Features extrahieren (scripts/extract_resnet18_features.py,
     modellagnostisch, ``model.name = resnet50``).
  3. Cosine-RDMs berechnen (scripts/compute_geometry.py).
  4. Pro ROI Spearman-RSA + CKA berechnen (scripts/compare_geometry_to_human.py).
  5. Ergebnisse zu einem Subject-Summary und einem Gesamt-Summary
     (alle Probanden) zusammenführen.

Beispiel (lokal):
  .venv/bin/python scripts/run_resnet50_pipeline.py \
      --images-root data/things_images \
      --output-root outputs_resnet50

Beispiel (Cluster, Bilder unter $WORK/things_images):
  python scripts/run_resnet50_pipeline.py \
      --images-root /work/dldevel/weniger/things_images \
      --output-root outputs_resnet50
"""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROIS = ["V1", "hV4", "LOC", "IT"]
LAYERS = ["conv1", "layer1", "layer2", "layer3", "layer4", "fc"]


def run_step(command):
    print("\n" + "=" * 80, flush=True)
    print(" ".join(str(c) for c in command), flush=True)
    print("=" * 80, flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def build_matched_stimuli(subject, images_root, out_csv):
    """Gematchte Stimulus-CSV mit umgeschriebenem Bildpfad erzeugen.

    Basis ist die bereits vorhandene ``ann_<subject>_matched_stimuli.csv``
    (gleiche 100 Stimuli für alle Probanden). Es wird nur die Spalte
    ``image_path`` auf den gewählten Bildwurzelordner gesetzt.
    """
    src = PROJECT_ROOT / "data" / f"ann_{subject}_matched_stimuli.csv"
    if not src.exists():
        raise FileNotFoundError(f"Gematchte Stimulusliste fehlt: {src}")

    rows = list(csv.DictReader(src.open(encoding="utf-8")))
    if not rows:
        raise RuntimeError(f"{src} ist leer.")

    images_root = Path(images_root)
    for row in rows:
        concept = row["concept"]
        image_id = row["image_id"]
        row["image_path"] = str(images_root / "object_images" / concept / image_id)

    out_csv = PROJECT_ROOT / out_csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return out_csv


def make_config(subject, model, stimuli_csv, output_root, human_roi):
    """Temporäre Config für einen Subject×ROI-Vergleich erzeugen."""
    config = {
        "experiment_name": f"ann_{model}_{subject}_matched",
        "model": {
            "name": model,
            "pretrained": True,
            "allow_untrained_fallback": False,
            "layers": LAYERS,
            "conv_pooling": "global_average",
        },
        "data": {
            "stimuli_csv": str(Path(stimuli_csv).relative_to(PROJECT_ROOT))
            if Path(stimuli_csv).is_absolute() else str(stimuli_csv),
            "image_id_column": "image_id",
            "concept_column": "concept",
            "category_column": "category",
            "image_path_column": "image_path",
        },
        "features": {"output_dir": f"{output_root}/features", "batch_size": 16},
        "geometry": {"metric": "cosine", "output_dir": f"{output_root}/geometry"},
        "human": {
            "distance_matrix": f"outputs/human/processed_fmri_{subject}_{human_roi}_distance.npy",
            "source": f"processed_fmri_{subject}_{human_roi}",
        },
        "results": {"output_dir": f"{output_root}/results"},
        "figures": {"output_dir": f"{output_root}/figures"},
    }
    return config


def write_config(config, path):
    path = PROJECT_ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="resnet50")
    parser.add_argument("--subjects", nargs="+", default=["S1", "S2", "S3"])
    parser.add_argument("--images-root", default="data/things_images",
                        help="Wurzelordner mit object_images/<concept>/<id>.")
    parser.add_argument("--output-root", default="outputs_resnet50",
                        help="Ausgabeordner relativ zum Projektstamm.")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--n-permutations", type=int, default=1000)
    args = parser.parse_args()

    python = args.python
    model = args.model
    output_root = args.output_root

    all_summary = []

    for subject in args.subjects:
        print(f"\n########## {subject} ({model}) ##########", flush=True)

        # 1. Stimulusliste mit korrektem Bildpfad
        stim_csv = build_matched_stimuli(
            subject, args.images_root,
            f"data/{model}_{subject}_matched_stimuli.csv",
        )

        # 2. Basis-Config (LOC dient nur als initiale human-RDM; wird pro ROI gepatcht)
        base_config = make_config(subject, model, stim_csv, output_root, "LOC")
        base_path = write_config(base_config, f"configs/tmp_{model}_{subject}_matched.json")

        # 3. Features + Geometrie (einmal pro Subject)
        run_step([python, "scripts/extract_resnet18_features.py", "--config", base_path])
        run_step([python, "scripts/compute_geometry.py", "--config", base_path])

        # 4. Pro ROI Spearman + CKA
        subject_rows = []
        for roi in ROIS:
            roi_config = make_config(subject, model, stim_csv, output_root, roi)
            roi_path = write_config(roi_config, f"configs/tmp_{model}_{subject}_{roi}.json")
            run_step([
                python, "scripts/compare_geometry_to_human.py",
                "--config", roi_path,
                "--n-permutations", str(args.n_permutations),
            ])

            result_csv = (PROJECT_ROOT / output_root / "results"
                          / f"ann_{model}_{subject}_matched_geometric_alignment.csv")
            for row in csv.DictReader(result_csv.open(encoding="utf-8")):
                row["subject"] = subject
                row["roi"] = roi
                row["p_value"] = row.get("p_value", row.get("spearman_p", ""))
                subject_rows.append(row)
                all_summary.append(dict(row))

        # 5. Subject-Summary schreiben
        out = PROJECT_ROOT / output_root / "results" / f"ann_{model}_{subject}_roi_summary.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["subject", "roi", "layer", "metric", "spearman_r", "p_value", "cka"]
        with out.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(subject_rows)
        print(f"Subject-Summary -> {out}", flush=True)

    # 6. Gesamt-Summary
    combined = PROJECT_ROOT / output_root / "results" / f"ann_{model}_all_subjects_roi_summary.csv"
    fieldnames = ["subject", "roi", "layer", "metric", "spearman_r", "p_value", "cka"]
    with combined.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_summary)
    print(f"\nGesamt-Summary -> {combined}  ({len(all_summary)} Zeilen)", flush=True)


if __name__ == "__main__":
    main()

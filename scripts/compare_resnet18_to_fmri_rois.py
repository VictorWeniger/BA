import argparse
import csv
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROIS = ["V1", "hV4", "LOC", "IT"]


def run_step(command):
    """Subkommando sichtbar ausführen.

    Der Orchestrator ruft mehrere Einzelskripte nacheinander auf. Durch das
    explizite Ausgeben des Befehls bleibt im Terminal nachvollziehbar, welcher
    Pipeline-Schritt gerade läuft.
    """
    print("\n" + "=" * 80, flush=True)
    print(" ".join(command), flush=True)
    print("=" * 80, flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def patch_config_for_roi(template_path, output_path, roi, subject):
    """Config-Datei auf eine konkrete Subject×ROI-fMRI-RDM umbiegen.

    Die ANN-Features und ANN-RDMs sind pro Subject nur einmal nötig. Für den
    Vergleich gegen V1/hV4/LOC/IT wird lediglich der Pfad zur fMRI-RDM in der
    Config ausgetauscht.
    """
    import json

    with open(PROJECT_ROOT / template_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    config["human"]["distance_matrix"] = f"outputs/human/processed_fmri_{subject}_{roi}_distance.npy"
    config["human"]["source"] = f"processed_fmri_{subject}_{roi}"

    output_path = PROJECT_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", default="S1")
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()

    python = args.python

    # Eine ANN-Stimulusliste pro Subject reicht aus. LOC dient nur als
    # Beispielquelle für die Stimulusreihenfolge; alle ROIs verwenden denselben
    # THINGS-Teststimulus-Satz und dieselbe Aggregation.
    run_step(
        [
            python,
            "scripts/build_ann_stimuli_from_fmri.py",
            "--fmri-stimuli",
            f"data/processed_fmri_{args.subject}_LOC_stimuli.csv",
            "--output",
            f"data/ann_{args.subject}_matched_stimuli.csv",
        ]
    )

    # Temporäre Config erzeugen. Sie enthält zunächst einen Human/fMRI-Pfad,
    # wird aber danach auf subject-weite ANN-Inputs gepatcht.
    template = "configs/ann_from_processed_s1_loc.json"
    base_config = f"configs/tmp_ann_resnet18_{args.subject}_matched.json"
    patch_config_for_roi(template, base_config, "LOC", args.subject)

    # Experimentname und Stimuluspfad so ändern, dass Outputs pro Subject
    # eindeutig heißen, z.B. ann_resnet18_S2_matched_layer3.npy.
    import json

    with open(PROJECT_ROOT / base_config, "r", encoding="utf-8") as f:
        config = json.load(f)
    config["experiment_name"] = f"ann_resnet18_{args.subject}_matched"
    config["data"]["stimuli_csv"] = f"data/ann_{args.subject}_matched_stimuli.csv"
    with open(PROJECT_ROOT / base_config, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    # ANN-Seite: Bilder durch ResNet schicken und daraus ANN-RDMs berechnen.
    # Diese Schritte passieren nur einmal pro Subject, nicht einmal pro ROI.
    run_step([python, "scripts/extract_resnet18_features.py", "--config", base_config])
    run_step([python, "scripts/compute_geometry.py", "--config", base_config])

    summary_rows = []
    for roi in ROIS:
        # fMRI-Seite: Für jede ROI wird dieselbe ANN-RDM gegen eine andere
        # fMRI-RDM verglichen.
        roi_config = f"configs/tmp_ann_resnet18_{args.subject}_{roi}.json"
        patch_config_for_roi(base_config, roi_config, roi, args.subject)
        run_step([python, "scripts/compare_geometry_to_human.py", "--config", roi_config])

        result_csv = PROJECT_ROOT / "outputs/results" / f"ann_resnet18_{args.subject}_matched_geometric_alignment.csv"
        # The compare script overwrites the same file for each ROI config, so copy
        # rows into a combined subject x ROI summary immediately.
        with open(result_csv, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                row["subject"] = args.subject
                row["roi"] = roi
                row["p_value"] = row.get("p_value", row.get("spearman_p", ""))
                summary_rows.append(row)

    output = PROJECT_ROOT / "outputs/results" / f"ann_resnet18_{args.subject}_roi_summary.csv"
    with open(output, "w", encoding="utf-8", newline="") as f:
        fieldnames = ["subject", "roi", "layer", "metric", "spearman_r", "p_value", "cka"]
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\nSaved ROI summary -> {output}", flush=True)


if __name__ == "__main__":
    main()

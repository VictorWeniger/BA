import csv
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROIS = ["V1", "hV4", "LOC", "IT"]


def patch_config(base_config, output_config, roi):
    with open(PROJECT_ROOT / base_config, "r", encoding="utf-8") as f:
        config = json.load(f)

    config["human"]["distance_matrix"] = f"outputs/human/processed_fmri_S1_{roi}_distance.npy"
    config["human"]["source"] = f"processed_fmri_S1_{roi}"

    with open(PROJECT_ROOT / output_config, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def main():
    python = sys.executable
    base_config = "configs/tmp_ann_resnet18_S1_matched.json"
    summary_rows = []

    for roi in ROIS:
        roi_config = f"configs/tmp_ann_resnet18_S1_{roi}.json"
        patch_config(base_config, roi_config, roi)
        subprocess.run(
            [python, "scripts/compare_geometry_to_human.py", "--config", roi_config],
            cwd=PROJECT_ROOT,
            check=True,
        )

        result_csv = PROJECT_ROOT / "outputs/results/ann_resnet18_S1_matched_geometric_alignment.csv"
        with open(result_csv, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                row["subject"] = "S1"
                row["roi"] = roi
                row["p_value"] = row.get("p_value", row.get("spearman_p", ""))
                summary_rows.append(row)

    output = PROJECT_ROOT / "outputs/results/ann_resnet18_S1_roi_summary.csv"
    with open(output, "w", encoding="utf-8", newline="") as f:
        fieldnames = ["subject", "roi", "layer", "metric", "spearman_r", "p_value", "cka"]
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Saved {output}")


if __name__ == "__main__":
    main()

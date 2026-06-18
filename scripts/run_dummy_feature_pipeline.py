import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_step(command):
    print("\n" + "=" * 80)
    print(" ".join(command))
    print("=" * 80)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dummy_resnet18.json")
    args = parser.parse_args()

    python = sys.executable

    run_step([python, "scripts/generate_dummy_stimuli.py"])
    run_step([python, "scripts/generate_dummy_features.py", "--config", args.config])
    run_step([python, "scripts/generate_dummy_human_matrix.py", "--config", args.config])
    run_step([python, "scripts/visualize_features.py", "--config", args.config])
    run_step([python, "scripts/compute_geometry.py", "--config", args.config])
    run_step([python, "scripts/compute_tda.py", "--config", args.config])
    run_step([python, "scripts/compare_geometry_to_human.py", "--config", args.config])
    run_step([python, "scripts/compare_tda_to_human.py", "--config", args.config])

    print("\nDummy feature pipeline finished.")


if __name__ == "__main__":
    main()

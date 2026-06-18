import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROIS = ["V1", "hV4", "LOC", "IT"]


def run_step(command):
    print("\n" + "=" * 80, flush=True)
    print(" ".join(command), flush=True)
    print("=" * 80, flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", default="S1")
    parser.add_argument("--max-trials", type=int, default=None)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()

    python = args.python

    for roi in ROIS:
        experiment_name = f"processed_fmri_{args.subject}_{roi}"
        cmd = [
            python,
            "scripts/import_processed_fmri.py",
            "--config",
            "configs/processed_fmri_s1_loc.json",
            "--subject",
            args.subject,
            "--roi",
            roi,
            "--experiment-name",
            experiment_name,
        ]
        if args.max_trials is not None:
            cmd += ["--max-trials", str(args.max_trials)]
        run_step(cmd)

    print("\nS1 ROI batch finished.", flush=True)


if __name__ == "__main__":
    main()

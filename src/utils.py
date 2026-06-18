import csv
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_path(path):
    """Resolve a path relative to the project root unless it is already absolute."""

    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_config(config_path):
    """Load a JSON config file."""

    with open(project_path(config_path), "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path):
    """Create a directory if needed and return it as Path."""

    path = project_path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_stimuli_csv(csv_path):
    """Read the stimuli table as a list of dictionaries."""

    with open(project_path(csv_path), "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(rows, output_path, fieldnames):
    """Write rows to a CSV file."""

    output_path = project_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_text(text, output_path):
    """Write text to a UTF-8 file."""

    output_path = project_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")

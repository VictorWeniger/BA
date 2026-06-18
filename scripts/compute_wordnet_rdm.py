"""
Build a WordNet-based distance matrix over THINGS concepts.

For each pair of concepts, Wu-Palmer similarity (wup) is computed via NLTK WordNet.
Distance = 1 - similarity. Concepts with no WordNet synset get similarity 0.

Requires: nltk, nltk corpus 'wordnet' and 'omw-1.4'
  pip install nltk
  python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.utils import ensure_dir, load_config, read_stimuli_csv


def get_synset(concept):
    """Return the first noun synset for a concept name, or None."""
    from nltk.corpus import wordnet as wn
    concept_clean = concept.lower().replace(" ", "_")
    synsets = wn.synsets(concept_clean, pos=wn.NOUN)
    if not synsets:
        synsets = wn.synsets(concept_clean)
    return synsets[0] if synsets else None


def wup_similarity(s1, s2):
    """Wu-Palmer similarity between two synsets, returns 0.0 if undefined."""
    if s1 is None or s2 is None:
        return 0.0
    sim = s1.wup_similarity(s2)
    return float(sim) if sim is not None else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/resnet50.json")
    args = parser.parse_args()

    try:
        import nltk
        from nltk.corpus import wordnet  # noqa: F401
    except ImportError:
        print("ERROR: nltk not installed. Run: pip install nltk")
        sys.exit(1)

    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        print("Downloading WordNet corpus...")
        nltk.download("wordnet")
        nltk.download("omw-1.4")

    config = load_config(args.config)
    wn_config = config.get("wordnet", {})
    out_dir = ensure_dir(wn_config.get("output_dir", "outputs/human"))

    rows = read_stimuli_csv(config["data"]["stimuli_csv"])
    concept_col = config["data"]["concept_column"]
    concepts = [row[concept_col] for row in rows]
    n = len(concepts)

    print(f"Computing WordNet RDM for {n} concepts...")
    synsets = [get_synset(c) for c in concepts]
    missing = sum(1 for s in synsets if s is None)
    if missing:
        print(f"  Warning: {missing}/{n} concepts have no WordNet synset (similarity set to 0)")

    sim_matrix = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        sim_matrix[i, i] = 1.0
        for j in range(i + 1, n):
            sim = wup_similarity(synsets[i], synsets[j])
            sim_matrix[i, j] = sim
            sim_matrix[j, i] = sim

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{n}")

    distance_matrix = 1.0 - sim_matrix

    out_path = out_dir / "wordnet_distance.npy"
    np.save(out_path, distance_matrix)
    print(f"Saved WordNet RDM {distance_matrix.shape} -> {out_path}")

    concepts_path = out_dir / "wordnet_concepts.txt"
    concepts_path.write_text("\n".join(concepts), encoding="utf-8")
    print(f"Saved concept order -> {concepts_path}")


if __name__ == "__main__":
    main()

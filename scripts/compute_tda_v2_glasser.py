"""Areale TDA (Whole-Brain): ResNet-50-Schichten vs. Glasser-Areale.

Whole-Brain-Variante von ``compute_tda_resnet50_fmri.py``. Statt vier ROIs werden
alle 180 oberflaechen-definierten Glasser/HCP-Areale topologisch mit den sechs
ResNet-50-Schichten verglichen. Die arealen fMRI-RDMs liegen bereits aus der
v2-Glasser-Pipeline vor (outputs_resnet50_v2_surface/glasser/), die ANN-Cosine-RDMs
aus dem ResNet-50-Lauf (outputs_resnet50/geometry/).

Methodik wie in v1: RDM auf [0,1] normieren, Vietoris-Rips-Persistenzdiagramm via
Ripser (maxdim=2 -> H0, H1, H2), 1-Wasserstein-Distanz zwischen Areal- und Schicht-
Diagrammen via persim. Beste Schicht je Areal = minimale Wasserstein-Distanz.
H2 (Hohlraeume) ist bei nur 100 Stimuli wenig belastbar und wird nur zur
Vollstaendigkeit mitberichtet.

  python scripts/compute_tda_v2_glasser.py
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from persim import wasserstein
from ripser import ripser

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LAYERS = ["conv1", "layer1", "layer2", "layer3", "layer4", "fc"]


def normalize_rdm(rdm):
    rdm = rdm.copy()
    np.fill_diagonal(rdm, 0.0)
    rdm[rdm < 0] = 0.0  # numerische Mini-Negativwerte abfangen
    vmax = rdm.max()
    if vmax > 0:
        rdm = rdm / vmax
    return rdm


def diagram_from_rdm(rdm, maxdim):
    return ripser(normalize_rdm(rdm), distance_matrix=True, maxdim=maxdim)["dgms"]


def wasserstein_safe(a, b):
    if len(a) == 0 and len(b) == 0:
        return 0.0
    return float(wasserstein(a, b))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="resnet50")
    parser.add_argument("--subjects", nargs="+", default=["S1", "S2", "S3"])
    parser.add_argument("--maxdim", type=int, default=2)
    parser.add_argument("--geometry-dir", default="outputs_resnet50/geometry")
    parser.add_argument("--glasser-dir",
                        default="outputs_resnet50_v2_surface/glasser")
    parser.add_argument("--results-dir",
                        default="outputs_resnet50_v2_surface/results")
    args = parser.parse_args()

    geom_dir = PROJECT_ROOT / args.geometry_dir
    glasser_dir = PROJECT_ROOT / args.glasser_dir
    results_dir = PROJECT_ROOT / args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    model = args.model

    # Vertexzahlen pro Areal/Proband aus der oberflaechenbasierten Zusammenfassung
    nvert = {}
    summ = results_dir / "glasser_surface_summary.csv"
    if summ.exists():
        with summ.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                nvert[(r["subject"], r["area"])] = int(r["n_vertices"])

    all_rows = []
    for subject in args.subjects:
        print(f"\n=== {subject} ===")
        ann_diagrams = {}
        for layer in LAYERS:
            rdm_file = geom_dir / f"ann_{model}_{subject}_matched_{layer}_cosine.npy"
            if not rdm_file.exists():
                print(f"  FEHLER: {rdm_file} fehlt")
                continue
            ann_diagrams[layer] = diagram_from_rdm(np.load(rdm_file), args.maxdim)
        print(f"  ANN-Diagramme: {len(ann_diagrams)}/{len(LAYERS)}")

        area_files = sorted(glasser_dir.glob(f"{subject}_glasser-*_rdm.npy"))
        print(f"  Areale: {len(area_files)}")
        for af in area_files:
            area = af.name[len(f"{subject}_glasser-"):-len("_rdm.npy")]
            fmri_dgms = diagram_from_rdm(np.load(af), args.maxdim)
            for layer in LAYERS:
                if layer not in ann_diagrams:
                    continue
                for hdim in range(args.maxdim + 1):
                    dist = wasserstein_safe(ann_diagrams[layer][hdim],
                                            fmri_dgms[hdim])
                    all_rows.append({
                        "subject": subject, "area": area,
                        "n_vertices": nvert.get((subject, area), -1),
                        "layer": layer, "homology_dim": hdim,
                        "wasserstein_dist": round(dist, 6),
                    })

    out_csv = results_dir / f"tda_{model}_glasser_all_subjects.csv"
    fields = ["subject", "area", "n_vertices", "layer", "homology_dim",
              "wasserstein_dist"]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nGespeichert: {out_csv}  ({len(all_rows)} Zeilen)")

    # Aggregation: pro Areal Mittel ueber Probanden, beste Schicht je Hdim
    # key (area, layer, hdim) -> list of dists
    agg = defaultdict(list)
    areas = []
    for r in all_rows:
        agg[(r["area"], r["layer"], r["homology_dim"])].append(r["wasserstein_dist"])
        if r["area"] not in areas:
            areas.append(r["area"])

    best_csv = results_dir / f"tda_{model}_glasser_bestlayer.csv"
    dist_counter = {h: defaultdict(int) for h in range(args.maxdim + 1)}
    with best_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["area", "homology_dim", "best_layer", "min_wasserstein"]
                   + [f"W_{l}" for l in LAYERS])
        for area in areas:
            for hdim in range(args.maxdim + 1):
                means = []
                for layer in LAYERS:
                    vals = agg.get((area, layer, hdim), [])
                    means.append(np.mean(vals) if vals else np.nan)
                means = np.array(means, dtype=float)
                if np.all(np.isnan(means)):
                    continue
                bi = int(np.nanargmin(means))
                best = LAYERS[bi]
                dist_counter[hdim][best] += 1
                w.writerow([area, hdim, best, round(float(means[bi]), 6)]
                           + [round(float(m), 6) if not np.isnan(m) else ""
                              for m in means])
    print(f"Gespeichert: {best_csv}")

    for hdim in range(args.maxdim + 1):
        total = sum(dist_counter[hdim].values())
        print(f"\nBeste-Schicht-Verteilung H{hdim} (n={total} Areale):")
        for layer in LAYERS:
            print(f"  {layer:7s} {dist_counter[hdim][layer]:4d}")


if __name__ == "__main__":
    main()

"""Surface-Figuren fuer den v2-B Searchlight: Best-Layer-Karte + Peak-RSA auf
der Kortexoberflaeche (nilearn). Liest Santis Inflated-Surfaces READ-ONLY.
Ausgabe: outputs_resnet50_v2_surface/figures/*.png  und  summary CSV.

Aufruf:
  python figures_v2_surface.py [--k 50]   # k=50 Santis Hauptanalyse (Standard)
  python figures_v2_surface.py --k 100    # k=100 Robustheitspruefung
"""
import argparse
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from nibabel.freesurfer import io as fsio
from nilearn import plotting

PROJECT_ROOT = Path(__file__).resolve().parents[1]
THINGS = Path("/work/dldevel/galella/datasets/THINGS-fMRI")
SL = PROJECT_ROOT / "outputs_resnet50_v2_surface" / "searchlight"
FIG = PROJECT_ROOT / "outputs_resnet50_v2_surface" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
LAYERS = ["conv1", "layer1", "layer2", "layer3", "layer4", "fc"]
SUBJECTS = ["S1", "S2", "S3"]

ap = argparse.ArgumentParser()
ap.add_argument("--k", type=int, default=50,
                help="Searchlight-Groesse (50 = Santis Hauptanalyse, 100 = Robustheit)")
args = ap.parse_args()
tag = f"k{args.k}"


def mesh(subj, hemi):
    c, f = fsio.read_geometry(THINGS / "freesurfer" / subj / "surf" / f"{hemi}.inflated")
    return c, f


# Best-Layer-Karte S1 (lateral + ventral, beide Hemis)
subj = "S1"
for hemi in ["lh", "rh"]:
    best = np.load(SL / f"{subj}_{hemi}_bestlayer_{tag}.npy")
    coords, faces = mesh(subj, hemi)
    for view in ["lateral", "ventral"]:
        fig = plotting.plot_surf_roi(
            (coords, faces), roi_map=np.nan_to_num(best, nan=-1) + 1,
            hemi="left" if hemi == "lh" else "right", view=view,
            cmap="viridis", vmin=0, vmax=6, bg_on_data=True, darkness=0.5,
            title=f"{subj} {hemi} beste Schicht ({view}, {tag})")
        fig.savefig(FIG / f"surf_bestlayer_{subj}_{hemi}_{view}_{tag}.png", dpi=140, bbox_inches="tight")
        plt.close("all")
    # Peak-RSA
    peak = np.load(SL / f"{subj}_{hemi}_peakr_{tag}.npy")
    fig = plotting.plot_surf_stat_map(
        (coords, faces), stat_map=np.nan_to_num(peak), hemi="left" if hemi == "lh" else "right",
        view="lateral", cmap="hot", threshold=0.03, colorbar=True,
        title=f"{subj} {hemi} Peak-RSA ({tag})")
    fig.savefig(FIG / f"surf_peakr_{subj}_{hemi}_lateral_{tag}.png", dpi=140, bbox_inches="tight")
    plt.close("all")
print(f"Surface-Plots S1 fertig ({tag})")

# Summary: Verteilung beste Schicht ueber kortikale Vertices (peakr>0.05), je Proband
rows = []
for subj in SUBJECTS:
    for hemi in ["lh", "rh"]:
        try:
            best = np.load(SL / f"{subj}_{hemi}_bestlayer_{tag}.npy")
            peak = np.load(SL / f"{subj}_{hemi}_peakr_{tag}.npy")
        except FileNotFoundError:
            continue
        sig = np.isfinite(best) & (peak > 0.05)
        for li, layer in enumerate(LAYERS):
            rows.append({"subject": subj, "hemi": hemi, "layer": layer, "k": args.k,
                         "n_vertices": int(np.sum(best[sig] == li))})
out = (PROJECT_ROOT / "outputs_resnet50_v2_surface" / "results"
       / f"surface_bestlayer_distribution_{tag}.csv")
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["subject", "hemi", "layer", "k", "n_vertices"])
    w.writeheader()
    w.writerows(rows)
print("Summary:", out)

# Gesamtverteilung (alle Probanden, beide Hemis)
tot = {l: 0 for l in LAYERS}
for r in rows:
    tot[r["layer"]] += r["n_vertices"]
print(f"Best-Layer-Verteilung ({tag}) ueber signifikante Vertices:", tot)

"""
Erzeugt alle Flatmap-Abbildungen für die BA:
  RSA (6 Schichten + Peak + BestLayer)
  CKA (6 Schichten)
  TDA Wasserstein H0 (6 Schichten + BestLayer)
  TDA Wasserstein H1 (6 Schichten)

Aufruf:
  python3 generate_all_flatmaps.py [--k 50] [--subject S1]
"""
import argparse
import tempfile
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.colors import Normalize, BoundaryNorm
from matplotlib.cm import ScalarMappable
from pathlib import Path
from nibabel.freesurfer import io as fsio
from nilearn import datasets, plotting, surface
from scipy.spatial import cKDTree

ap = argparse.ArgumentParser()
ap.add_argument("--k",       type=int, default=50)
ap.add_argument("--subject",           default="S1")
args = ap.parse_args()

TAG  = f"k{args.k}"
LAYERS = ["conv1", "layer1", "layer2", "layer3", "layer4", "fc"]
subj   = args.subject

BASE    = Path(__file__).parent
SL_DIR  = BASE / "cluster_results_v2_surface" / "searchlight"
FS_DIR  = BASE / "ba_analyse_software" / "data" / "freesurfer_S1"
FIG_DIR = BASE / "figures_resnet50_v2"
FIG_DIR.mkdir(parents=True, exist_ok=True)

print("Lade fsaverage …")
fsa = datasets.fetch_surf_fsaverage(mesh="fsaverage")

# KD-Trees für lh und rh einmal bauen
trees = {}
for hemi in ["lh", "rh"]:
    s1_reg, _ = fsio.read_geometry(FS_DIR / f"{hemi}.sphere.reg")
    trees[hemi] = (s1_reg, cKDTree(s1_reg))

def resample(hemi, values):
    s1_reg, tree = trees[hemi]
    fsa_key  = "sphere_left" if hemi == "lh" else "sphere_right"
    fsa_sph  = surface.load_surf_mesh(fsa[fsa_key]).coordinates
    _, idx   = tree.query(fsa_sph)
    out = np.full(len(idx), np.nan)
    valid = np.isfinite(values)
    out[np.arange(len(idx))] = np.where(valid[idx], values[idx], np.nan)
    return out

def load(name):
    return np.load(SL_DIR / f"{subj}_{name}_{TAG}.npy")

def sulc(hemi):
    return surface.load_surf_data(
        fsa["sulc_left"] if hemi == "lh" else fsa["sulc_right"])

def flat_mesh(hemi):
    return fsa["flat_left"] if hemi == "lh" else fsa["flat_right"]


# ── Render-Hilfsfunktionen ────────────────────────────────────────────────────

def render_hemi(hemi, stat_fsa, vmin, vmax, cmap, tmp_dir, tag):
    fig = plotting.plot_surf_stat_map(
        flat_mesh(hemi),
        stat_map=np.nan_to_num(stat_fsa, nan=0.0),
        bg_map=sulc(hemi),
        hemi="left" if hemi == "lh" else "right",
        view="dorsal",
        cmap=cmap,
        vmin=vmin, vmax=vmax,
        colorbar=False,
        bg_on_data=True,
    )
    p = Path(tmp_dir) / f"{hemi}_{tag}.png"
    fig.savefig(str(p), dpi=150, bbox_inches="tight")
    plt.close("all")
    return p


def assemble(panels_lh, panels_rh, title, cmap, vmin, vmax, out_path,
             cbar_label, discrete_colors=None):
    """Baut lh + rh Panels zu einer Figure zusammen."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, hemi, p in zip(axes, ["lh", "rh"], [panels_lh, panels_rh]):
        ax.imshow(mpimg.imread(str(p)))
        ax.axis("off")
        ax.set_title("Linke Hemisphäre" if hemi == "lh" else "Rechte Hemisphäre",
                     fontsize=12, pad=6)

    if discrete_colors is not None:
        bounds = np.arange(-0.5, len(LAYERS))
        norm   = BoundaryNorm(bounds, len(LAYERS))
        sm     = ScalarMappable(cmap=matplotlib.colormaps[cmap], norm=norm)
        sm.set_array([])
        cbar   = fig.colorbar(sm, ax=axes, shrink=0.7, pad=0.015, aspect=22,
                               ticks=range(len(LAYERS)))
        cbar.set_ticklabels(LAYERS)
    else:
        sm   = ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax),
                              cmap=matplotlib.colormaps[cmap])
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes, shrink=0.7, pad=0.015, aspect=25)

    cbar.set_label(cbar_label, fontsize=11)
    cbar.ax.tick_params(labelsize=9)
    fig.suptitle(title, fontsize=13, y=1.01)
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()
    print("  →", out_path.name)


# ── Alle Karten erzeugen ──────────────────────────────────────────────────────

with tempfile.TemporaryDirectory() as tmp:

    # ── RSA (6 Schichten) ─────────────────────────────────────────────────────
    print("\n=== RSA (6 Schichten) ===")
    rsa_data = {h: load(f"{h}_rsa_perlayer") for h in ["lh","rh"]}
    all_rsa  = np.concatenate([rsa_data["lh"].ravel(), rsa_data["rh"].ravel()])
    rsa_abs  = np.nanpercentile(np.abs(all_rsa[np.isfinite(all_rsa)]), 95)

    for li, layer in enumerate(LAYERS):
        panels = {}
        for hemi in ["lh","rh"]:
            v = resample(hemi, rsa_data[hemi][:, li])
            panels[hemi] = render_hemi(hemi, v, -rsa_abs, rsa_abs,
                                       "RdBu_r", tmp, f"rsa_{layer}_{hemi}")
        assemble(panels["lh"], panels["rh"],
                 f"RSA Spearman $r$ — Schicht {layer}  ({subj}, {TAG})",
                 "RdBu_r", -rsa_abs, rsa_abs,
                 FIG_DIR / f"fig_flatmap_rsa_{layer}_{TAG}_{subj}.pdf",
                 "Spearman $r$")

    # ── Peak-RSA ──────────────────────────────────────────────────────────────
    print("\n=== Peak-RSA ===")
    peak_data = {h: load(f"{h}_peakr") for h in ["lh","rh"]}
    all_peak  = np.concatenate([peak_data["lh"], peak_data["rh"]])
    peak_max  = float(np.nanpercentile(all_peak[np.isfinite(all_peak)], 95))
    panels = {}
    for hemi in ["lh","rh"]:
        v = resample(hemi, peak_data[hemi])
        panels[hemi] = render_hemi(hemi, v, 0, peak_max, "hot", tmp, f"peakr_{hemi}")
    assemble(panels["lh"], panels["rh"],
             f"Peak-RSA (max Spearman $r$ über Schichten)  ({subj}, {TAG})",
             "hot", 0, peak_max,
             FIG_DIR / f"fig_flatmap_peakr_{TAG}_{subj}.pdf",
             "Peak Spearman $r$")

    # ── RSA Best-Layer ────────────────────────────────────────────────────────
    print("\n=== RSA Best-Layer ===")
    best_data = {h: load(f"{h}_bestlayer") for h in ["lh","rh"]}
    panels = {}
    for hemi in ["lh","rh"]:
        v = resample(hemi, best_data[hemi])
        panels[hemi] = render_hemi(hemi, v, 0, 5, "viridis", tmp, f"bestlayer_{hemi}")
    assemble(panels["lh"], panels["rh"],
             f"Beste RSA-Schicht je Vertex  ({subj}, {TAG})",
             "viridis", 0, 5,
             FIG_DIR / f"fig_flatmap_rsa_bestlayer_{TAG}_{subj}.pdf",
             "Beste Schicht",
             discrete_colors=True)

    # ── CKA (6 Schichten) ─────────────────────────────────────────────────────
    print("\n=== CKA (6 Schichten) ===")
    cka_data = {h: load(f"{h}_cka_perlayer") for h in ["lh","rh"]}
    all_cka  = np.concatenate([cka_data["lh"].ravel(), cka_data["rh"].ravel()])
    cka_max  = float(np.nanpercentile(all_cka[np.isfinite(all_cka)], 95))

    for li, layer in enumerate(LAYERS):
        panels = {}
        for hemi in ["lh","rh"]:
            v = resample(hemi, cka_data[hemi][:, li])
            panels[hemi] = render_hemi(hemi, v, 0, cka_max,
                                       "YlOrRd", tmp, f"cka_{layer}_{hemi}")
        assemble(panels["lh"], panels["rh"],
                 f"CKA — Schicht {layer}  ({subj}, {TAG})",
                 "YlOrRd", 0, cka_max,
                 FIG_DIR / f"fig_flatmap_cka_{layer}_{TAG}_{subj}.pdf",
                 "CKA")

    # ── TDA Wasserstein H0 + H1 (je 6 Schichten) ─────────────────────────────
    for hdim in [0, 1]:
        print(f"\n=== TDA Wasserstein H{hdim} (6 Schichten) ===")
        wd_data = {h: load(f"{h}_tda_wd_H{hdim}_perlayer") for h in ["lh","rh"]}
        all_wd  = np.concatenate([wd_data["lh"].ravel(), wd_data["rh"].ravel()])
        wd_min  = float(np.nanpercentile(all_wd[np.isfinite(all_wd)],  5))
        wd_max  = float(np.nanpercentile(all_wd[np.isfinite(all_wd)], 95))

        for li, layer in enumerate(LAYERS):
            panels = {}
            for hemi in ["lh","rh"]:
                v = resample(hemi, wd_data[hemi][:, li])
                panels[hemi] = render_hemi(hemi, v, wd_min, wd_max,
                                           "plasma", tmp,
                                           f"wd_H{hdim}_{layer}_{hemi}")
            assemble(panels["lh"], panels["rh"],
                     f"TDA Wasserstein-Distanz $H_{hdim}$ — Schicht {layer}  ({subj}, {TAG})",
                     "plasma", wd_min, wd_max,
                     FIG_DIR / f"fig_flatmap_tda_H{hdim}_{layer}_{TAG}_{subj}.pdf",
                     f"Wasserstein-Distanz $H_{hdim}$")

    # ── TDA Best-Layer ────────────────────────────────────────────────────────
    print("\n=== TDA Best-Layer ===")
    tda_best = {h: load(f"{h}_tda_bestlayer") for h in ["lh","rh"]}
    panels = {}
    for hemi in ["lh","rh"]:
        v = resample(hemi, tda_best[hemi])
        panels[hemi] = render_hemi(hemi, v, 0, 5, "viridis", tmp,
                                   f"tda_bestlayer_{hemi}")
    assemble(panels["lh"], panels["rh"],
             f"Beste TDA-Schicht je Vertex (min WD)  ({subj}, {TAG})",
             "viridis", 0, 5,
             FIG_DIR / f"fig_flatmap_tda_bestlayer_{TAG}_{subj}.pdf",
             "Beste Schicht (min Wasserstein)",
             discrete_colors=True)

print("\nFertig. Alle Abbildungen in:", FIG_DIR)

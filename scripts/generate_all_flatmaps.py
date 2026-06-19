"""
Alle kortikalen Flatmaps: RSA, CKA, TDA H0/H1, Best-Layer-Karten.
Direkte tripcolor-Darstellung auf den fsaverage-Flat-Koordinaten (kein nilearn-Bug).
lh: 270° gedreht; rh: 270° gedreht + horizontal gespiegelt → Medialwände innen.

Aufruf:
  python3 generate_all_flatmaps.py [--k 50] [--subject S1]
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import Normalize, BoundaryNorm
from matplotlib.cm import ScalarMappable
from pathlib import Path
from nibabel.freesurfer import io as fsio
from nilearn import datasets, surface
from scipy.spatial import cKDTree

ap = argparse.ArgumentParser()
ap.add_argument("--k",      type=int, default=50)
ap.add_argument("--subject",          default="S1")
args = ap.parse_args()

TAG    = f"k{args.k}"
LAYERS = ["conv1", "layer1", "layer2", "layer3", "layer4", "fc"]
subj   = args.subject

BASE    = Path(__file__).parent
SL_DIR  = BASE / "cluster_results_v2_surface" / "searchlight"
FS_DIR  = BASE / "ba_analyse_software" / "data" / "freesurfer_S1"
FIG_DIR = BASE / "figures_resnet50_v2"
FIG_DIR.mkdir(parents=True, exist_ok=True)

print("Lade fsaverage …")
fsa = datasets.fetch_surf_fsaverage(mesh="fsaverage")

# ── Flat-Surface-Geometrie mit Rotation aufbauen ──────────────────────────────

ROT_LH =   0    # lh: keine Rotation
ROT_RH =   0    # rh: keine Rotation (beide 90° zueinander gedreht ggü. ±90°)

def rotate2d(xy, deg):
    r = np.radians(deg)
    R = np.array([[np.cos(r), -np.sin(r)],
                  [np.sin(r),  np.cos(r)]])
    return (R @ xy.T).T

def build_surface(hemi):
    key   = "flat_left" if hemi == "lh" else "flat_right"
    mesh  = surface.load_surf_mesh(fsa[key])
    c     = mesh.coordinates
    f     = mesh.faces
    sulc  = surface.load_surf_data(
        fsa["sulc_left"] if hemi == "lh" else fsa["sulc_right"])

    rot = ROT_LH if hemi == "lh" else ROT_RH
    xy  = rotate2d(c[:, :2], rot)

    # Nahtdreiecke entfernen
    xf   = xy[f, 0]
    keep = np.abs(xf.max(axis=1) - xf.min(axis=1)) < 100
    tri  = mtri.Triangulation(xy[:, 0], xy[:, 1], f[keep])
    return tri, sulc

print("Baue Dreiecksgitter …")
surfs = {h: build_surface(h) for h in ["lh", "rh"]}

# ── Resample S1 → fsaverage ───────────────────────────────────────────────────

trees = {}
for hemi in ["lh", "rh"]:
    s1_reg, _ = fsio.read_geometry(FS_DIR / f"{hemi}.sphere.reg")
    trees[hemi] = cKDTree(s1_reg)

def resample(hemi, values):
    fsa_key = "sphere_left" if hemi == "lh" else "sphere_right"
    fsa_sph = surface.load_surf_mesh(fsa[fsa_key]).coordinates
    _, idx  = trees[hemi].query(fsa_sph)
    out = np.where(np.isfinite(values[idx]), values[idx], np.nan)
    return out

def load(name):
    return np.load(SL_DIR / f"{subj}_{name}_{TAG}.npy")


# ── Zeichenroutinen ───────────────────────────────────────────────────────────

def draw_hemi(ax, hemi, values_fsa, cmap, norm):
    tri, sulc = surfs[hemi]
    # Hintergrund: sulcal depth
    ax.tripcolor(tri, np.clip(sulc, -5, 5), cmap="gray",
                 shading="gouraud", vmin=-5, vmax=5, rasterized=True)
    # Vordergrund: Metrik
    vis = np.where(np.isfinite(values_fsa), values_fsa,
                   np.nanmean(values_fsa[np.isfinite(values_fsa)]))
    ax.tripcolor(tri, vis, cmap=cmap, norm=norm,
                 shading="gouraud", alpha=0.85, rasterized=True)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Linke Hemisphäre" if hemi == "lh" else "Rechte Hemisphäre",
                 fontsize=12, pad=6)


def save_fig(title, cmap, norm, out_path, cbar_label,
             vals_lh, vals_rh, discrete_ticks=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    draw_hemi(axes[0], "lh", vals_lh, cmap, norm)
    draw_hemi(axes[1], "rh", vals_rh, cmap, norm)

    sm = ScalarMappable(cmap=matplotlib.colormaps[cmap], norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, shrink=0.72, pad=0.015, aspect=25)
    if discrete_ticks is not None:
        cbar.set_ticks(range(len(LAYERS)))
        cbar.set_ticklabels(LAYERS)
    cbar.set_label(cbar_label, fontsize=11)
    cbar.ax.tick_params(labelsize=9)
    fig.suptitle(title, fontsize=13, y=1.01)
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()
    print("  →", out_path.name)


# ── RSA (6 Schichten) ─────────────────────────────────────────────────────────
print("\n=== RSA ===")
rsa_raw = {h: load(f"{h}_rsa_perlayer") for h in ["lh", "rh"]}
all_rsa = np.concatenate([rsa_raw["lh"].ravel(), rsa_raw["rh"].ravel()])
rsa_abs = float(np.nanpercentile(np.abs(all_rsa[np.isfinite(all_rsa)]), 95))

for li, layer in enumerate(LAYERS):
    save_fig(
        f"RSA Spearman $r$ — Schicht {layer}  ({subj}, {TAG})",
        "RdBu_r", Normalize(-rsa_abs, rsa_abs),
        FIG_DIR / f"fig_flatmap_rsa_{layer}_{TAG}_{subj}.pdf",
        "Spearman $r$",
        resample("lh", rsa_raw["lh"][:, li]),
        resample("rh", rsa_raw["rh"][:, li]),
    )

# ── Peak-RSA ──────────────────────────────────────────────────────────────────
print("\n=== Peak-RSA ===")
peak_raw = {h: load(f"{h}_peakr") for h in ["lh", "rh"]}
all_peak = np.concatenate([peak_raw["lh"], peak_raw["rh"]])
peak_max = float(np.nanpercentile(all_peak[np.isfinite(all_peak)], 95))
save_fig(
    f"Peak-RSA (max Spearman $r$ über Schichten)  ({subj}, {TAG})",
    "hot", Normalize(0, peak_max),
    FIG_DIR / f"fig_flatmap_peakr_{TAG}_{subj}.pdf",
    "Peak Spearman $r$",
    resample("lh", peak_raw["lh"]),
    resample("rh", peak_raw["rh"]),
)

# ── RSA Best-Layer ────────────────────────────────────────────────────────────
print("\n=== RSA Best-Layer ===")
best_raw = {h: load(f"{h}_bestlayer") for h in ["lh", "rh"]}
save_fig(
    f"Beste RSA-Schicht je Vertex  ({subj}, {TAG})",
    "viridis", BoundaryNorm(np.arange(-0.5, len(LAYERS)), len(LAYERS)),
    FIG_DIR / f"fig_flatmap_rsa_bestlayer_{TAG}_{subj}.pdf",
    "Beste Schicht",
    resample("lh", best_raw["lh"]),
    resample("rh", best_raw["rh"]),
    discrete_ticks=True,
)

# ── CKA (6 Schichten) ─────────────────────────────────────────────────────────
print("\n=== CKA ===")
cka_raw = {h: load(f"{h}_cka_perlayer") for h in ["lh", "rh"]}
all_cka = np.concatenate([cka_raw["lh"].ravel(), cka_raw["rh"].ravel()])
cka_max = float(np.nanpercentile(all_cka[np.isfinite(all_cka)], 95))

for li, layer in enumerate(LAYERS):
    save_fig(
        f"CKA — Schicht {layer}  ({subj}, {TAG})",
        "YlOrRd", Normalize(0, cka_max),
        FIG_DIR / f"fig_flatmap_cka_{layer}_{TAG}_{subj}.pdf",
        "CKA",
        resample("lh", cka_raw["lh"][:, li]),
        resample("rh", cka_raw["rh"][:, li]),
    )

# ── TDA H0 + H1 (je 6 Schichten) ─────────────────────────────────────────────
for hdim in [0, 1]:
    print(f"\n=== TDA H{hdim} ===")
    wd_raw = {h: load(f"{h}_tda_wd_H{hdim}_perlayer") for h in ["lh", "rh"]}
    all_wd = np.concatenate([wd_raw["lh"].ravel(), wd_raw["rh"].ravel()])
    wd_min = float(np.nanpercentile(all_wd[np.isfinite(all_wd)],  5))
    wd_max = float(np.nanpercentile(all_wd[np.isfinite(all_wd)], 95))

    for li, layer in enumerate(LAYERS):
        save_fig(
            f"TDA Wasserstein-Distanz $H_{hdim}$ — Schicht {layer}  ({subj}, {TAG})",
            "plasma", Normalize(wd_min, wd_max),
            FIG_DIR / f"fig_flatmap_tda_H{hdim}_{layer}_{TAG}_{subj}.pdf",
            f"Wasserstein-Distanz $H_{hdim}$",
            resample("lh", wd_raw["lh"][:, li]),
            resample("rh", wd_raw["rh"][:, li]),
        )

# ── TDA Best-Layer ────────────────────────────────────────────────────────────
print("\n=== TDA Best-Layer ===")
tda_best = {h: load(f"{h}_tda_bestlayer") for h in ["lh", "rh"]}
save_fig(
    f"Beste TDA-Schicht je Vertex (min Wasserstein)  ({subj}, {TAG})",
    "viridis", BoundaryNorm(np.arange(-0.5, len(LAYERS)), len(LAYERS)),
    FIG_DIR / f"fig_flatmap_tda_bestlayer_{TAG}_{subj}.pdf",
    "Beste Schicht (min Wasserstein)",
    resample("lh", tda_best["lh"]),
    resample("rh", tda_best["rh"]),
    discrete_ticks=True,
)

print(f"\nFertig. Alle Abbildungen in: {FIG_DIR}")

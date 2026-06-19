"""
Echte Cortex-Flatmap der TDA Wasserstein-Distanz.
S1-Daten werden über sphere.reg auf fsaverage resamplet,
dann auf der fsaverage-Flatmap dargestellt (je 1 Bild pro Hemisphäre).

Aufruf:
  python3 generate_flatmap_proper.py [--k 50] [--hdim 0] [--layer layer4]
"""
import argparse
import tempfile
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from pathlib import Path
from nibabel.freesurfer import io as fsio
import nibabel as nib
from nilearn import datasets, plotting, surface
from scipy.spatial import cKDTree

ap = argparse.ArgumentParser()
ap.add_argument("--k",      type=int, default=50)
ap.add_argument("--subject",          default="S1")
ap.add_argument("--layer",            default="layer4")
ap.add_argument("--hdim", type=int,   default=0, choices=[0, 1])
args = ap.parse_args()

TAG    = f"k{args.k}"
LAYERS = ["conv1","layer1","layer2","layer3","layer4","fc"]
LI     = LAYERS.index(args.layer)
subj   = args.subject
hdim   = args.hdim

BASE    = Path(__file__).parent
SL_DIR  = BASE / "cluster_results_v2_surface" / "searchlight"
FS_DIR  = BASE / "ba_analyse_software" / "data" / "freesurfer_S1"
FIG_DIR = BASE / "figures_resnet50_v2"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# fsaverage Oberflächen holen (lokal gecacht nach erstem Download)
print("Lade fsaverage …")
fsa = datasets.fetch_surf_fsaverage(mesh="fsaverage")


# ── Resample S1 → fsaverage via sphere.reg ───────────────────────────────────

def resample_to_fsa(hemi, values):
    """
    Nächster-Nachbar-Interpolation von S1-Vertices auf fsaverage-Vertices
    über die Kugelregistrierung (sphere.reg).
    """
    s1_reg, _  = fsio.read_geometry(FS_DIR / f"{hemi}.sphere.reg")
    fsa_key    = "sphere_left" if hemi == "lh" else "sphere_right"
    fsa_sphere = surface.load_surf_mesh(fsa[fsa_key]).coordinates

    tree = cKDTree(s1_reg)
    _, idx = tree.query(fsa_sphere)
    return values[idx]


def load_wd(hemi):
    arr = np.load(SL_DIR / f"{subj}_{hemi}_tda_wd_H{hdim}_perlayer_{TAG}.npy")
    return arr[:, LI].astype(float)


# ── Farbskala ────────────────────────────────────────────────────────────────

all_vals = []
for hemi in ["lh", "rh"]:
    v = load_wd(hemi)
    all_vals.append(v[np.isfinite(v)])
all_vals = np.concatenate(all_vals)
vmin = float(np.percentile(all_vals, 5))
vmax = float(np.percentile(all_vals, 95))
print(f"Farbskala: {vmin:.2f} – {vmax:.2f}")


# ── Ein Panel rendern ─────────────────────────────────────────────────────────

def render_panel(hemi, wd_fsa, tmp_dir):
    flat_key  = "flat_left"  if hemi == "lh" else "flat_right"
    sulc_key  = "sulc_left"  if hemi == "lh" else "sulc_right"
    nl_hemi   = "left"       if hemi == "lh" else "right"

    flat_mesh = fsa[flat_key]
    sulc_data = surface.load_surf_data(fsa[sulc_key])

    # NaN → 0 für Rendering, threshold auf kleinen negativen Wert damit
    # auch Werte nahe 0 sichtbar bleiben
    stat = np.nan_to_num(wd_fsa, nan=0.0)

    fig = plotting.plot_surf_stat_map(
        flat_mesh,
        stat_map=stat,
        bg_map=sulc_data,
        hemi=nl_hemi,
        view="dorsal",          # bei Flatmap egal, dorsal gibt saubere Draufsicht
        cmap="plasma",
        vmin=vmin, vmax=vmax,
        colorbar=False,
        bg_on_data=True,
    )
    p = Path(tmp_dir) / f"{hemi}.png"
    fig.savefig(str(p), dpi=160, bbox_inches="tight")
    plt.close("all")
    return p


# ── Assembliieren ─────────────────────────────────────────────────────────────

with tempfile.TemporaryDirectory() as tmp:
    panels = {}
    for hemi in ["lh", "rh"]:
        print(f"  {hemi}: resamplen …")
        wd      = load_wd(hemi)
        wd_fsa  = resample_to_fsa(hemi, wd)
        print(f"  {hemi}: rendern …")
        panels[hemi] = render_panel(hemi, wd_fsa, tmp)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    titles = {"lh": "Linke Hemisphäre", "rh": "Rechte Hemisphäre"}
    for ax, hemi in zip(axes, ["lh", "rh"]):
        img = mpimg.imread(str(panels[hemi]))
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(titles[hemi], fontsize=13, pad=8)

    sm = ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax),
                        cmap=matplotlib.colormaps["plasma"])
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, shrink=0.7, pad=0.015, aspect=25)
    cbar.set_label(f"Wasserstein-Distanz $H_{hdim}$ (Schicht: {args.layer})",
                   fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    fig.suptitle(
        f"Kortikale Flatmap — TDA Wasserstein-Distanz $H_{hdim}$, "
        f"Schicht {args.layer}  ({subj}, {TAG})",
        fontsize=13, y=1.01
    )
    plt.tight_layout()

    out = FIG_DIR / f"fig_flatmap_proper_H{hdim}_{args.layer}_{TAG}_{subj}.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close()
    print("Gespeichert:", out)

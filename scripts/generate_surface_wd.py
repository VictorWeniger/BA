"""
Seitenansicht der TDA Wasserstein-Distanz auf der aufgeblasenen FreeSurfer-Oberfläche.
Stil analog zu den bestlayer/peakR-Karten (Abbildung 3.5).

Aufruf:
  python3 generate_surface_wd.py [--k 50] [--hdim 0] [--layer layer4]
  python3 generate_surface_wd.py --layer all   # alle 6 Schichten als Grid
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
from nilearn import plotting

ap = argparse.ArgumentParser()
ap.add_argument("--k",      type=int, default=50)
ap.add_argument("--subject",          default="S1")
ap.add_argument("--layer",            default="layer4")   # oder "all"
ap.add_argument("--hdim", type=int,   default=0, choices=[0,1])
args = ap.parse_args()

TAG    = f"k{args.k}"
LAYERS = ["conv1","layer1","layer2","layer3","layer4","fc"]
subj   = args.subject
hdim   = args.hdim

BASE    = Path(__file__).parent
SL_DIR  = BASE / "cluster_results_v2_surface" / "searchlight"
FS_DIR  = BASE / "ba_analyse_software" / "data" / "freesurfer_S1"
FIG_DIR = BASE / "figures_resnet50_v2"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def mesh(hemi):
    c, f = fsio.read_geometry(FS_DIR / f"{hemi}.inflated")
    return c, f

def bg_map(hemi):
    return fsio.read_morph_data(FS_DIR / f"{hemi}.sulc")

def load_wd(hemi, layer_idx):
    arr = np.load(SL_DIR / f"{subj}_{hemi}_tda_wd_H{hdim}_perlayer_{TAG}.npy")
    return arr[:, layer_idx].astype(float)


# ── Farbskala ────────────────────────────────────────────────────────────────

layer_list = LAYERS if args.layer == "all" else [args.layer]

all_vals = []
for hemi in ["lh", "rh"]:
    for lay in layer_list:
        v = load_wd(hemi, LAYERS.index(lay))
        all_vals.append(v[np.isfinite(v)])
all_vals = np.concatenate(all_vals)
vmin = float(np.percentile(all_vals, 5))
vmax = float(np.percentile(all_vals, 95))
print(f"Farbskala: {vmin:.2f} – {vmax:.2f}")


# ── Ein Panel als PNG (nilearn erzeugt eigene Figure) ────────────────────────

def render_panel(hemi, view, layer_idx, tmp_dir):
    coords, faces = mesh(hemi)
    wd  = load_wd(hemi, layer_idx)
    bg  = bg_map(hemi)
    nl_hemi = "left" if hemi == "lh" else "right"

    fig = plotting.plot_surf_stat_map(
        (coords, faces),
        stat_map=np.nan_to_num(wd, nan=0.0),
        bg_map=bg,
        hemi=nl_hemi,
        view=view,
        cmap="plasma",
        vmin=vmin, vmax=vmax,
        colorbar=False,
        bg_on_data=True,
    )
    p = Path(tmp_dir) / f"{hemi}_{view}_{layer_idx}.png"
    fig.savefig(str(p), dpi=130, bbox_inches="tight")
    plt.close("all")
    return p


# ── Grid zusammenbauen ───────────────────────────────────────────────────────

VIEWS = [
    ("lh", "lateral"),
    ("lh", "medial"),
    ("rh", "medial"),
    ("rh", "lateral"),
]
COL_LABELS = ["LH lateral", "LH medial", "RH medial", "RH lateral"]

n_rows = len(layer_list)
n_cols = len(VIEWS)

with tempfile.TemporaryDirectory() as tmp:
    # Alle Panels rendern
    panels = {}
    for ri, layer_name in enumerate(layer_list):
        li = LAYERS.index(layer_name)
        for ci, (hemi, view) in enumerate(VIEWS):
            print(f"  {layer_name} {hemi} {view} …")
            panels[(ri, ci)] = render_panel(hemi, view, li, tmp)

    # Grid-Figure aufbauen
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.2 * n_cols, 3.6 * n_rows + 0.7),
    )
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for ri, layer_name in enumerate(layer_list):
        for ci in range(n_cols):
            ax = axes[ri, ci]
            img = mpimg.imread(str(panels[(ri, ci)]))
            ax.imshow(img)
            ax.axis("off")
            if ri == 0:
                ax.set_title(COL_LABELS[ci], fontsize=11, pad=6)
            if ci == 0:
                ax.set_ylabel(layer_name, fontsize=11, rotation=90, labelpad=6)

    # Farblegende
    sm = ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax),
                        cmap=matplotlib.colormaps["plasma"])
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, shrink=0.7, pad=0.02, aspect=28)
    cbar.set_label(f"Wasserstein-Distanz $H_{hdim}$", fontsize=11)
    cbar.ax.tick_params(labelsize=9)

    tag_layer = "all" if args.layer == "all" else args.layer
    title = (f"TDA Wasserstein-Distanz $H_{hdim}$ — "
             f"{'alle Schichten' if args.layer == 'all' else args.layer}  "
             f"({subj}, {TAG})")
    fig.suptitle(title, fontsize=13, y=1.01)
    plt.tight_layout()

    out = FIG_DIR / f"fig_surf_wd_H{hdim}_{tag_layer}_{TAG}_{subj}.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close()
    print("Gespeichert:", out)

"""
Voxelweiser Within-Subject Noise Ceiling als pycortex-Flatmap.

Methode:
  1. Rohe Betas laden (ganzes Gehirn, mmap)
  2. Per-Session Z-Scoring (über 100 Stimuli pro Session)
  3. Leave-one-session-out Pearson r pro Voxel → NC_lower
  4. Spearman-Brown-Korrektur für 12 Sessionen → NC_12
  5. Erklärte Varianz = NC_12^2 * 100  [%]
  6. Pycortex-Flatmap für S1/S2/S3 nebeneinander

Verwendung:
  venv_cortex/bin/python scripts/generate_pycortex_nc_flatmap.py
"""

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cortex

PROCESSED_ROOT = Path("/Volumes/Sonstige Backups/Data/processed")
DB_ROOT        = Path("/Volumes/Sonstige Backups/Data/db")
PROJECT_ROOT   = Path(__file__).resolve().parents[1]
FIGURES_DIR    = PROJECT_ROOT / "outputs" / "figures"
RESULTS_DIR    = PROJECT_ROOT / "outputs" / "results"

SUBJECTS     = ["S1", "S2", "S3"]
N_SESSIONS   = 12
N_STIMULI    = 100
VMAX_PERCENT = 70   # colour scale max [%]


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------

def load_test_betas(subject):
    """
    Lädt alle Test-THINGS-Betas, sortiert nach Session × Stimulus.
    Gibt zurück:
      betas          (N_SESSIONS, N_STIMULI, n_vox)  float32
      canonical_order list[str]  – alphabetisch sortierte Stimulus-Namen
    """
    stim_path = PROCESSED_ROOT / subject / "stimuli" / f"{subject}_stimuli.csv"
    with open(stim_path, encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))

    test_entries = [
        (i, int(r["Session"]), r["Stimulus"])
        for i, r in enumerate(all_rows)
        if r["Split"] == "test" and r["Dataset"] == "THINGS"
    ]

    canonical_order = sorted(set(e[2] for e in test_entries))
    stim_to_idx = {s: i for i, s in enumerate(canonical_order)}
    unique_sessions = sorted(set(e[1] for e in test_entries))

    betas_path = PROCESSED_ROOT / subject / "betas" / "vol" / f"{subject}_betas_vol.npy"
    print(f"  {subject}: Loading betas (mmap)…", flush=True)
    betas_all = np.load(betas_path, mmap_mode="r")
    n_vox = betas_all.shape[1]

    result = np.zeros((N_SESSIONS, N_STIMULI, n_vox), dtype=np.float32)
    for trial_idx, sess, stim in test_entries:
        s_idx = unique_sessions.index(sess)
        c_idx = stim_to_idx[stim]
        result[s_idx, c_idx, :] = betas_all[trial_idx, :]

    return result, canonical_order


def per_session_zscore(betas):
    """Z-Score über Stimuli (axis=1) innerhalb jeder Session."""
    out = betas.copy()
    for s in range(betas.shape[0]):
        m = out[s].mean(axis=0, keepdims=True)
        sd = out[s].std(axis=0, keepdims=True)
        sd[sd == 0] = 1.0
        out[s] = (out[s] - m) / sd
    return out


def pearson_r_batch(x, y):
    """Pearson r zwischen x und y für jeden Voxel parallel.
    x, y: (N_STIMULI, n_vox)  → returns (n_vox,)
    """
    xc = x - x.mean(axis=0, keepdims=True)
    yc = y - y.mean(axis=0, keepdims=True)
    num   = (xc * yc).sum(axis=0)
    denom = np.sqrt((xc**2).sum(axis=0) * (yc**2).sum(axis=0))
    denom[denom == 0] = 1.0
    return num / denom


def compute_nc_voxelwise(betas_zscored):
    """
    Leave-one-session-out NC pro Voxel.
    betas_zscored: (12, 100, n_vox)
    Gibt NC_lower (n_vox,) zurück.
    """
    n_sess, n_stim, n_vox = betas_zscored.shape
    mean_all = betas_zscored.mean(axis=0)       # (100, n_vox)
    nc_lower = np.zeros(n_vox, dtype=np.float64)

    for s in range(n_sess):
        sess = betas_zscored[s]                 # (100, n_vox)
        mean_others = (mean_all * n_sess - sess) / (n_sess - 1)
        nc_lower += pearson_r_batch(sess, mean_others)
        print(f"    session {s+1}/{n_sess} done", end="\r", flush=True)

    print()
    return (nc_lower / n_sess).astype(np.float32)


def spearman_brown(r, n):
    """Spearman-Brown correction for n-fold reliability."""
    r = np.clip(r, -1.0, 1.0)
    denom = 1.0 + (n - 1) * r
    denom[np.abs(denom) < 1e-9] = 1e-9
    return np.clip((n * r) / denom, 0.0, 1.0)


def nc_to_volume(nc_flat, mask_nii):
    """Mapped (n_vox,) NC-Werte in (72,91,75) NIfTI-Raum."""
    mask = mask_nii.get_fdata().astype(bool)
    vol = np.zeros(mask.shape, dtype=np.float32)
    vol[mask] = nc_flat
    return vol


# ---------------------------------------------------------------------------
# Hauptprogramm
# ---------------------------------------------------------------------------

def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    db = cortex.database.Database(str(DB_ROOT))
    mask_nii = {
        subj: nib.load(str(PROCESSED_ROOT / subj / "brainmasks" / f"{subj}_brainmask.nii.gz"))
        for subj in SUBJECTS
    }

    nc_volumes = {}   # subject → (75, 91, 72) für pycortex

    for subj in SUBJECTS:
        print(f"\n{'='*50}\n{subj}\n{'='*50}", flush=True)

        betas, _ = load_test_betas(subj)
        print(f"  betas shape: {betas.shape}", flush=True)

        betas_z = per_session_zscore(betas)
        print(f"  Per-session z-scoring done.", flush=True)

        print(f"  Computing leave-one-session-out NC…", flush=True)
        nc_lower = compute_nc_voxelwise(betas_z)

        # Spearman-Brown → Zuverlässigkeit des 12-Sitzungen-Mittels
        nc_12 = spearman_brown(nc_lower, N_SESSIONS)

        # Erklärte Varianz [%]
        ev = np.clip(nc_12 ** 2 * 100, 0, 100)

        print(f"  EV stats: mean={ev.mean():.2f}%  max={ev.max():.2f}%  "
              f"p95={np.percentile(ev, 95):.2f}%", flush=True)

        # Speichere NC-Werte als CSV-Zeile
        np.save(str(RESULTS_DIR / f"nc_voxelwise_{subj}.npy"), ev)

        # Rekonstruiere 3D-Volumen (72,91,75) → transponiere für pycortex (75,91,72)
        vol_3d = nc_to_volume(ev, mask_nii[subj])
        nc_volumes[subj] = np.transpose(vol_3d, (2, 1, 0))

    # -----------------------------------------------------------------------
    # Pycortex-Flatmaps: drei Probanden nebeneinander
    # -----------------------------------------------------------------------
    print("\nRendering pycortex flatmaps…", flush=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    fig.patch.set_facecolor("black")

    for ax, subj in zip(axes, SUBJECTS):
        vol_data = nc_volumes[subj]
        cv = cortex.Volume(
            vol_data, subj, "align_auto",
            cmap="hot", vmin=0, vmax=VMAX_PERCENT,
        )

        # make_figure accepts an Axes as its `fig` argument
        cortex.quickflat.make_figure(
            cv,
            fig=ax,
            with_curvature=True,
            with_labels=False,
            with_rois=True,
            with_sulci=False,
            with_colorbar=False,
        )
        ax.set_title(subj, color="white", fontsize=14, fontweight="bold", pad=8)

    # Gemeinsame Colorbar
    sm = plt.cm.ScalarMappable(
        cmap="hot",
        norm=plt.Normalize(vmin=0, vmax=VMAX_PERCENT),
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, fraction=0.015, pad=0.03, aspect=30)
    cbar.set_label("Explainable Variance [%]", color="white", fontsize=11)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")
    cbar.outline.set_edgecolor("white")

    fig.suptitle(
        "Within-Subject Noise Ceiling – Explainable Variance (S1–S3)",
        color="white", fontsize=13, y=1.01,
    )

    out_png = FIGURES_DIR / "nc_pycortex_flatmap.png"
    out_pdf = FIGURES_DIR / "nc_pycortex_flatmap.pdf"
    fig.savefig(str(out_png), dpi=150, bbox_inches="tight",
                facecolor="black", edgecolor="none")
    fig.savefig(str(out_pdf), bbox_inches="tight",
                facecolor="black", edgecolor="none")
    plt.close(fig)

    print(f"\nSaved: {out_png}")
    print(f"Saved: {out_pdf}")


if __name__ == "__main__":
    main()

"""
Within-Subject Noise Ceiling für RSA-Analyse (nach Santi/Galella et al.).

Methode (pro Subject, pro ROI):
  1. Lade rohe Betas aus der Festplatte (1200 Trials = 100 Stimuli × 12 Sessions)
  2. Per-Session Z-Scoring: für jede Session separat über die 100 Stimuli z-standardisieren
  3. Berechne eine RDM pro Session (paarweise Cosine-Distanzen der 100 z-skorierten Betas)
  4. Leave-one-session-out:
       NC_lower_s = Spearman r(rdm_s, mean(rdm_{alle anderen Sessions}))
       NC_upper_s = Spearman r(rdm_s, mean(rdm_{alle 12 Sessions inkl. s}))
  5. NC_lower/upper = Mittelwert über alle 12 Sessions
  Ergebnis: NC pro Subject getrennt (nicht gemittelt über Subjects).

Verwendung:
  venv_cortex/bin/python scripts/compute_noise_ceiling.py
"""

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR     = PROJECT_ROOT / "data"
RESULTS_DIR  = PROJECT_ROOT / "outputs" / "results"
FIGURES_DIR  = PROJECT_ROOT / "outputs" / "figures"

PROCESSED_ROOT = Path("/Volumes/Sonstige Backups/Data/processed")

SUBJECTS = ["S1", "S2", "S3"]
ROIS     = ["V1", "hV4", "LOC", "IT"]

ROI_CONFIG = {
    "V1":  ("localizers", "V1"),
    "hV4": ("localizers", "hV4"),
    "LOC": ("localizers", "LOC"),
    "IT":  ("localizers", "IT"),
}

ROI_COLORS = {"V1": "#e41a1c", "hV4": "#ff7f00", "LOC": "#4daf4a", "IT": "#377eb8"}
LAYERS = ["conv1", "layer1", "layer2", "layer3", "layer4", "fc"]


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------

def spearman_r(x, y):
    """Kleine Spearman-Implementierung für Noise-Ceiling-RDMs.

    Hier werden bereits flache RDM-Vektoren verglichen. Es reicht also, beide
    Vektoren zu ranken und anschließend Pearson auf den Rängen zu berechnen.
    """
    def rank(v):
        order = np.argsort(v)
        r = np.empty_like(order, dtype=float)
        r[order] = np.arange(len(v))
        return r
    rx, ry = rank(x), rank(y)
    rx -= rx.mean(); ry -= ry.mean()
    denom = np.sqrt((rx**2).sum() * (ry**2).sum())
    return float(np.dot(rx, ry) / denom) if denom > 0 else 0.0


def find_column_index(csv_path, column_name):
    with open(csv_path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split(",")
    if column_name not in header:
        raise KeyError(f"Column '{column_name}' not found in {csv_path}")
    return header.index(column_name)


def load_roi_mask(subject_dir, roi_type, roi_name):
    roi_path = subject_dir / "rois" / roi_type / f"{subject_dir.name}_{roi_type}.csv"
    col_idx = find_column_index(roi_path, roi_name)
    mask = np.loadtxt(roi_path, delimiter=",", skiprows=1,
                      usecols=[col_idx], dtype=np.uint8).astype(bool)
    return mask


def load_raw_data(subject, roi_type, roi_name):
    """
    Lädt alle Test-THINGS-Trials des Subjects mit Session-Labels und ROI-maskierten Betas.
    Gibt zurück: (betas, sessions, stimuli) mit
      betas   shape (n_trials, n_roi_voxels)
      sessions list of str, len n_trials
      stimuli  list of str, len n_trials (Stimulus-Namen)
    """
    subject_dir = PROCESSED_ROOT / subject
    stimuli_path = subject_dir / "stimuli" / f"{subject}_stimuli.csv"

    with open(stimuli_path, "r", encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))

    # Nur THINGS-Testtrials verwenden. Diese Auswahl entspricht der Analyse,
    # die auch für ANN-vs-fMRI genutzt wird.
    row_indices = [
        i for i, r in enumerate(all_rows)
        if r.get("Split") == "test" and r.get("Dataset") == "THINGS"
    ]
    selected_rows = [all_rows[i] for i in row_indices]

    betas_path = subject_dir / "betas" / "vol" / f"{subject}_betas_vol.npy"
    betas_all = np.load(betas_path, mmap_mode="r")
    roi_mask = load_roi_mask(subject_dir, roi_type, roi_name)
    betas = np.asarray(betas_all[np.array(row_indices)][:, roi_mask], dtype=np.float32)

    sessions = [r["Session"] for r in selected_rows]
    stimuli  = [r["Stimulus"] for r in selected_rows]
    return betas, sessions, stimuli


def per_session_zscore(betas, sessions):
    """Z-score betas über Stimuli innerhalb jeder Session."""
    result = betas.copy()
    for sess in sorted(set(sessions)):
        idx = [i for i, s in enumerate(sessions) if s == sess]
        subset = result[idx, :]
        m = subset.mean(axis=0, keepdims=True)
        s = subset.std(axis=0, keepdims=True)
        s[s == 0] = 1.0
        result[idx, :] = (subset - m) / s
    return result


def compute_session_rdms(betas, sessions, stimuli):
    """
    Berechnet eine RDM pro Session.
    Gibt zurück: rdms array (n_sessions, n_stimuli*(n_stimuli-1)//2)
    und canonical_order (sortierte Stimulus-Namen).
    """
    # Eine feste Stimulusreihenfolge ist entscheidend: RDM-Eintrag (i,j) muss
    # in jeder Session dasselbe Stimuluspaar meinen.
    canonical_order = sorted(set(stimuli))
    n = len(canonical_order)
    stim_to_idx = {s: i for i, s in enumerate(canonical_order)}
    unique_sessions = sorted(set(sessions))

    rdms = []
    for sess in unique_sessions:
        sess_idx = [i for i, s in enumerate(sessions) if s == sess]
        sess_stimuli = [stimuli[i] for i in sess_idx]
        sess_betas = betas[sess_idx, :]

        # Ordne die Trial-Betas dieser Session in die kanonische Reihenfolge.
        # Danach ist jede Session-RDM direkt mit jeder anderen vergleichbar.
        ordered = np.zeros((n, betas.shape[1]), dtype=np.float32)
        for i, stim in enumerate(sess_stimuli):
            ordered[stim_to_idx[stim], :] = sess_betas[i]

        # L2-Normalisierung und Cosine-Distanz
        norms = np.linalg.norm(ordered, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normed = ordered / norms
        sim = normed @ normed.T
        dist = (1.0 - sim).astype(np.float32)

        tril = np.tril_indices(n, k=-1)
        rdms.append(dist[tril])

    return np.stack(rdms), canonical_order  # (12, 4950)


def compute_within_subject_nc(rdms):
    """
    Leave-one-session-out Noise Ceiling.
    rdms: (n_sessions, n_pairs)
    Gibt (nc_upper, nc_lower, per_session_upper, per_session_lower) zurück.
    """
    n = rdms.shape[0]
    mean_all = rdms.mean(axis=0)
    upper_vals, lower_vals = [], []
    for i in range(n):
        # Upper: Session gegen Mittel aller Sessions, inklusive sich selbst.
        # Dadurch ist dieser Wert leicht optimistisch.
        r_upper = spearman_r(rdms[i], mean_all)
        others = [j for j in range(n) if j != i]
        # Lower: Session gegen Mittel der anderen Sessions. Das ist die
        # konservativere Schätzung der erreichbaren Konsistenz.
        r_lower = spearman_r(rdms[i], rdms[others].mean(axis=0))
        upper_vals.append(r_upper)
        lower_vals.append(r_lower)
    return float(np.mean(upper_vals)), float(np.mean(lower_vals)), upper_vals, lower_vals


# ---------------------------------------------------------------------------
# Hauptprogramm
# ---------------------------------------------------------------------------

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    nc_results = {}

    print("Within-Subject Noise Ceiling (per-session z-scoring)\n" + "=" * 50)

    for roi in ROIS:
        roi_type, roi_name = ROI_CONFIG[roi]
        print(f"\n{roi}:")
        subj_uppers, subj_lowers = {}, {}

        for subj in SUBJECTS:
            try:
                betas, sessions, stimuli = load_raw_data(subj, roi_type, roi_name)
            except Exception as e:
                print(f"  {subj}: FEHLER beim Laden — {e}")
                continue

            betas_zscored = per_session_zscore(betas, sessions)
            rdms, _ = compute_session_rdms(betas_zscored, sessions, stimuli)
            nc_upper, nc_lower, _, _ = compute_within_subject_nc(rdms)

            print(f"  {subj}: NC_upper={nc_upper:.4f}, NC_lower={nc_lower:.4f}  ({rdms.shape[0]} sessions)")
            subj_uppers[subj] = nc_upper
            subj_lowers[subj] = nc_lower
            summary_rows.append({
                "roi": roi, "subject": subj,
                "nc_upper": round(nc_upper, 6),
                "nc_lower": round(nc_lower, 6),
            })

        if subj_uppers:
            mean_upper = float(np.mean(list(subj_uppers.values())))
            mean_lower = float(np.mean(list(subj_lowers.values())))
            print(f"  Mittelwert: NC_upper={mean_upper:.4f}, NC_lower={mean_lower:.4f}")
            nc_results[roi] = {
                "upper": mean_upper, "lower": mean_lower,
                "per_upper": subj_uppers, "per_lower": subj_lowers,
            }
            summary_rows.append({
                "roi": roi, "subject": "mean",
                "nc_upper": round(mean_upper, 6),
                "nc_lower": round(mean_lower, 6),
            })

    out_csv = RESULTS_DIR / "noise_ceiling.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["roi", "subject", "nc_upper", "nc_lower"])
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\nGespeichert: {out_csv}")

    _plot_with_noise_ceiling(nc_results)


def _plot_with_noise_ceiling(nc_results):
    rsa_csv = RESULTS_DIR / "ann_resnet18_all_subjects_roi_summary.csv"
    if not rsa_csv.exists():
        print("RSA-CSV nicht gefunden, kein Plot.")
        return

    rsa_data = {}
    with open(rsa_csv) as f:
        for row in csv.DictReader(f):
            key = (row["roi"], row["layer"])
            val = float(row.get("spearman_r") or row.get("spearman_r_ann_human") or 0)
            rsa_data.setdefault(key, []).append(val)
    rsa_mean = {k: float(np.mean(v)) for k, v in rsa_data.items()}

    fig, axes = plt.subplots(1, len(ROIS), figsize=(14, 4), sharey=False)

    for ax, roi in zip(axes, ROIS):
        nc = nc_results.get(roi)
        y_vals = [rsa_mean.get((roi, layer), np.nan) for layer in LAYERS]
        ax.plot(LAYERS, y_vals, marker="o", color=ROI_COLORS.get(roi, "gray"),
                linewidth=2, label="ResNet18", zorder=3)
        if nc:
            ax.axhspan(nc["lower"], nc["upper"], alpha=0.2, color="gray", label="Noise Ceiling")
            ax.axhline(nc["upper"], color="gray", linewidth=1, linestyle="--", alpha=0.7)
            ax.axhline(nc["lower"], color="gray", linewidth=1, linestyle=":", alpha=0.7)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="-", alpha=0.4)
        ax.set_title(roi, fontsize=12, color=ROI_COLORS.get(roi, "black"))
        ax.set_xlabel("ResNet18 Layer")
        if ax == axes[0]:
            ax.set_ylabel("Spearman r (RDM)")
        ax.tick_params(axis="x", rotation=30)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8, loc="lower right")

    fig.suptitle("RSA: ResNet18 vs. fMRI-ROIs mit Within-Subject Noise Ceiling (S1–S3)",
                 fontsize=13)
    plt.tight_layout()
    out = FIGURES_DIR / "rsa_resnet18_fmri_with_noise_ceiling.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Abbildung: {out}")


if __name__ == "__main__":
    main()

"""v2 (Surface/Whole-Brain): ResNet-50 vs. fMRI ueber die Glasser/HCP-Parzellierung.

Ersetzt die 4 handverlesenen Volumen-ROIs durch die ~180 oberflaechen-definierten
Glasser-Areale ueber den ganzen Kortex (Santis Whole-Brain-Idee, areale Variante).

Liest Santis THINGS-fMRI-Daten READ-ONLY:
  /work/dldevel/galella/datasets/THINGS-fMRI/betas_csv/sub-0X_ResponseData.h5
  ... StimulusMetadata.csv, VoxelMetadata.csv
Schreibt ausschliesslich in das v2-Ausgabeverzeichnis (weniger-Bereich).

Pro Proband:
  1. Test-Trials (1200 = 100 Stimuli x 12 Sessions) selektieren.
  2. Pro Session voxelweise z-standardisieren, dann ueber die 12 Sessions mitteln
     -> (100 Stimuli x n_voxel). Stimulus-Reihenfolge an die ResNet-50-image_order
     angeglichen.
  3. Pro Glasser-Areal: Cosine-RDM (L2-only) + sessionbasiertes Noise Ceiling.
  4. RSA (Spearman) + CKA gegen jede ResNet-50-Schicht.
Ausgabe: outputs_resnet50_v2_surface/results/glasser_all_subjects_summary.csv
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import h5py

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
from src.stats_utils import kernel_cka, spearman_corr, upper_triangle_values  # noqa: E402

LAYERS = ["conv1", "layer1", "layer2", "layer3", "layer4", "fc"]
SUBJ_MAP = {"S1": "sub-01", "S2": "sub-02", "S3": "sub-03"}


def per_session_zscore_average(betas_test, sessions, stim_ids, n_stim):
    """betas_test: (n_vox, 1200). sessions/stim_ids: (1200,). -> (n_stim, n_vox)."""
    n_vox = betas_test.shape[0]
    agg = np.zeros((n_stim, n_vox), dtype=np.float64)
    counts = np.zeros(n_stim)
    for s in np.unique(sessions):
        cols = np.where(sessions == s)[0]
        block = betas_test[:, cols].astype(np.float64)  # (n_vox, n_stim_in_session)
        mu = block.mean(axis=1, keepdims=True)
        sd = block.std(axis=1, keepdims=True)
        sd[sd == 0] = 1.0
        z = (block - mu) / sd  # (n_vox, n_stim_in_session)
        for k, c in enumerate(cols):
            sid = stim_ids[c]
            agg[sid] += z[:, k]
            counts[sid] += 1
    counts[counts == 0] = 1
    return agg / counts[:, None]


def cosine_rdm_l2(features):
    """L2-only Cosine-Distanz (fMRI-Seite): keine Spalten-Standardisierung."""
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    n = features / norms
    return 1.0 - n @ n.T


def session_rdms(betas_test, sessions, stim_ids, n_stim, vox_mask):
    """Pro Session eine RDM (fuer Noise Ceiling). -> list of (n_stim,n_stim)."""
    rdms = []
    for s in np.unique(sessions):
        cols = np.where(sessions == s)[0]
        block = betas_test[vox_mask][:, cols].astype(np.float64)
        mu = block.mean(axis=1, keepdims=True)
        sd = block.std(axis=1, keepdims=True)
        sd[sd == 0] = 1.0
        z = ((block - mu) / sd).T  # (n_stim_in_session, n_vox)
        order = np.argsort([stim_ids[c] for c in cols])
        rdms.append(cosine_rdm_l2(z[order]))
    return rdms


def noise_ceiling(rdms):
    tris = [upper_triangle_values(r) for r in rdms]
    mean_all = np.mean(tris, axis=0)
    up, lo = [], []
    for i, t in enumerate(tris):
        up.append(spearman_corr(t, mean_all))
        loo = np.mean([tris[j] for j in range(len(tris)) if j != i], axis=0)
        lo.append(spearman_corr(t, loo))
    return float(np.mean(up)), float(np.mean(lo))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--things-root", default="/work/dldevel/galella/datasets/THINGS-fMRI")
    ap.add_argument("--ann-geometry", default="outputs_resnet50/geometry")
    ap.add_argument("--ann-order-dir", default="outputs_resnet50/features")
    ap.add_argument("--output-root", default="outputs_resnet50_v2_surface")
    ap.add_argument("--subjects", nargs="+", default=["S1", "S2", "S3"])
    ap.add_argument("--min-voxels", type=int, default=20)
    args = ap.parse_args()

    things = Path(args.things_root)
    out = PROJECT_ROOT / args.output_root
    (out / "glasser").mkdir(parents=True, exist_ok=True)
    (out / "results").mkdir(parents=True, exist_ok=True)

    rows = []
    for subj in args.subjects:
        sub = SUBJ_MAP[subj]
        print(f"\n==== {subj} ({sub}) ====", flush=True)
        sm = pd.read_csv(things / "betas_csv" / f"{sub}_StimulusMetadata.csv")
        test = sm[sm.trial_type == "test"].reset_index(drop=True)
        test_cols = sm.index[sm.trial_type == "test"].to_numpy()
        sessions = test.session.to_numpy()
        stim_names = test.stimulus.to_numpy()

        # Stimulus-Reihenfolge an ResNet-50 angleichen
        order_csv = PROJECT_ROOT / args.ann_order_dir / f"ann_resnet50_{subj}_matched_image_order.csv"
        ann_order = [r["image_id"] for r in csv.DictReader(open(order_csv))]
        stim_to_idx = {name: i for i, name in enumerate(ann_order)}
        n_stim = len(ann_order)
        stim_ids = np.array([stim_to_idx.get(s, -1) for s in stim_names])
        if (stim_ids < 0).any():
            missing = set(stim_names[stim_ids < 0])
            raise RuntimeError(f"{subj}: {len(missing)} Test-Stimuli nicht in ANN-order, z.B. {list(missing)[:3]}")

        # Betas der Test-Trials laden (read-only)
        with h5py.File(things / "betas_csv" / f"{sub}_ResponseData.h5", "r") as f:
            betas_test = f["ResponseData"]["block0_values"][:, test_cols]  # (n_vox, 1200)
        print(f"  betas_test {betas_test.shape}", flush=True)

        agg = per_session_zscore_average(betas_test, sessions, stim_ids, n_stim)  # (100, n_vox)

        # ANN-RDMs (upper tri) je Layer laden
        ann_tri = {}
        for layer in LAYERS:
            ann = np.load(PROJECT_ROOT / args.ann_geometry / f"ann_resnet50_{subj}_matched_{layer}_cosine.npy")
            ann_tri[layer] = upper_triangle_values(ann)

        vm = pd.read_csv(things / "betas_csv" / f"{sub}_VoxelMetadata.csv")
        glasser_cols = [c for c in vm.columns if c.startswith("glasser-")]

        for area in glasser_cols:
            mask = vm[area].to_numpy().astype(bool)
            nvox = int(mask.sum())
            if nvox < args.min_voxels:
                continue
            fmri_rdm = cosine_rdm_l2(agg[:, mask])
            np.save(out / "glasser" / f"{subj}_{area}_rdm.npy", fmri_rdm)
            fmri_tri = upper_triangle_values(fmri_rdm)
            K_h = 1.0 - fmri_rdm
            np.fill_diagonal(K_h, 1.0)
            nc_up, nc_lo = noise_ceiling(session_rdms(betas_test, sessions, stim_ids, n_stim, mask))
            nc_meanvox = float(np.nanmean(vm.loc[mask, "nc_testset"].to_numpy())) if "nc_testset" in vm else np.nan
            for layer in LAYERS:
                rho = spearman_corr(fmri_tri, ann_tri[layer])
                ann_full = np.load(PROJECT_ROOT / args.ann_geometry / f"ann_resnet50_{subj}_matched_{layer}_cosine.npy")
                K_d = 1.0 - ann_full
                np.fill_diagonal(K_d, 1.0)
                cka = kernel_cka(K_d, K_h)
                rows.append({
                    "subject": subj, "area": area.replace("glasser-", ""), "n_voxels": nvox,
                    "layer": layer, "spearman_r": round(rho, 6),
                    "cka": round(cka, 6) if not np.isnan(cka) else "nan",
                    "nc_upper": round(nc_up, 6), "nc_lower": round(nc_lo, 6),
                    "nc_testset_meanvox": round(nc_meanvox, 4),
                })
            print(f"  {area:16s} nvox={nvox:5d} NCup={nc_up:.3f}", flush=True)

    out_csv = out / "results" / "glasser_all_subjects_summary.csv"
    fields = ["subject", "area", "n_voxels", "layer", "spearman_r", "cka",
              "nc_upper", "nc_lower", "nc_testset_meanvox"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\nGespeichert: {out_csv}  ({len(rows)} Zeilen)", flush=True)


if __name__ == "__main__":
    main()

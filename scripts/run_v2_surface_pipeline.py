"""v2 (Surface / Whole-Brain): ResNet-50 vs. fMRI, 1:1 nach Galella et al. (2025).

Vollstaendig oberflaechenbasierte Pipeline mit GEODAETISCHEM Searchlight (Santis
vorberechnete naechsten Vertices), getrennt pro Hemisphaere. Pro Vertex wird
die lokale fMRI-RDM aus den 100 Stimuli gebildet und gegen jede ResNet-50-Schicht
sowohl per RSA (Spearman) als auch per CKA verglichen. Der Searchlight laeuft fuer
alle angegebenen k-Werte (Standard: 50 und 100), entsprechend Santis Hauptanalyse
(k=50) und seiner Robustheitspruefung (k=100). Areal-Schaetzer entstehen durch
Mittelung der Vertex-Werte ueber alle Vertices eines Glasser/HCP-Areals
(k-unabhaengig, da die vollen Areal-Vertices verwendet werden). Zusaetzlich wird
pro Areal eine oberflaechenbasierte Cosine-RDM gespeichert (TDA-Schritt H0/H1/H2).

Unterschied zur frueheren Fassung (run_v2_surface_searchlight.py):
  * geodaetische NN statt euklidischer cKDTree-50-NN,
  * CKA zusaetzlich zu RSA pro Vertex,
  * mehrere k-Werte in einem Lauf (Robustheitspruefung),
  * areale Aggregation auf der Oberflaeche + oberflaechenbasierte Areal-RDMs,
  * optionale vertex-weise TDA (--tda): Wasserstein-Distanz pro Vertex und Layer.

Alle fremden Daten werden NUR GELESEN:
  Santis Betas:    /work/dldevel/galella/datasets/THINGS-fMRI/betas_csv/*.h5
  Santis Surfaces: /work/dldevel/galella/datasets/THINGS-fMRI/freesurfer/S{n}/
  Santis Nachbarn: <neighbors-dir>/S{n}_neighbors_surf_{lh,rh}.npz  (vorberechnet)
  Brainmask:       things_surface_dl/brainmasks/sub-0X_space-T1w_brainmask.nii.gz
Ausgaben nur in outputs_resnet50_v2_surface/ (weniger-Bereich).
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import h5py
import nibabel as nib
from nibabel.freesurfer import io as fsio
from nibabel.affines import apply_affine
from nilearn import surface
from scipy.stats import rankdata

try:
    from ripser import ripser as _ripser
    from persim import wasserstein as _wasserstein
    HAS_TDA = True
except ImportError:
    HAS_TDA = False

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
from src.stats_utils import spearman_corr, upper_triangle_values  # noqa: E402

LAYERS = ["conv1", "layer1", "layer2", "layer3", "layer4", "fc"]
SUBJ_MAP = {"S1": "sub-01", "S2": "sub-02", "S3": "sub-03"}


# --------------------------------------------------------------------------- #
# Betas: Test-Trials sessionweise z-standardisieren und ueber Sessions mitteln
# --------------------------------------------------------------------------- #
def aggregate_betas(things, sub, ann_order):
    sm = pd.read_csv(things / "betas_csv" / f"{sub}_StimulusMetadata.csv")
    test_cols = sm.index[sm.trial_type == "test"].to_numpy()
    sessions = sm.loc[test_cols, "session"].to_numpy()
    names = sm.loc[test_cols, "stimulus"].to_numpy()
    idx = {n: i for i, n in enumerate(ann_order)}
    sid = np.array([idx[n] for n in names])
    n_stim = len(ann_order)
    with h5py.File(things / "betas_csv" / f"{sub}_ResponseData.h5", "r") as f:
        betas = f["ResponseData"]["block0_values"][:, test_cols]  # (n_vox, 1200)
    agg = np.zeros((n_stim, betas.shape[0]))
    cnt = np.zeros(n_stim)
    for s in np.unique(sessions):
        cols = np.where(sessions == s)[0]
        b = betas[:, cols].astype(np.float64)
        sd = b.std(1, keepdims=True)
        b = (b - b.mean(1, keepdims=True)) / np.where(sd == 0, 1, sd)
        for k, c in enumerate(cols):
            agg[sid[c]] += b[:, k]
            cnt[sid[c]] += 1
    return agg / cnt[:, None]  # (100, n_vox)


def make_volume(values, mask_img, vox_xyz, dtype=np.float32):
    """Tabellen-Werte (n, n_vox) -> 4D-Nifti im T1w-Raum der Maske."""
    vx, vy, vz = vox_xyz[:, 0], vox_xyz[:, 1], vox_xyz[:, 2]
    if values.ndim == 1:
        vol = np.zeros(mask_img.shape, dtype=dtype)
        vol[vx, vy, vz] = values
    else:
        vol = np.zeros(mask_img.shape + (values.shape[0],), dtype=dtype)
        vol[vx, vy, vz, :] = values.T
    return nib.Nifti1Image(vol, mask_img.affine)


def tkr_to_scanner(coords, orig):
    M = orig.header.get_vox2ras() @ np.linalg.inv(orig.header.get_vox2ras_tkr())
    return apply_affine(M, coords)


def _surf(fs_dir, hemi, names):
    for n in names:
        p = fs_dir / "surf" / f"{hemi}.{n}"
        if p.exists():
            return p
    raise FileNotFoundError(f"keine Surface {names} fuer {hemi} in {fs_dir}")


def hemi_meshes(fs_dir, hemi, orig):
    white_c, faces = fsio.read_geometry(_surf(fs_dir, hemi, ["white"]))
    pial_c, _ = fsio.read_geometry(_surf(fs_dir, hemi, ["pial", "pial.T1", "pial.T2"]))
    white = (tkr_to_scanner(white_c, orig), faces)
    pial = (tkr_to_scanner(pial_c, orig), faces)
    return white, pial


def project(vol_img, white, pial, interpolation="linear"):
    """vol_to_surf zwischen white und pial (Mid-Thickness-Sampling)."""
    tex = surface.vol_to_surf(vol_img, surf_mesh=pial, inner_mesh=white,
                              interpolation=interpolation)
    return tex  # (n_vert,) oder (n_vert, n)


# --------------------------------------------------------------------------- #
# Statistik-Helfer
# --------------------------------------------------------------------------- #
def _zrank(x):
    r = rankdata(x)
    s = r.std()
    return (r - r.mean()) / (s if s > 0 else 1.0)


def _double_center(K):
    rm = K.mean(0, keepdims=True)
    return K - rm - rm.T + K.mean()


def cka_from_centered(Kc_h, ann_centered, ann_self):
    """CKA zwischen einer (schon zentrierten) fMRI-Kernmatrix und vorzentrierten
    ANN-Kernmatrizen. ann_self = HSIC(K_d,K_d) je Layer (vorberechnet)."""
    h_hh = np.sum(Kc_h * Kc_h)
    if h_hh <= 0:
        return np.full(len(ann_centered), np.nan, dtype=np.float32)
    out = np.empty(len(ann_centered), dtype=np.float32)
    for i, Kc_d in enumerate(ann_centered):
        denom = np.sqrt(ann_self[i] * h_hh)
        out[i] = (np.sum(Kc_d * Kc_h) / denom) if denom > 0 else np.nan
    return out


def _normalize_rdm(rdm):
    m = rdm.max()
    return rdm / m if m > 0 else rdm


def compute_ann_pds(ann_rdms_raw, maxdim):
    """Vorberechnung der ANN-Persistenzdiagramme (einmal pro Layer).

    Gibt eine Liste zurueck: pds[layer_idx][hdim] = array (n_features, 2)
    ohne unendliche Todeszeiten.
    """
    pds = []
    for rdm in ann_rdms_raw:
        norm = _normalize_rdm(rdm)
        dgms = _ripser(norm, maxdim=maxdim, distance_matrix=True)["dgms"]
        pds.append([dgm[np.isfinite(dgm[:, 1])] for dgm in dgms])
    return pds


def searchlight(tex, neighbors, ann_rank, ann_centered, ann_self,
                ann_pds=None, tda_maxdim=0):
    """tex: (100, n_vert). neighbors: (n_vert, k) int. Pro Vertex RSA + CKA je Layer.

    RSA vektorisiert: ANN-Dreiecke vorgerankt/z-normiert, fMRI-RDM einmal ranken,
    alle 6 Spearman-Korrelationen per Matrixprodukt. CKA per Doppel-Zentrierung.

    Wenn ann_pds angegeben: zusaetzlich TDA-Wasserstein-Distanz pro Vertex und Layer.
    ann_pds[layer_idx][hdim] = vorberechnetes ANN-Persistenzdiagramm.
    """
    n_stim, n_vert = tex.shape
    iu = np.triu_indices(n_stim, 1)
    n_pairs = iu[0].size
    rsa = np.full((n_vert, len(LAYERS)), np.nan, dtype=np.float32)
    cka = np.full((n_vert, len(LAYERS)), np.nan, dtype=np.float32)
    n_hdim = tda_maxdim + 1
    wd = np.full((n_vert, len(LAYERS), n_hdim), np.nan, dtype=np.float32) \
        if ann_pds is not None else None
    for v in range(n_vert):
        patch = tex[:, neighbors[v]]  # (n_stim, k)
        norms = np.linalg.norm(patch, axis=1, keepdims=True)
        if not np.all(norms > 0):
            continue
        p = patch / norms
        sim = p @ p.T  # Cosine-Aehnlichkeit (100x100)
        tri = (1.0 - sim)[iu]
        if not np.isfinite(tri).all() or tri.std() == 0:
            continue
        rsa[v] = (ann_rank @ _zrank(tri)) / n_pairs
        Kc_h = _double_center(sim)
        cka[v] = cka_from_centered(Kc_h, ann_centered, ann_self)
        if ann_pds is not None:
            rdm_v = _normalize_rdm(1.0 - sim)
            dgms_v = _ripser(rdm_v, maxdim=tda_maxdim, distance_matrix=True)["dgms"]
            for hi in range(n_hdim):
                fmri_dgm = dgms_v[hi][np.isfinite(dgms_v[hi][:, 1])]
                for li in range(len(LAYERS)):
                    ann_dgm = ann_pds[li][hi]
                    if len(fmri_dgm) == 0 and len(ann_dgm) == 0:
                        wd[v, li, hi] = 0.0
                    else:
                        wd[v, li, hi] = float(_wasserstein(ann_dgm, fmri_dgm))
    return rsa, cka, wd


def cosine_rdm_l2(features):
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    n = features / norms
    return 1.0 - n @ n.T


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--things-root", default="/work/dldevel/galella/datasets/THINGS-fMRI")
    ap.add_argument("--mask-dir", default="things_surface_dl/brainmasks")
    ap.add_argument("--neighbors-dir", default="things_surface_dl/neighbors",
                    help="Verzeichnis mit S{n}_neighbors_surf_{lh,rh}.npz (Santi)")
    ap.add_argument("--ann-geometry", default="outputs_resnet50/geometry")
    ap.add_argument("--ann-order-dir", default="outputs_resnet50/features")
    ap.add_argument("--output-root", default="outputs_resnet50_v2_surface")
    ap.add_argument("--subjects", nargs="+", default=["S1", "S2", "S3"])
    ap.add_argument("--k", type=int, nargs="+", default=[50, 100],
                    help="Searchlight-Groessen (Santi: 50 Hauptanalyse, 100 Robustheit)")
    ap.add_argument("--min-vertices", type=int, default=20)
    ap.add_argument("--save-tex", action="store_true",
                    help="projizierte (100,n_vert)-Betas speichern (fuer Betas-Check)")
    ap.add_argument("--tda", action="store_true",
                    help="Vertex-weise TDA (Wasserstein-Distanz) zusaetzlich berechnen")
    ap.add_argument("--tda-maxdim", type=int, default=0, choices=[0, 1],
                    help="Maximale Homologie-Dimension: 0=H0 (schnell), 1=H0+H1")
    args = ap.parse_args()

    things = Path(args.things_root)
    nbr_dir = PROJECT_ROOT / args.neighbors_dir
    out = PROJECT_ROOT / args.output_root
    (out / "searchlight").mkdir(parents=True, exist_ok=True)
    (out / "glasser").mkdir(parents=True, exist_ok=True)
    (out / "results").mkdir(parents=True, exist_ok=True)

    areal_rows = []
    for subj in args.subjects:
        sub = SUBJ_MAP[subj]
        print(f"\n==== {subj} ({sub}) ====", flush=True)
        order = [r["image_id"] for r in csv.DictReader(
            open(PROJECT_ROOT / args.ann_order_dir / f"ann_resnet50_{subj}_matched_image_order.csv"))]

        # ANN-RDMs vorbereiten: Rang-Dreiecke (RSA), zentrierte Kernel (CKA), Roh-RDMs (TDA)
        ann_rank, ann_centered, ann_self, ann_rdms_raw = [], [], [], []
        for l in LAYERS:
            rdm = np.load(PROJECT_ROOT / args.ann_geometry / f"ann_resnet50_{subj}_matched_{l}_cosine.npy")
            ann_rdms_raw.append(rdm)
            ann_rank.append(_zrank(upper_triangle_values(rdm)))
            Kd = _double_center(1.0 - rdm)
            ann_centered.append(Kd)
            ann_self.append(float(np.sum(Kd * Kd)))
        ann_rank = np.stack(ann_rank)  # (6, n_pairs)

        # ANN-Persistenzdiagramme vorberechnen (einmal pro Proband, alle Layer)
        ann_pds = None
        if args.tda:
            if not HAS_TDA:
                raise ImportError("--tda benoetigt ripser und persim: pip install ripser persim")
            print(f"  Vorberechnung ANN-Persistenzdiagramme (maxdim={args.tda_maxdim}) ...", flush=True)
            ann_pds = compute_ann_pds(ann_rdms_raw, args.tda_maxdim)

        # Betas projizieren
        agg = aggregate_betas(things, sub, order)  # (100, n_vox)
        vm = pd.read_csv(things / "betas_csv" / f"{sub}_VoxelMetadata.csv")
        vox_xyz = vm[["voxel_x", "voxel_y", "voxel_z"]].to_numpy().astype(int)
        glasser_cols = [c for c in vm.columns if c.startswith("glasser-")]
        onehot = vm[glasser_cols].to_numpy()
        area_id = np.where(onehot.sum(1) > 0, onehot.argmax(1) + 1, 0).astype(np.int32)

        mask = nib.load(PROJECT_ROOT / args.mask_dir / f"{sub}_space-T1w_brainmask.nii.gz")
        beta_vol = make_volume(agg, mask, vox_xyz)
        label_vol = make_volume(area_id.astype(np.float32), mask, vox_xyz)
        orig = nib.load(things / "freesurfer" / subj / "mri" / "orig.mgz")
        fs_dir = things / "freesurfer" / subj

        tex_all, label_all = [], []  # ueber beide Hemis fuer areale RDMs
        # Nachbar-Arrays einmal laden (max(k) Spalten reichen fuer alle k)
        nbr_full = {}
        k_max = max(args.k)
        for hemi in ["lh", "rh"]:
            white, pial = hemi_meshes(fs_dir, hemi, orig)
            tex = project(beta_vol, white, pial, "linear").T  # (100, n_vert)
            lab = np.rint(project(label_vol, white, pial, "nearest")).astype(np.int32)
            nbr_full[hemi] = np.load(
                nbr_dir / f"{subj}_neighbors_surf_{hemi}.npz")["indices"][:, :k_max]
            assert nbr_full[hemi].shape[0] == tex.shape[1], (
                f"{subj} {hemi}: Nachbarn {nbr_full[hemi].shape[0]} != Vertices {tex.shape[1]} "
                "(Vertex-Reihenfolge passt nicht zu Santis Mesh)")
            # Searchlight fuer jeden k-Wert
            for k in args.k:
                nbr_k = nbr_full[hemi][:, :k]
                print(f"  {hemi}: tex {tex.shape}, geodaetischer Searchlight k={k} ...", flush=True)
                rsa, cka, wd = searchlight(tex, nbr_k, ann_rank, ann_centered, ann_self,
                                           ann_pds=ann_pds, tda_maxdim=args.tda_maxdim)
                tag = f"k{k}"
                np.save(out / "searchlight" / f"{subj}_{hemi}_rsa_perlayer_{tag}.npy", rsa)
                np.save(out / "searchlight" / f"{subj}_{hemi}_cka_perlayer_{tag}.npy", cka)
                best = np.full(rsa.shape[0], np.nan, dtype=np.float32)
                ok = ~np.all(np.isnan(rsa), axis=1)
                best[ok] = np.nanargmax(np.where(np.isnan(rsa), -np.inf, rsa), axis=1)[ok]
                np.save(out / "searchlight" / f"{subj}_{hemi}_bestlayer_{tag}.npy", best)
                np.save(out / "searchlight" / f"{subj}_{hemi}_peakr_{tag}.npy",
                        np.nanmax(rsa, axis=1).astype(np.float32))
                if wd is not None:
                    n_hdim = args.tda_maxdim + 1
                    for hi in range(n_hdim):
                        np.save(out / "searchlight" /
                                f"{subj}_{hemi}_tda_wd_H{hi}_perlayer_{tag}.npy",
                                wd[:, :, hi].astype(np.float32))
                    # Beste Schicht nach minimaler WD (Mittel ueber H-Dimensionen)
                    wd_mean = np.nanmean(wd, axis=2)  # (n_vert, 6)
                    best_tda = np.full(wd_mean.shape[0], np.nan, dtype=np.float32)
                    ok_tda = ~np.all(np.isnan(wd_mean), axis=1)
                    best_tda[ok_tda] = np.nanargmin(
                        np.where(np.isnan(wd_mean[ok_tda]), np.inf, wd_mean[ok_tda]), axis=1)
                    np.save(out / "searchlight" /
                            f"{subj}_{hemi}_tda_bestlayer_{tag}.npy", best_tda)
                    print(f"    TDA-Outputs gespeichert (H0..H{args.tda_maxdim})", flush=True)
            if args.save_tex:
                np.save(out / "searchlight" / f"{subj}_{hemi}_tex.npy", tex.astype(np.float32))
            tex_all.append(tex)
            label_all.append(lab)

        tex_all = np.concatenate(tex_all, axis=1)        # (100, n_lh+n_rh)
        label_all = np.concatenate(label_all)            # (n_lh+n_rh,)

        # Areale RSA/CKA + oberflaechenbasierte Areal-RDM (fuer TDA)
        for a_idx, col in enumerate(glasser_cols, start=1):
            sel = label_all == a_idx
            nvert = int(sel.sum())
            if nvert < args.min_vertices:
                continue
            # Areal-Namen saeubern (einige VoxelMetadata-Header enthalten Umbrueche)
            area = col.replace("glasser-", "").replace("\n", "").replace("\r", "").strip()
            # Areale RSA/CKA aus der vollen Areal-Oberflaechen-RDM (alle Vertices des
            # Areals), konsistent mit dem arealen TDA-Schritt. Der geodaetische
            # Searchlight liefert die vertexweise Karte (rsa_all/cka_all -> bestlayer).
            fmri_rdm = cosine_rdm_l2(tex_all[:, sel])
            np.save(out / "glasser" / f"{subj}_glasser-{area}_rdm.npy", fmri_rdm.astype(np.float32))
            iu = np.triu_indices(fmri_rdm.shape[0], 1)
            fmri_rank = _zrank(fmri_rdm[iu])
            Kc_h = _double_center(1.0 - fmri_rdm)
            rsa_area = (ann_rank @ fmri_rank) / iu[0].size
            cka_area = cka_from_centered(Kc_h, ann_centered, ann_self)
            for li, l in enumerate(LAYERS):
                areal_rows.append({
                    "subject": subj, "area": area,
                    "n_vertices": nvert, "layer": l,
                    "spearman_r": round(float(rsa_area[li]), 6),
                    "cka": round(float(cka_area[li]), 6),
                })
        print(f"  areale Aggregation fertig ({len(glasser_cols)} Areale geprueft)", flush=True)

    out_csv = out / "results" / "glasser_surface_summary.csv"
    fields = ["subject", "area", "n_vertices", "layer", "spearman_r", "cka"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(areal_rows)
    print(f"\nFertig. Areal-Summary: {out_csv} ({len(areal_rows)} Zeilen)", flush=True)


if __name__ == "__main__":
    main()

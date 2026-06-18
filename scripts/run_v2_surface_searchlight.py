"""v2-B: ECHTE Surface-Searchlight-Analyse (ResNet-50 vs. fMRI), Santis Methode.

Projiziert Santis THINGS-fMRI-Betas (T1w-Volumenraum, 211339 Voxel) auf die
FreeSurfer-Kortexoberfläche und schiebt einen geodätischen Searchlight (50
nächste Vertices) über den ganzen Kortex; pro Vertex wird die lokale fMRI-RDM
gebildet und gegen jede ResNet-50-Schicht via RSA (Spearman) verglichen.

Alle fremden Daten werden NUR GELESEN:
  Santis Betas:    /work/dldevel/galella/datasets/THINGS-fMRI/betas_csv/*.h5
  Santis Surfaces: /work/dldevel/galella/datasets/THINGS-fMRI/freesurfer/S{n}/
  Brainmask:       things_surface_dl/brainmasks/sub-0X_space-T1w_brainmask.nii.gz
Ausgaben nur in outputs_resnet50_v2_surface/searchlight/ (weniger-Bereich).
"""
import argparse, csv, sys
from pathlib import Path
import numpy as np
import pandas as pd
import h5py
import nibabel as nib
from nibabel.freesurfer import io as fsio
from nibabel.affines import apply_affine
from nilearn import surface
from scipy.spatial import cKDTree
from scipy.stats import rankdata

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
from src.stats_utils import spearman_corr, upper_triangle_values  # noqa: E402

LAYERS = ["conv1", "layer1", "layer2", "layer3", "layer4", "fc"]
SUBJ_MAP = {"S1": "sub-01", "S2": "sub-02", "S3": "sub-03"}


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
        b = (b - b.mean(1, keepdims=True)) / np.where(b.std(1, keepdims=True) == 0, 1, b.std(1, keepdims=True))
        for k, c in enumerate(cols):
            agg[sid[c]] += b[:, k]; cnt[sid[c]] += 1
    return agg / cnt[:, None]  # (100, n_vox)


def make_volume(agg, mask_img, vox_xyz):
    """Tabellen-Betas (100, n_vox) -> 4D-Nifti im T1w-Raum der Maske.

    Platzierung über die echten voxel_x/y/z aus den THINGS-VoxelMetadata
    (Spalte j von agg == Zeile j der VoxelMetadata == voxel_id j im h5).
    """
    vx, vy, vz = vox_xyz[:, 0], vox_xyz[:, 1], vox_xyz[:, 2]
    vol = np.zeros(mask_img.shape + (agg.shape[0],), dtype=np.float32)
    vol[vx, vy, vz, :] = agg.T
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


def project_hemi(vol_img, fs_dir, hemi, orig):
    white_c, faces = fsio.read_geometry(_surf(fs_dir, hemi, ["white"]))
    pial_c, _ = fsio.read_geometry(_surf(fs_dir, hemi, ["pial", "pial.T1", "pial.T2"]))
    white = (tkr_to_scanner(white_c, orig), faces)
    pial = (tkr_to_scanner(pial_c, orig), faces)
    # Mid-thickness-Sampling: zwischen white und pial
    tex = surface.vol_to_surf(vol_img, surf_mesh=pial, inner_mesh=white,
                              interpolation="linear")  # (n_vert, 100)
    return tex.T, white[0], faces  # (100, n_vert), coords, faces


def _zrank(x):
    r = rankdata(x)
    s = r.std()
    return (r - r.mean()) / (s if s > 0 else 1.0)


def searchlight(tex, coords, ann_tri, k=50):
    """tex: (100, n_vert). Pro Vertex 50-NN-RDM vs ANN-RDMs -> r je Layer.

    Vektorisiert: ANN-Vektoren einmal vorranken/z-normieren, dann pro Vertex die
    fMRI-RDM einmal ranken und alle 6 Spearman-Korrelationen via Matrixprodukt.
    """
    n_stim = tex.shape[0]
    n_vert = tex.shape[1]
    iu = np.triu_indices(n_stim, 1)
    n_pairs = iu[0].size
    ann_mat = np.stack([_zrank(ann_tri[l]) for l in LAYERS])  # (6, n_pairs)
    tree = cKDTree(coords)
    _, nn = tree.query(coords, k=k)  # (n_vert, k)
    out = np.full((n_vert, len(LAYERS)), np.nan, dtype=np.float32)
    for v in range(n_vert):
        patch = tex[:, nn[v]]  # (n_stim, k)
        norms = np.linalg.norm(patch, axis=1, keepdims=True)
        if not np.all(norms > 0):
            continue
        p = patch / norms
        tri = (1.0 - p @ p.T)[iu]
        if not np.isfinite(tri).all() or tri.std() == 0:
            continue
        out[v] = (ann_mat @ _zrank(tri)) / n_pairs  # Spearman = Pearson auf Raengen
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--things-root", default="/work/dldevel/galella/datasets/THINGS-fMRI")
    ap.add_argument("--mask-dir", default="things_surface_dl/brainmasks")
    ap.add_argument("--ann-geometry", default="outputs_resnet50/geometry")
    ap.add_argument("--ann-order-dir", default="outputs_resnet50/features")
    ap.add_argument("--output-root", default="outputs_resnet50_v2_surface")
    ap.add_argument("--subjects", nargs="+", default=["S1", "S2", "S3"])
    ap.add_argument("--k", type=int, default=50)
    args = ap.parse_args()

    things = Path(args.things_root)
    out = PROJECT_ROOT / args.output_root / "searchlight"
    out.mkdir(parents=True, exist_ok=True)

    for subj in args.subjects:
        sub = SUBJ_MAP[subj]
        print(f"\n==== {subj} ====", flush=True)
        order = [r["image_id"] for r in csv.DictReader(
            open(PROJECT_ROOT / args.ann_order_dir / f"ann_resnet50_{subj}_matched_image_order.csv"))]
        ann_tri = {l: upper_triangle_values(np.load(
            PROJECT_ROOT / args.ann_geometry / f"ann_resnet50_{subj}_matched_{l}_cosine.npy")) for l in LAYERS}

        agg = aggregate_betas(things, sub, order)
        vm = pd.read_csv(things / "betas_csv" / f"{sub}_VoxelMetadata.csv",
                         usecols=["voxel_x", "voxel_y", "voxel_z"])
        vox_xyz = vm[["voxel_x", "voxel_y", "voxel_z"]].to_numpy().astype(int)
        mask = nib.load(PROJECT_ROOT / args.mask_dir / f"{sub}_space-T1w_brainmask.nii.gz")
        vol = make_volume(agg, mask, vox_xyz)
        orig = nib.load(things / "freesurfer" / subj / "mri" / "orig.mgz")
        fs_dir = things / "freesurfer" / subj

        for hemi in ["lh", "rh"]:
            tex, coords, faces = project_hemi(vol, fs_dir, hemi, orig)
            print(f"  {hemi}: projected {tex.shape}, searchlight ...", flush=True)
            r = searchlight(tex, coords, ann_tri, k=args.k)  # (n_vert, 6)
            np.save(out / f"{subj}_{hemi}_rsa_perlayer.npy", r.astype(np.float32))
            best = np.nanargmax(np.where(np.isnan(r), -np.inf, r), axis=1).astype(np.float32)
            best[np.all(np.isnan(r), axis=1)] = np.nan
            np.save(out / f"{subj}_{hemi}_bestlayer.npy", best)
            np.save(out / f"{subj}_{hemi}_peakr.npy", np.nanmax(r, axis=1).astype(np.float32))
            print(f"  {hemi}: saved (n_vert={r.shape[0]})", flush=True)

    print("\nSurface-Searchlight fertig.", flush=True)


if __name__ == "__main__":
    main()

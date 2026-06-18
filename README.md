# Brain–ANN Alignment Analysis

Code for the bachelor's thesis: **"Which layers of artificial neural networks align geometrically and topologically with human visual cortex representations?"**

Victor Weniger, Goethe University Frankfurt, 2026.

Analysis pipeline comparing ResNet-50 layer activations with fMRI responses from the
[THINGS-fMRI dataset](https://things-initiative.org/) (Galella et al. 2025) using RSA,
CKA, and topological data analysis (TDA). The primary analysis projects fMRI betas onto
the cortical surface and runs a geodesic searchlight (k=50 vertices, following Galella et
al. 2025).

---

## Repository structure

```
ba_analyse_software/
├── scripts/            # Pipeline entry points (run_*, figures_*, compute_*)
├── src/                # Shared utilities (stats_utils.py, utils.py, ...)
├── configs/            # JSON configs for model/data paths
├── data/               # Local data (NOT committed — see below)
├── docs/               # Notes and documentation
└── requirements.txt    # Python dependencies
```

**Not committed (too large or external):**
- `outputs_*/` — computed features, RDMs, searchlight results, figures
- `data/things_images/` — stimulus images (download from OSF `jum2f`, password `things4all`)
- fMRI betas — provided by Galella et al. via the THINGS-fMRI dataset (cluster or external drive)

---

## Setup

```bash
python3 -m venv venv_cortex
source venv_cortex/bin/activate
pip install -r requirements.txt
```

Requires Python 3.10+. Key dependencies: `torch`, `torchvision`, `nilearn`, `nibabel`,
`ripser`, `persim`, `scikit-learn`, `scipy`, `pandas`, `numpy`, `matplotlib`.

---

## Running the surface pipeline (cluster)

```bash
# Submit SLURM job (k=50 main + k=100 robustness check)
sbatch cluster/slurm_v2_surface_geodesic.sh

# Generate figures after job completes
python scripts/figures_v2_surface.py --k 50
python scripts/figures_v2_surface.py --k 100
```

THINGS-fMRI data path on cluster: `/work/dldevel/galella/datasets/THINGS-fMRI`
(READ-ONLY — never modify Santi's data).

---

## Main scripts

| Script | Purpose |
|--------|---------|
| `run_v2_surface_pipeline.py` | Full surface pipeline: vol→surf projection, geodesic searchlight (RSA+CKA), Glasser areal aggregation, TDA-RDMs |
| `figures_v2_surface.py` | Best-layer cortex maps and summary CSV |
| `compute_tda_v2_glasser.py` | TDA (H0/H1/H2, Wasserstein) on Glasser-area RDMs |
| `extract_resnet18_features.py` | Feature extraction for ResNet-50 (and ResNet-18) |
| `run_resnet50_pipeline.py` | Volume-based ROI pipeline (legacy) |

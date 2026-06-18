"""
Flatmap-Visualisierung der fMRI-Betas.

Funktioniert direkt mit den vorhandenen pycortex-Cache-Dateien ohne
externe Cortex-Masken. Die Vertex-Reihenfolge entspricht den surf-Betas:
  lh: Index 0 .. 138862
  rh: Index 138863 .. 279600

Koordinatensystem: Das SVG (1960.5 x 1024) ist um 90° gegen den Uhrzeiger
gedreht gespeichert:
  SVG-x (0..1960.5)  →  Bild-Zeilen  (0..2243), Skalierung 2244/1960.5
  SVG-y (0..1024)    →  Bild-Spalten (0..1023), Skalierung 1

Verwendung — einzelner Trial:
  venv_cortex/bin/python scripts/visualize_flatmap.py \
      --subject S1 --trial 0

ROI-Overlay hinzufügen:
      --roi LOC

Mehrere Trials als Durchschnitt:
      --trials 0 1 2 3 4

Ausgabe als PNG:
      --output outputs/figures/flatmap_S1_trial0.png
"""

import argparse
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import scipy.sparse as sparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BETAS_ROOT   = Path("/Volumes/Sonstige Backups/Data/processed")
DB_ROOT      = Path("/Volumes/Sonstige Backups/Data/db")

ROI_COLORS = {
    "V1":  "#e41a1c",
    "hV4": "#ff7f00",
    "LOC": "#4daf4a",
    "IT":  "#377eb8",
}

# SVG-Gruppen-IDs, die jede Analyse-ROI approximieren.
# Mehrere Einträge werden kombiniert gezeichnet.
ROI_SVG_MAP = {
    "V1":  ["V1"],
    "hV4": ["V3"],       # V4 hat keine Pfade; V3 ist die angrenzende frühe visuelle Area
    "LOC": ["TOS"],      # TOS (Transversale Okzipitalfurche) ist der Kern des LOC
    "IT":  ["FFA", "EBA"],  # FFA + EBA sind IT-Subbereiche
}

N_LH = 138863


# ---------------------------------------------------------------------------
# Cache / curvature / beta loaders
# ---------------------------------------------------------------------------

def load_flatmap_cache(subject):
    cache = DB_ROOT / subject / "cache"
    fv = np.load(cache / "flatverts_1024.npz")
    flatverts = sparse.csr_matrix(
        (fv["data"], fv["indices"], fv["indptr"]),
        shape=tuple(fv["shape"]),
    )
    flatmask = np.load(cache / "flatmask_1024.npz")["mask"]
    return flatverts, flatmask


def load_curvature(subject):
    curv = np.load(DB_ROOT / subject / "surface-info" / "curvature.npz")
    return np.concatenate([curv["left"], curv["right"]])


def load_betas(subject, trial_indices):
    lh = np.load(BETAS_ROOT / subject / "betas/surf" / f"{subject}_betas_surf_lh.npy",
                 mmap_mode="r")
    rh = np.load(BETAS_ROOT / subject / "betas/surf" / f"{subject}_betas_surf_rh.npy",
                 mmap_mode="r")
    lh_mean = lh[trial_indices].mean(axis=0)
    rh_mean = rh[trial_indices].mean(axis=0)
    return np.concatenate([lh_mean, rh_mean])


def vertex_to_flatmap(data, flatverts, flatmask):
    """Mappt Vertex-Daten (279601,) auf 2D-Flatmap-Bild."""
    pixel_values = flatverts.dot(data)
    img = np.full(flatmask.shape, np.nan)
    img[flatmask] = pixel_values
    return img


# ---------------------------------------------------------------------------
# SVG-basierte ROI-Konturen
# ---------------------------------------------------------------------------

def _parse_svg_path_d(d, n_bezier=20):
    """
    Parst den SVG-Pfad-String 'd' in eine Liste von numpy-Arrays [(N,2), ...].
    Jedes Array ist ein zusammenhängendes Liniensegment in SVG-Koordinaten.
    Unterstützte Befehle: M, m, L, l, C, c, Z, z.
    """
    tokens = re.findall(
        r'[MmLlCcZz]|[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?', d
    )

    segments = []
    cur_seg = []
    cx = cy = 0.0
    sx = sy = 0.0  # Startpunkt des aktuellen Teilpfades (für Z)
    cmd = None
    i = 0

    def _flush():
        nonlocal cur_seg
        if cur_seg:
            segments.append(np.array(cur_seg))
            cur_seg = []

    while i < len(tokens):
        tok = tokens[i]
        if re.fullmatch(r'[MmLlCcZz]', tok):
            cmd = tok
            i += 1
            continue

        if cmd == 'M':
            _flush()
            cx, cy = float(tokens[i]), float(tokens[i + 1])
            i += 2
            sx, sy = cx, cy
            cur_seg.append([cx, cy])
            cmd = 'L'
        elif cmd == 'm':
            _flush()
            cx += float(tokens[i])
            cy += float(tokens[i + 1])
            i += 2
            sx, sy = cx, cy
            cur_seg.append([cx, cy])
            cmd = 'l'
        elif cmd == 'L':
            cx, cy = float(tokens[i]), float(tokens[i + 1])
            i += 2
            cur_seg.append([cx, cy])
        elif cmd == 'l':
            cx += float(tokens[i])
            cy += float(tokens[i + 1])
            i += 2
            cur_seg.append([cx, cy])
        elif cmd in ('C', 'c'):
            x1 = cx + float(tokens[i])     if cmd == 'c' else float(tokens[i])
            y1 = cy + float(tokens[i + 1]) if cmd == 'c' else float(tokens[i + 1])
            x2 = cx + float(tokens[i + 2]) if cmd == 'c' else float(tokens[i + 2])
            y2 = cy + float(tokens[i + 3]) if cmd == 'c' else float(tokens[i + 3])
            ex = cx + float(tokens[i + 4]) if cmd == 'c' else float(tokens[i + 4])
            ey = cy + float(tokens[i + 5]) if cmd == 'c' else float(tokens[i + 5])
            i += 6
            ts = np.linspace(0, 1, n_bezier + 1)[1:]
            p0x, p0y = cx, cy
            for t in ts:
                mt = 1 - t
                bx = mt**3 * p0x + 3 * mt**2 * t * x1 + 3 * mt * t**2 * x2 + t**3 * ex
                by = mt**3 * p0y + 3 * mt**2 * t * y1 + 3 * mt * t**2 * y2 + t**3 * ey
                cur_seg.append([bx, by])
            cx, cy = ex, ey
        elif cmd in ('Z', 'z'):
            cur_seg.append([sx, sy])
            _flush()
            cx, cy = sx, sy
        else:
            i += 1

    _flush()
    return segments


def load_roi_svg_paths(subject, roi):
    """
    Gibt die SVG-Pfadsegmente für eine ROI zurück.
    Rückgabe: Liste von numpy-Arrays (N,2) in SVG-Koordinaten.
    """
    svg_file = DB_ROOT / subject / "overlays.svg"
    tree = ET.parse(svg_file)
    root = tree.getroot()

    paths = []
    for g in root.iter("{http://www.w3.org/2000/svg}g"):
        if g.attrib.get("id") == roi:
            for path_el in g.findall("{http://www.w3.org/2000/svg}path"):
                d = path_el.attrib.get("d", "")
                if d:
                    paths.extend(_parse_svg_path_d(d))
    return paths


def draw_roi_contour(ax, subject, roi, flatmask, color, label):
    """
    Zeichnet ROI-Kontur aus overlays.svg.

    Da nicht alle Analyse-ROIs direkt als SVG-Gruppe vorhanden sind, werden
    über ROI_SVG_MAP Approximationen verwendet (z.B. LOC → TOS, IT → FFA+EBA).

    Koordinatentransformation (overlays.svg ist 90° gedreht gespeichert):
        SVG x (0..svg_w) → Bild-Zeile  → matplotlib y-Achse
        SVG y (0..svg_h) → Bild-Spalte → matplotlib x-Achse
    """
    svg_file = DB_ROOT / subject / "overlays.svg"
    tree = ET.parse(svg_file)
    root = tree.getroot()
    svg_w = float(root.attrib.get("width",  "1960.49635354"))
    svg_h = float(root.attrib.get("height", "1024.0"))

    img_rows, img_cols = flatmask.shape
    scale_row = img_rows / svg_w   # SVG-x  →  Bild-Zeile
    scale_col = img_cols / svg_h   # SVG-y  →  Bild-Spalte

    svg_rois = ROI_SVG_MAP.get(roi, [roi])
    all_segments = []
    for svg_roi in svg_rois:
        segs = load_roi_svg_paths(subject, svg_roi)
        all_segments.extend(segs)

    if not all_segments:
        print(f"  Warnung: Keine SVG-Pfade für ROI '{roi}' "
              f"(SVG-Gruppen: {svg_rois}).")
        return

    first = True
    for seg in all_segments:
        xs_mpl = seg[:, 1] * scale_col   # SVG-y → Bild-Spalte → matplotlib x
        ys_mpl = seg[:, 0] * scale_row   # SVG-x → Bild-Zeile  → matplotlib y
        lbl = label if first else None
        ax.plot(xs_mpl, ys_mpl, color=color, linewidth=1.5, alpha=0.9, label=lbl)
        first = False


# ---------------------------------------------------------------------------
# Hauptprogramm
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject",  default="S1")
    parser.add_argument("--trial",    type=int, default=None,
                        help="Einzelner Trial-Index")
    parser.add_argument("--trials",   type=int, nargs="+", default=None,
                        help="Mehrere Trial-Indizes (werden gemittelt)")
    parser.add_argument("--roi",      type=str, nargs="*", default=None,
                        help="ROI(s) als Overlay (z.B. --roi V1 LOC)")
    parser.add_argument("--all-rois", action="store_true",
                        help="Alle vier ROIs als Overlay zeigen")
    parser.add_argument("--vmin",     type=float, default=None,
                        help="Colorbar-Minimum (Standard: auto ±3σ)")
    parser.add_argument("--vmax",     type=float, default=None,
                        help="Colorbar-Maximum (Standard: auto ±3σ)")
    parser.add_argument("--output",   default=None)
    args = parser.parse_args()

    subject = args.subject

    if args.trial is not None:
        trial_indices = [args.trial]
    elif args.trials is not None:
        trial_indices = args.trials
    else:
        trial_indices = [0]

    print(f"Subject: {subject}, Trials: {trial_indices}")

    flatverts, flatmask = load_flatmap_cache(subject)
    curvature = load_curvature(subject)
    betas     = load_betas(subject, trial_indices)

    curv_img = vertex_to_flatmap(curvature, flatverts, flatmask)
    beta_img = vertex_to_flatmap(betas,     flatverts, flatmask)

    # Auto-scale: ±3 Standardabweichungen der sichtbaren Beta-Werte
    finite_betas = beta_img[np.isfinite(beta_img)]
    sigma = finite_betas.std() if len(finite_betas) > 0 else 1.0
    vmin = args.vmin if args.vmin is not None else -3 * sigma
    vmax = args.vmax if args.vmax is not None else  3 * sigma
    print(f"Beta σ={sigma:.4f}  →  vmin={vmin:.4f}, vmax={vmax:.4f}")

    fig, ax = plt.subplots(figsize=(10, 14))

    curv_norm = np.where(np.isnan(curv_img), np.nan,
                         np.clip(curv_img, -1, 1) * 0.5 + 0.5)
    ax.imshow(curv_norm, cmap="gray", vmin=0, vmax=1,
              origin="upper", aspect="equal", interpolation="nearest")

    beta_rgba = plt.cm.RdBu_r(
        plt.Normalize(vmin=vmin, vmax=vmax)(
            np.where(np.isnan(beta_img), 0, beta_img)
        )
    )
    beta_rgba[..., 3] = np.where(np.isnan(beta_img), 0, 0.85)
    ax.imshow(beta_rgba, origin="upper", aspect="equal", interpolation="nearest")

    rois_to_draw = []
    if args.all_rois:
        rois_to_draw = list(ROI_COLORS.keys())
    elif args.roi:
        rois_to_draw = args.roi

    for roi in rois_to_draw:
        color = ROI_COLORS.get(roi, "white")
        draw_roi_contour(ax, subject, roi, flatmask, color, roi)

    if rois_to_draw:
        ax.legend(loc="lower right", fontsize=10, framealpha=0.8)

    sm = plt.cm.ScalarMappable(
        cmap="RdBu_r",
        norm=mcolors.Normalize(vmin=vmin, vmax=vmax)
    )
    plt.colorbar(sm, ax=ax, label="Beta estimate (a.u.)", shrink=0.5, pad=0.02)

    trial_label = (f"Trial {trial_indices[0]}" if len(trial_indices) == 1
                   else f"Ø Trials {trial_indices[0]}–{trial_indices[-1]}")
    ax.set_title(f"{subject} — {trial_label}", fontsize=14)
    ax.axis("off")

    output = args.output or f"outputs/figures/flatmap_{subject}_trial{trial_indices[0]}.png"
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Gespeichert: {output}")
    plt.close(fig)


if __name__ == "__main__":
    main()

"""Liest die ResNet-50-Ergebnis-CSVs und gibt fertige LaTeX-Tabellenzeilen
sowie die zentralen Kennzahlen für die Bachelorarbeit aus.

Nach dem Cluster-Lauf ausführen:
  .venv/bin/python scripts/fill_thesis_resnet50.py

Die Ausgabe lässt sich direkt in ba_thesis_resnet50.tex einsetzen
(RSA-, CKA-, Noise-Ceiling- und TDA-Tabellen sowie die Peak-Schicht-Sätze
für Diskussion und Fazit).
"""

import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
R50 = PROJECT_ROOT / "outputs_resnet50" / "results"
NC_CSV = PROJECT_ROOT / "outputs" / "results" / "noise_ceiling.csv"
LAYERS = ["conv1", "layer1", "layer2", "layer3", "layer4", "fc"]
ROIS = ["V1", "hV4", "LOC", "IT"]


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


def load(path):
    return list(csv.DictReader(open(path, encoding="utf-8")))


def fmt(v, dec=3, sign=False):
    if v != v:  # nan
        return "--"
    s = f"{v:+.{dec}f}" if sign else f"{v:.{dec}f}"
    return s.replace(".", "{,}")


def aggregate(rows, value_key, filt=None):
    agg = {roi: {l: [] for l in LAYERS} for roi in ROIS}
    for r in rows:
        if filt and not filt(r):
            continue
        if r.get("roi") in ROIS and r.get("layer") in LAYERS:
            try:
                agg[r["roi"]][r["layer"]].append(float(r[value_key]))
            except (ValueError, KeyError):
                pass
    return {roi: {l: mean(agg[roi][l]) for l in LAYERS} for roi in ROIS}


def print_table(title, vals, sign=False, best="max"):
    print(f"\n% ===== {title} =====")
    for roi in ROIS:
        cells = []
        order = [vals[roi][l] for l in LAYERS]
        valid = [v for v in order if v == v]
        peak = (max if best == "max" else min)(valid) if valid else None
        for l in LAYERS:
            v = vals[roi][l]
            cell = fmt(v, sign=sign)
            if peak is not None and v == peak:
                cell = f"\\textbf{{{cell}}}"
            cells.append(cell)
        print(f"{roi:4s} & " + " & ".join(cells) + r" \\")


def main():
    geom = load(R50 / "ann_resnet50_all_subjects_roi_summary.csv")
    rsa = aggregate(geom, "spearman_r")
    cka = aggregate(geom, "cka")
    print_table("RSA: Spearman-rho (Tab. tab:rsa)", rsa, sign=True, best="max")
    print_table("CKA (Tab. tab:cka)", cka, sign=False, best="max")

    # Peak-Schichten RSA für Diskussion/Fazit
    print("\n% ----- RSA Peak-Schichten -----")
    for roi in ROIS:
        order = [(l, rsa[roi][l]) for l in LAYERS if rsa[roi][l] == rsa[roi][l]]
        if order:
            l, v = max(order, key=lambda t: t[1])
            print(f"%   {roi}: Peak in {l} (rho = {fmt(v, sign=True)})")

    # Noise Ceiling (modellunabhängig) + relative Leistung
    if NC_CSV.exists():
        nc = load(NC_CSV)
        nc_data = {roi: {"upper": [], "lower": []} for roi in ROIS}
        for r in nc:
            if r.get("roi") in ROIS and r.get("subject") in ("S1", "S2", "S3"):
                nc_data[r["roi"]]["upper"].append(float(r["nc_upper"]))
                nc_data[r["roi"]]["lower"].append(float(r["nc_lower"]))
        print("\n% ===== Noise Ceiling + relative Leistung (Tab. tab:nc) =====")
        for roi in ROIS:
            up = mean(nc_data[roi]["upper"])
            lo = mean(nc_data[roi]["lower"])
            order = [(l, rsa[roi][l]) for l in LAYERS if rsa[roi][l] == rsa[roi][l]]
            pl, pv = max(order, key=lambda t: t[1]) if order else ("--", float("nan"))
            rel = f"{100*pv/up:.0f}" if up and up == up else "--"
            print(f"{roi:4s} & {fmt(up)} & {fmt(lo)} & {fmt(pv, sign=False)}\\,(\\code{{{pl}}}) & {rel}\\,\\% \\\\")

    # TDA
    tda_path = R50 / "tda_resnet50_all_subjects_roi_summary.csv"
    if tda_path.exists():
        tda = load(tda_path)
        for hdim, label in [(0, "H0 (Tab. tab:h0)"), (1, "H1 (Tab. tab:h1)")]:
            vals = aggregate(tda, "wasserstein_dist",
                             filt=lambda r, h=hdim: int(r["homology_dim"]) == h)
            print_table(f"TDA {label} — Minimum fett", vals, sign=False, best="min")
    print("\n% Fertig. Werte in ba_thesis_resnet50.tex einsetzen.")


if __name__ == "__main__":
    main()

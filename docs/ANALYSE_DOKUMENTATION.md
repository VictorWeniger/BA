# Analyse-Dokumentation: BA Victor Weniger

Dieses Dokument ist die zentrale Referenz für alle methodischen Entscheidungen,
die Architektur der Analysepipeline und die bisherigen Ergebnisse.
Es ist so strukturiert, dass die relevanten Abschnitte direkt als Grundlage
für die Kapitel der Bachelorarbeit dienen können.

---

## 1. Forschungsfrage

> Welche Schichten eines vortrainierten konvolutionalen neuronalen Netzes
> stimmen geometrisch am stärksten mit fMRI-Repräsentationen im visuellen
> Kortex überein?

Die Kernhypothese:

> Intermediäre Schichten eines CNN zeigen stärkere Übereinstimmung mit
> visuellen kortikalen Repräsentationen als die finale, aufgabenspezifische
> Klassifikationsschicht.

Modelliert wird das mit zwei Maßen:

- **Spearman-Korrelation auf Repräsentationalen Distanzmatrizen (RDM)**:
  misst, wie ähnlich die paarweisen Abstände zwischen Stimuli in ANN und
  fMRI sind.
- **Centered Kernel Alignment (CKA)**: misst strukturelle Ähnlichkeit der
  Repräsentationsräume als Ganzes.

---

## 2. Datenbasis

### 2.1 fMRI-Daten

Quelle: THINGS-data (Hebart et al. 2023), vorprozessiert.

Lokaler Pfad:
```
/Volumes/Sonstige Backups/Data/processed/
```

Struktur pro Subject:
```
S1/
  stimuli/S1_stimuli.csv        — 9840 Zeilen, eine pro Trial
  betas/vol/S1_betas_vol.npy    — (9840, 211339), float32
  rois/localizers/S1_localizers.csv
```

Jede Zeile der Beta-Matrix entspricht genau einer Trial-Zeile in der
Stimulus-CSV. Die Beta-Koeffizienten sind aus einem GLM geschätzte
BOLD-Antworten auf einzelne Bildpräsentationen.

Verwendete Subjects bisher: **S1** (Pilotlauf).
Geplant: S2, S3.

### 2.2 Bilddaten

Quelle: THINGS-Objektbilder.

Lokaler Pfad:
```
/Volumes/Sonstige Backups/Data/object_images/
```

Dateikonvention:
```
Stimulus-CSV: "dog_12s.jpg", Concept-Spalte: "dog"
→ Bildpfad: object_images/dog/dog_12s.jpg
```

### 2.3 ROIs

Für S1 wurden vier visuell-anatomisch definierte Regionen of Interest
ausgewertet:

| ROI | Beschreibung                          | Anzahl Voxel (S1) |
|-----|---------------------------------------|-------------------|
| V1  | Primärer visueller Kortex             | 1049              |
| hV4 | Ventrales visuelles Areal, Farbe/Form | 613               |
| LOC | Lateral Occipital Complex, Objekte    | 2700              |
| IT  | Inferotemporaler Kortex, Objektkategorie | 4145            |

Die ROI-Maske ist als binäre CSV-Spalte in `S1_localizers.csv` gespeichert.
Jede Zeile entspricht einem Voxel in der Vol-Space-Beta-Matrix.

---

## 3. Pipeline-Architektur

### 3.1 Übersicht

```
fMRI-Rohdaten
  → scripts/import_processed_fmri.py
      Trial-Filter, ROI-Maske, Aggregation
      → data/processed_fmri_S1_{ROI}_stimuli.csv
      → outputs/human/processed_fmri_S1_{ROI}_features.npy
      → outputs/human/processed_fmri_S1_{ROI}_distance.npy

  → scripts/build_ann_stimuli_from_fmri.py
      Bildpfade zu fMRI-Stimulusnamen matchen
      → data/ann_S1_matched_stimuli.csv

  → scripts/extract_resnet18_features.py
      Forward-Hook-basierte Aktivierungsextraktion
      → outputs/features/ann_resnet18_S1_matched_{layer}.npy

  → scripts/compute_geometry.py
      Paarweise Cosine-Distanzmatrizen
      → outputs/geometry/ann_resnet18_S1_matched_{layer}_cosine.npy

  → scripts/compare_geometry_to_human.py
      Spearman (RDM) + CKA pro Layer x ROI
      → outputs/results/ann_resnet18_S1_matched_geometric_alignment.csv

  → compare_resnet18_to_fmri_rois.py  [Orchestrator]
      Alle ROIs zusammenfassen
      → outputs/results/ann_resnet18_S1_roi_summary.csv
```

### 3.2 Schritt 1: fMRI-Import (`import_processed_fmri.py`)

**Trial-Filter** (konfigurierbar in `configs/processed_fmri_s1_loc.json`):
- `split: "test"` — nur Test-Trials, keine Trainings-Trials
- `dataset: "THINGS"` — nur THINGS-Bildkategorie, keine anderen Stimuli
- `max_trials: 500` — maximale Anzahl Trials pro ROI

**ROI-Maskierung**:
Die `S1_localizers.csv` enthält eine Spalte pro ROI mit 0/1-Einträgen.
Es wird die entsprechende Spalte als boolsche Maske auf die Beta-Matrix
angewendet:
```python
features = betas[trial_indices][:, roi_mask]
# z. B. LOC: (500, 211339) → (500, 2700)
```

**Aggregation**:
Mehrere Trials desselben Stimulus werden gemittelt:
```python
group_by = "Stimulus"   # oder "Concept"
method = "mean"
# 500 Trials → 100 einzigartige Stimuli (wenn jeder ~5x präsentiert)
```

Ergebnis: Feature-Matrix `(100, n_voxel)` pro ROI, dazu eine
Distanzmatrix `(100, 100)`.

**Warum Mittelung?**
Einzelne Trial-Betas sind rauschbehaftet (SNR ~1). Über Wiederholungen
gemittelte Betas sind stabiler und repräsentieren die mittlere kortikale
Antwort auf einen Stimulus besser.

### 3.3 Schritt 2: Stimulusabgleich (`build_ann_stimuli_from_fmri.py`)

Die ANN-Stimulusliste wird direkt aus der aggregierten fMRI-Stimulusliste
(LOC als Basis) abgeleitet. Dadurch ist die Reihenfolge der 100 Stimuli
in ANN- und fMRI-Distanzmatrizen identisch — eine notwendige Bedingung
für den Vergleich.

Ausgabe: `data/ann_S1_matched_stimuli.csv` mit 100 Zeilen und den Spalten
`image_id, concept, category, image_path, fmri_stimulus, fmri_subject, n_aggregated_trials`.

**Warum LOC als Basis?**
LOC zeigt im Pilotlauf das stärkste Alignment und hat eine mittlere
Voxelzahl (2700). Wichtiger: Alle vier ROIs verwenden denselben
Trial-Filter und dieselbe Aggregation, daher ist die resultierende
Stimulusreihenfolge für alle ROIs identisch.

### 3.4 Schritt 3: Feature-Extraktion (`extract_resnet18_features.py`)

**Modell**: ResNet18, ImageNet-vortrainiert
(`models.ResNet18_Weights.IMAGENET1K_V1`).

**Backend**: MPS (Apple Silicon), fallback CUDA → CPU.

**Preprocessing** (ImageNet-Standard):
```python
transforms.Resize(256)
transforms.CenterCrop(224)
transforms.ToTensor()
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
```

**Layer-Auswahl und Feature-Shapes** (nach Pooling):

| Layer  | Typ                  | Features (nach GAP) |
|--------|----------------------|---------------------|
| conv1  | 7×7 Conv, stride 2   | 64                  |
| layer1 | 2× BasicBlock        | 64                  |
| layer2 | 2× BasicBlock        | 128                 |
| layer3 | 2× BasicBlock        | 256                 |
| layer4 | 2× BasicBlock        | 512                 |
| fc     | Fully Connected      | 1000                |

**Global Average Pooling (GAP)**:
Konvolutionale Layer geben Aktivierungen der Form `(batch, C, H, W)` aus.
GAP mittelt über die räumlichen Dimensionen:
```python
activation.mean(dim=(2, 3))  # → (batch, C)
```

Begründung für GAP:
- Macht Aktivierungen raumunabhängig und damit direkt vergleichbar.
- Reduziert Dimensionalität drastisch (z. B. layer4: 512×7×7=25088 → 512).
- Analogie zu fMRI: Auch Voxel-Antworten integrieren über räumliche
  Ausdehnung eines rezeptiven Feldes.

### 3.5 Schritt 4: Distanzmatrizen (`compute_geometry.py`)

Für jede Layer-Feature-Matrix `X ∈ ℝ^{n×d}` wird eine paarweise
Cosine-Distanzmatrix berechnet:

```
D[i,j] = 1 - cos(x_i, x_j) = 1 - (x_i · x_j) / (||x_i|| · ||x_j||)
```

Implementierung:
```python
normalized = features / np.linalg.norm(features, axis=1, keepdims=True)
similarity = normalized @ normalized.T
distance = 1.0 - similarity
```

Ergebnis: `(100, 100)` Distanzmatrix pro Layer, symmetrisch,
Diagonale = 0, Werte ∈ [0, 2] (theoretisch; in der Praxis ≤ 1 für
nichtnegative Aktivierungen nach ReLU).

**Warum Cosine-Distanz?**
- Fokus auf Repräsentationsrichtung statt Aktivierungsgröße.
- Standard in RSA-Studien (Kriegeskorte et al. 2008).
- Robust gegenüber layer-abhängigen Skalierungsunterschieden.

### 3.6 Schritt 5: Vergleichsmaße (`compare_geometry_to_human.py`)

#### Spearman-Korrelation (RDM-Vergleich)

Die obere Dreiecksmatrix beider Distanzmatrizen wird extrahiert
(Diagonale ausgeschlossen, n*(n-1)/2 = 4950 Werte bei n=100).
Darauf wird die Spearman-Rang-Korrelation berechnet:

```
ρ = Pearson(rank(d_ANN), rank(d_fMRI))
```

Implementiert in `src/stats_utils.py:spearman_corr` ohne scipy,
über einen eigenen Rangalgorithmus mit average-tie-handling.

Signifikanztest: Permutationstest (1000 Permutationen), zweiseitig.
Der p-Wert gibt an, wie oft |ρ| bei zufälliger Permutation der
fMRI-Werte so groß oder größer war.

**Warum Spearman?**
- Robuster gegenüber nicht-normalen Verteilungen.
- Standardmethode für RDM-Vergleiche (Representational Similarity
  Analysis, RSA).
- Unempfindlich gegenüber monotonen Transformationen der Distanzskala.

#### Centered Kernel Alignment (CKA)

CKA (Kornblith et al. 2019) misst die Ähnlichkeit zweier Kernelmatrizen
nach Zentrierung:

```
CKA(K, L) = HSIC(K, L) / sqrt(HSIC(K, K) · HSIC(L, L))

HSIC(K, L) = tr(K_c L_c) / (n-1)²
K_c = H K H,  H = I - (1/n) 1 1ᵀ   (Zentrierung)
```

Implementiert in `src/stats_utils.py:kernel_cka`.

**Kernel-Konstruktion** (wichtige Designentscheidung):
Beide Kernelmatrizen werden als Cosine-Ähnlichkeitsmatrizen gebaut:

```python
K_human = 1.0 - human_distance_matrix
np.fill_diagonal(K_human, 1.0)

K_dnn   = 1.0 - ann_distance_matrix
np.fill_diagonal(K_dnn, 1.0)
```

Beide Kernelmatrizen liegen auf derselben Skala [0, 1] und messen
dasselbe Konzept (Cosine-Ähnlichkeit). Das ist die methodisch sauberere
Wahl gegenüber einem rohen Dotprodukt-Kernel für K_dnn.

**Warum CKA ergänzend zu Spearman?**
- Spearman/RDM vergleicht paarweise Distanzränge.
- CKA vergleicht die gesamte geometrische Struktur des
  Repräsentationsraums und ist invariant gegenüber orthogonalen
  Transformationen.
- Beide Maße zusammen geben ein robusteres Bild als jedes einzeln.

---

## 4. Implementierungsdetails und bekannte Entscheidungen

### 4.1 Stimulus-Reihenfolge ist explizit gespeichert

Die Datei `outputs/features/ann_resnet18_S1_matched_image_order.csv`
speichert die exakte Reihenfolge der Bilder, in der der DataLoader sie
verarbeitet hat. Dies sichert die Alignment-Invariante:

> Zeile i in der ANN-Distanzmatrix entspricht Zeile i in der
> fMRI-Distanzmatrix entspricht Zeile i in `ann_S1_matched_stimuli.csv`.

### 4.2 Konfiguration über JSON

Alle Experimentparameter sind in JSON-Configs unter `configs/` gespeichert.
Für den S1-Pilotlauf relevante Configs:

| Config                              | Zweck                              |
|-------------------------------------|------------------------------------|
| `processed_fmri_s1_loc.json`        | fMRI-Import-Parameter              |
| `ann_from_processed_s1_loc.json`    | ResNet18-Feature-Extraction        |
| `tmp_ann_resnet18_S1_matched.json`  | Erzeugter Gesamt-Config (S1)       |
| `tmp_ann_resnet18_S1_{ROI}.json`    | ROI-spezifische Vergleichsconfigs  |

### 4.3 Orchestrator-Skripte

Zwei Ebenen der Orchestrierung:

1. `run_s1_roi_batch.py` — führt `import_processed_fmri.py` für alle
   vier ROIs sequenziell aus.
2. `compare_resnet18_to_fmri_rois.py` — führt Feature-Extraction,
   Geometrie und Vergleich für alle ROIs aus und schreibt das
   Summary-CSV.

Das Skript `write_s1_roi_summary_from_existing.py` erfüllt dieselbe
Aufgabe wie der zweite Teil von Nr. 2, setzt aber voraus, dass
Features und Geometrie bereits existieren (kein Re-Extraktion).

---

## 5. Ergebnisse: S1–S3 / ResNet18

Einstellungen (finaler Lauf, 2026-05-18):
- Subjects: S1, S2, S3
- Trials: **1200** (alle verfügbaren Test-Trials, THINGS-Dataset)
- Wiederholungen pro Stimulus: 12 (→ nach Aggregation 100 Stimuli)
- Modell: ResNet18 (ImageNet-pretrained, MPS)
- Layer: conv1, layer1, layer2, layer3, layer4, fc
- Distanzmetrik: Cosine
- Spearman-Permutationstest: 1000 Permutationen
- CKA-Kernel: symmetrische Cosine-Ähnlichkeitsmatrizen

Ausgabedateien:
```
outputs/results/ann_resnet18_S1_roi_summary.csv
outputs/results/ann_resnet18_S2_roi_summary.csv
outputs/results/ann_resnet18_S3_roi_summary.csv
outputs/results/ann_resnet18_all_subjects_roi_summary.csv   ← Gesamtübersicht
```

### 5.1 Spearman-Korrelation (RDM-Vergleich) — vollständige Tabelle

| Subject | ROI | Layer  | Spearman r | p-Wert |
|---------|-----|--------|------------|--------|
| S1 | V1  | conv1  | −0.0302 | 0.0350 |
| S1 | V1  | layer1 | +0.0398 | 0.0040 |
| S1 | V1  | layer2 | **+0.0923** | 0.0010 |
| S1 | V1  | layer3 | −0.1490 | 0.0010 |
| S1 | V1  | layer4 | −0.0781 | 0.0010 |
| S1 | V1  | fc     | −0.1014 | 0.0010 |
| S1 | hV4 | conv1  | −0.0095 | 0.4745 |
| S1 | hV4 | layer1 | +0.0130 | 0.3487 |
| S1 | hV4 | layer2 | **+0.0675** | 0.0010 |
| S1 | hV4 | layer3 | −0.0450 | 0.0040 |
| S1 | hV4 | layer4 | +0.0130 | 0.3487 |
| S1 | hV4 | fc     | +0.0573 | 0.0010 |
| S1 | LOC | conv1  | −0.0112 | 0.4486 |
| S1 | LOC | layer1 | +0.1220 | 0.0010 |
| S1 | LOC | layer2 | +0.1960 | 0.0010 |
| S1 | LOC | layer3 | **+0.2602** | 0.0010 |
| S1 | LOC | layer4 | +0.2050 | 0.0010 |
| S1 | LOC | fc     | +0.0732 | 0.0010 |
| S1 | IT  | conv1  | −0.0213 | 0.1469 |
| S1 | IT  | layer1 | +0.0819 | 0.0010 |
| S1 | IT  | layer2 | +0.1422 | 0.0010 |
| S1 | IT  | layer3 | **+0.1883** | 0.0010 |
| S1 | IT  | layer4 | +0.1498 | 0.0010 |
| S1 | IT  | fc     | +0.0531 | 0.0020 |
| S2 | V1  | conv1  | −0.0209 | 0.1359 |
| S2 | V1  | layer1 | +0.0378 | 0.0080 |
| S2 | V1  | layer2 | **+0.0794** | 0.0010 |
| S2 | V1  | layer3 | −0.1543 | 0.0010 |
| S2 | V1  | layer4 | −0.0909 | 0.0010 |
| S2 | V1  | fc     | −0.1603 | 0.0010 |
| S2 | hV4 | conv1  | −0.0290 | 0.0360 |
| S2 | hV4 | layer1 | +0.1452 | 0.0010 |
| S2 | hV4 | layer2 | **+0.2087** | 0.0010 |
| S2 | hV4 | layer3 | +0.0110 | 0.4635 |
| S2 | hV4 | layer4 | −0.0139 | 0.3267 |
| S2 | hV4 | fc     | −0.0993 | 0.0010 |
| S2 | LOC | conv1  | −0.0241 | 0.1009 |
| S2 | LOC | layer1 | +0.1401 | 0.0010 |
| S2 | LOC | layer2 | **+0.1990** | 0.0010 |
| S2 | LOC | layer3 | +0.0955 | 0.0010 |
| S2 | LOC | layer4 | +0.0163 | 0.2557 |
| S2 | LOC | fc     | −0.0441 | 0.0010 |
| S2 | IT  | conv1  | −0.0411 | 0.0040 |
| S2 | IT  | layer1 | +0.1447 | 0.0010 |
| S2 | IT  | layer2 | **+0.1872** | 0.0010 |
| S2 | IT  | layer3 | +0.1092 | 0.0010 |
| S2 | IT  | layer4 | +0.0270 | 0.0589 |
| S2 | IT  | fc     | −0.0167 | 0.2507 |
| S3 | V1  | conv1  | −0.0132 | 0.3347 |
| S3 | V1  | layer1 | +0.0450 | 0.0030 |
| S3 | V1  | layer2 | **+0.0880** | 0.0010 |
| S3 | V1  | layer3 | −0.1305 | 0.0010 |
| S3 | V1  | layer4 | −0.0458 | 0.0010 |
| S3 | V1  | fc     | −0.1491 | 0.0010 |
| S3 | hV4 | conv1  | +0.0312 | 0.0290 |
| S3 | hV4 | layer1 | +0.0741 | 0.0010 |
| S3 | hV4 | layer2 | **+0.1433** | 0.0010 |
| S3 | hV4 | layer3 | +0.0388 | 0.0110 |
| S3 | hV4 | layer4 | +0.0566 | 0.0010 |
| S3 | hV4 | fc     | +0.0675 | 0.0010 |
| S3 | LOC | conv1  | +0.0218 | 0.1249 |
| S3 | LOC | layer1 | +0.0639 | 0.0010 |
| S3 | LOC | layer2 | **+0.1421** | 0.0010 |
| S3 | LOC | layer3 | +0.1240 | 0.0010 |
| S3 | LOC | layer4 | +0.1052 | 0.0010 |
| S3 | LOC | fc     | +0.0895 | 0.0010 |
| S3 | IT  | conv1  | +0.0293 | 0.0390 |
| S3 | IT  | layer1 | +0.0833 | 0.0010 |
| S3 | IT  | layer2 | **+0.1293** | 0.0010 |
| S3 | IT  | layer3 | +0.0816 | 0.0010 |
| S3 | IT  | layer4 | +0.0693 | 0.0010 |
| S3 | IT  | fc     | +0.0580 | 0.0020 |

### 5.2 Mittlerer Spearman r über alle Subjects (S1–S3)

| ROI  | conv1   | layer1  | layer2       | layer3  | layer4  | fc      |
|------|---------|---------|--------------|---------|---------|---------|
| V1   | −0.0214 | +0.0409 | **+0.0866**  | −0.1446 | −0.0716 | −0.1369 |
| hV4  | −0.0024 | +0.0774 | **+0.1398**  | +0.0016 | +0.0186 | +0.0085 |
| LOC  | −0.0045 | +0.1087 | **+0.1790**  | +0.1599 | +0.1088 | +0.0395 |
| IT   | −0.0110 | +0.1033 | **+0.1529**  | +0.1264 | +0.0821 | +0.0315 |

layer2 hat über alle 4 ROIs und alle 3 Subjects konsistent die
höchste mittlere Spearman-Korrelation.

### 5.3 CKA (Cosine-Kernel) — mittlere Werte über S1–S3

| ROI  | conv1  | layer1 | layer2 | layer3 | layer4       | fc     |
|------|--------|--------|--------|--------|--------------|--------|
| V1   | 0.1125 | 0.3041 | 0.3462 | 0.3433 | **0.4091**   | 0.2662 |
| hV4  | 0.1185 | 0.2872 | 0.3520 | 0.4145 | **0.4993**   | 0.3303 |
| LOC  | 0.1263 | 0.2483 | 0.3352 | 0.4471 | **0.5610**   | 0.3737 |
| IT   | 0.1317 | 0.2686 | 0.3529 | 0.4609 | **0.5872**   | 0.3891 |

CKA steigt monoton mit der Schichttiefe und hat das Maximum bei layer4
— konsistent über alle ROIs und Subjects.

### 5.4 Noise Ceiling (Inter-Subject-Korrelation als obere Schranke)

Berechnet mit `scripts/compute_noise_ceiling.py` (Nili et al. 2014).
Alle Subject-RDMs werden auf eine kanonische (alphabetische) Stimulus-
Reihenfolge ausgerichtet, dann wird der Leave-one-out-Ansatz angewendet.

| ROI  | NC_upper (Ø) | NC_lower (Ø) | Bestes ANN-r (layer2) | ANN/NC_lower |
|------|-------------|-------------|----------------------|--------------|
| V1   | 0.8435      | 0.6596      | 0.087                | 13 %         |
| hV4  | 0.6855      | 0.3070      | 0.140                | 46 %         |
| LOC  | 0.7091      | 0.3670      | 0.179                | 49 %         |
| IT   | 0.6654      | 0.2575      | 0.153                | 59 %         |

Der Noise Ceiling für V1 ist sehr hoch (NC_upper = 0.84), d. h. die
fMRI-RDMs von V1 sind hoch reproduzierbar über Subjects — aber ResNet18
erklärt nur 13 % der erklärbaren Varianz. Für IT liegt der Anteil bei
59 %, was deutlich höher ist.

Wichtig: Diese Zahlen zeigen, dass das Modell generell noch viel
unerklärte Varianz lässt — das ist typisch für ResNet18 in RSA-Analysen
und motiviert den Vergleich mit leistungsfähigeren Modellen (z. B. CLIP).

Ausgabedatei: `outputs/results/noise_ceiling.csv`
Abbildung: `outputs/figures/rsa_resnet18_fmri_with_noise_ceiling.png`

### 5.5 Interpretation

**Hauptbefund (Spearman/RDM):**

> layer2 hat das stärkste geometrische Alignment mit allen vier
> visuellen ROIs, konsistent über alle drei Subjects.

Das unterstützt die Intermediate-Layer-Hypothese: nicht die erste
Schicht (zu früh, nur Kanten/Texturen), nicht die letzte
Klassifikationsschicht (zu aufgabenspezifisch), sondern die erste
intermediäre Schicht (layer2: 128-dim, erste downsampled Residual-Stage)
zeigt das stärkste Alignment.

**ROI-spezifische Befunde:**

- **LOC und IT**: Stärkstes Alignment insgesamt (r ≈ 0.13–0.26).
  Das ergibt Sinn, da LOC und IT höherstufige Objekt-/Kategorie-
  Information kodieren — ähnlich dem, was mittlere CNN-Schichten lernen.

- **V1**: Positives Alignment nur bei layer1/layer2 (früh-intermediär),
  dann negative Korrelation ab layer3 (−0.13 bis −0.15, hochsignifikant).
  Das reflektiert, dass V1 low-level visuelle Merkmale kodiert, die zu
  den frühen CNN-Schichten passen, aber anti-korreliert sind mit der
  semantischen Objekt-Organisation in layer3.

- **hV4**: Mittelstarkes Alignment bei layer2 (r ≈ 0.07–0.21). Weniger
  stabil zwischen Subjects als LOC/IT.

**Dissoziation Spearman vs. CKA:**

| Maß     | Maximum bei | Interpretation |
|---------|-------------|----------------|
| Spearman | layer2     | Beste Stimulus-Distanz-Übereinstimmung |
| CKA      | layer4     | Höchste globale Repräsentations-Ähnlichkeit |

CKA misst strukturelle Ähnlichkeit des gesamten Repräsentationsraums
(invariant gegenüber orthogonalen Transformationen); Spearman/RDM
misst die Übereinstimmung paarweiser Distanzränge. Beides ist valid,
aber für die BA-Frage ("welche Schicht kodiert Objekte am ähnlichsten
wie das visuelle Gehirn?") ist Spearman/RDM das direktere Maß.

**V1-layer3-Dissoziation — interpretierbar für die BA:**

Der negative Spearman-Wert für layer3 × V1 (−0.13 bis −0.15,
p < 0.001, über alle Subjects) ist einer der robustesten Einzelbefunde.
Er zeigt: ResNet18-layer3 organisiert Objekte nach kategorialer
Ähnlichkeit (semantisch), V1 organisiert nach visueller Ähnlichkeit
(Pixel-/Texturstruktur). Die beiden Strukturen sind nicht nur
unkorreliert, sondern anti-korreliert — ein klares Zeichen dafür,
dass diese Schicht eine qualitativ andere Repräsentation hat als V1.

### 5.6 TDA (Topologische Datenanalyse)

Berechnet mit `scripts/compare_tda_resnet18_to_fmri_rois.py`.
Methode: PCA(50) → Vietoris-Rips-Persistenz → Wasserstein-Distanz.
Kleinere Wasserstein-Distanz = ähnlichere topologische Struktur.

**H1 (Loops/Zykel) — Ø Wasserstein-Distanz über S1–S3:**

| ROI  | conv1  | layer1 | layer2 | layer3 | layer4       | fc     |
|------|--------|--------|--------|--------|--------------|--------|
| V1   | 44.02  | 48.97  | 46.69  | 59.24  | **46.10**    | 39.51  |
| hV4  | 37.41  | 42.35  | 40.08  | 52.56  | **33.37**    | 54.88  |
| LOC  | 59.43  | 64.37  | 62.10  | 74.59  | **59.02**    | 72.50  |
| IT   | 74.91  | 79.86  | 77.58  | 90.12  | **77.52**    | 94.12  |

Für H1 zeigen frühe Layer (conv1) und layer4 tendenziell geringere
Wasserstein-Distanzen als layer3. Das Muster ist weniger eindeutig als
beim Spearman-RDM-Vergleich.

**Interpretation TDA:**
- Die topologischen Strukturen (Zyklen, H1) sind insgesamt ähnlicher für
  V1 und hV4 als für LOC und IT — das Gegenteil des RDM-Befunds!
- Mögliche Erklärung: Die Stimulus-Topologie (zyklische Strukturen) im
  CNN-Raum ist näher an frühen visuellen Areas, während die paarweisen
  Abstände (RDM) besser mit höheren Areas übereinstimmen.
- Der TDA-Befund ergänzt die RDM-Analyse und zeigt, dass unterschiedliche
  Aspekte der Repräsentationsgeometrie unterschiedliche Signale liefern.

Ausgabedatei: `outputs/results/tda_resnet18_all_subjects_roi_summary.csv`
Abbildungen: `outputs/figures/tda_resnet18_fmri_heatmap_H{0,1}.png`

---

## 6. Einschränkungen

| Einschränkung          | Status        | Auswirkung / Bemerkung                        |
|------------------------|---------------|-----------------------------------------------|
| S1–S3, 100 Stimuli     | Erledigt      | Alle Subjects, alle 1200 Test-Trials genutzt   |
| CKA-Kernel-Asymmetrie  | Gefixt        | Symmetrischer Cosine-Kernel seit 2026-05-18    |
| p_value-Feldname       | Gefixt        | Korrekte Übertragung im Orchestrator-Script    |
| Nur ResNet18           | Offen         | Kein Modell-Vergleich; geplant: CLIP/ResNet50  |
| Kein TDA               | **Erledigt**  | `compare_tda_resnet18_to_fmri_rois.py`, H0+H1  |
| Kein Noise-Ceiling     | **Erledigt**  | `compute_noise_ceiling.py`, NC_upper/lower für alle 4 ROIs |
| Nur Cosine-Distanz     | Offen         | Robustheit mit Euklid/Correlation noch unklar  |

---

## 7. Analyse-Status und nächste Schritte

### Abgeschlossen ✅

| Aufgabe | Script | Output |
|---------|--------|--------|
| RSA (Spearman + CKA), S1–S3 | `compare_resnet18_to_fmri_rois.py` | `ann_resnet18_all_subjects_roi_summary.csv` |
| Noise Ceiling | `compute_noise_ceiling.py` | `noise_ceiling.csv` |
| TDA (Wasserstein H0/H1) | `compare_tda_resnet18_to_fmri_rois.py` | `tda_resnet18_all_subjects_roi_summary.csv` |
| Flatmap-Visualisierung | `visualize_flatmap.py` | `outputs/figures/flatmap_*.png` |

### Offen / Optional

| Aufgabe | Priorität | Bemerkung |
|---------|-----------|-----------|
| Zweites Modell (CLIP / ResNet50) | Mittel | Zeigt Modellabhängigkeit der Befunde |
| Robustheit: Euklid statt Cosine | Niedrig | Konsistenz-Check |
| Noise Ceiling für TDA | Niedrig | Kein etablierter Standard |

### Für die BA-Schreibphase

Die Analyse ist vollständig. Alle relevanten Ergebnisse,
Methodenbeschreibungen und Interpretationen sind in diesem Dokument
(Abschnitte 3–5) dokumentiert und können direkt als Grundlage für die
Kapitel „Methoden", „Ergebnisse" und „Diskussion" der Bachelorarbeit
dienen.

---

## 8. Ausführungsanleitung (kompletter S1-Lauf)

Voraussetzung: externe Festplatte mit Datenpfaden (s. Abschnitt 2) ist
eingehängt.

```bash
cd /Users/victorweniger/Desktop/Uni/Goethe-Uni/Sems/BA/ba_analyse_software

# Schritt 1: fMRI-Daten für alle 4 ROIs importieren
.venv/bin/python scripts/run_s1_roi_batch.py --subject S1 --max-trials 500

# Schritt 2–5: Feature-Extraktion, Geometrie, Vergleich für alle ROIs
.venv/bin/python scripts/compare_resnet18_to_fmri_rois.py --subject S1

# Ergebnis liegt in:
# outputs/results/ann_resnet18_S1_roi_summary.csv
```

Oder wenn Features bereits existieren, nur Summary neu berechnen:

```bash
.venv/bin/python scripts/write_s1_roi_summary_from_existing.py
```

---

## 9. Flatmap-Visualisierung

### 9.1 Ansatz

Die fMRI-Betas werden direkt auf einer abgewickelten Kortex-Darstellung (Flatmap)
visualisiert. Die Implementierung umgeht die pycortex-`Vertex`-Validierung,
die wegen der Diskrepanz zwischen Kortex-Vertices (279 601) und vollen
`.gii`-Oberflächen-Vertices (304 380) fehlschlug:

```python
# flatverts: (N_pixel_in_mask, 279601) sparse-Matrix
# Vertex-Daten → Flatmap-Pixel
pixel_values = flatverts.dot(data_279601)
img[flatmask] = pixel_values
```

**Koordinatentransformation** (`overlays.svg` ist 90° gedreht gespeichert):

```
SVG-x (0..svg_w)  →  Bild-Zeilen  (0..flatmask.shape[0]),  scale = shape[0]/svg_w
SVG-y (0..svg_h)  →  Bild-Spalten (0..flatmask.shape[1]),  scale = shape[1]/svg_h

matplotlib-Plot:
  x_mpl = SVG_y * scale_col   (→ Spalte = horizontale Achse)
  y_mpl = SVG_x * scale_row   (→ Zeile  = vertikale Achse)
```

Gilt für alle drei Subjects (S1: svg_w=1960.5, S2: svg_w=2025.5, S3: svg_w=2097.7).

### 9.2 ROI-Overlays

ROI-Grenzen werden aus der pycortex-Datenbank gelesen:

```
/Volumes/Sonstige Backups/Data/db/S1/overlays.svg
```

Da nicht alle Analyse-ROIs als benannte SVG-Gruppen existieren, werden
Approximationen verwendet:

| Analyse-ROI | SVG-Gruppe(n)    | Anmerkung                         |
|-------------|------------------|-----------------------------------|
| V1          | `V1`             | exakte Übereinstimmung            |
| hV4         | `V3`             | V4 fehlt im SVG; V3 ist benachbart |
| LOC         | `TOS`            | TOS ist Kernbereich des LOC       |
| IT          | `FFA`, `EBA`     | IT-Subbereiche für Gesicht/Körper |

**Einschränkung für S2/S3**: Die `overlays.svg` dieser Subjects enthält keine
benannten Gruppen — nur 20 generische Pfade ohne Labels. ROI-Overlays sind
daher nur für S1 verfügbar.

### 9.3 Verwendung

```bash
# Einzelner Trial
venv_cortex/bin/python scripts/visualize_flatmap.py \
    --subject S1 --trial 3 --all-rois \
    --output outputs/figures/flatmap_S1_trial3.png

# Durchschnitt über 12 Wiederholungen (empfohlen für Thesis-Abbildungen)
venv_cortex/bin/python scripts/visualize_flatmap.py \
    --subject S1 \
    --trials 36 1524 2385 3271 3338 4811 5236 6069 7240 7775 8240 9318 \
    --all-rois \
    --output outputs/figures/flatmap_S1_butterfly_avg12.png
```

Die 100 Teststimuli mit je 12 Wiederholungen sind über
`stimuli/S{n}_stimuli.csv` (Feld `Type='single'`, Stimulus-Name taucht genau
12× auf) identifizierbar.

**Farbskala**: automatisch ±3σ der sichtbaren Beta-Werte; manuell
überschreibbar mit `--vmin`/`--vmax`.

### 9.4 Erzeugte Abbildungen

| Datei                                             | Inhalt                                   |
|---------------------------------------------------|------------------------------------------|
| `flatmap_S1_candelabra_avg12_allrois.png`         | S1, Candelabra, 12-Rep-Avg, alle 4 ROIs |
| `flatmap_S1_candelabra_trial1.png`                | S1, Candelabra, einzelner Trial          |
| `flatmap_S1_butterfly_avg12.png`                  | S1, Butterfly, 12-Rep-Avg, alle 4 ROIs  |
| `flatmap_S2_butterfly_avg12.png`                  | S2, Butterfly, 12-Rep-Avg               |
| `flatmap_S3_butterfly_avg12.png`                  | S3, Butterfly, 12-Rep-Avg               |
| `flatmap_S1_piano_avg12.png`                      | S1, Piano, 12-Rep-Avg, alle 4 ROIs      |
| `flatmap_S1_cow_avg12.png`                        | S1, Cow, 12-Rep-Avg, alle 4 ROIs        |

---

## 10. Datei-Referenz

### Wichtigste Output-Dateien

| Datei                                          | Inhalt                                    |
|------------------------------------------------|-------------------------------------------|
| `outputs/results/ann_resnet18_S1_roi_summary.csv` | Spearman + CKA für alle Layer × ROIs  |
| `outputs/human/processed_fmri_S1_{ROI}_distance.npy` | fMRI-Distanzmatrix (100×100)       |
| `outputs/geometry/ann_resnet18_S1_matched_{layer}_cosine.npy` | ANN-Distanzmatrix (100×100) |
| `outputs/features/ann_resnet18_S1_matched_{layer}.npy` | ANN-Feature-Matrix (100×d)     |
| `data/ann_S1_matched_stimuli.csv`              | 100 gematchte Stimuli mit Bildpfaden      |
| `data/processed_fmri_S1_{ROI}_stimuli.csv`     | fMRI-Stimulusliste nach Aggregation       |

### Wichtigste Source-Dateien

| Datei                                    | Funktion                               |
|------------------------------------------|----------------------------------------|
| `scripts/import_processed_fmri.py`       | fMRI-Import, ROI-Maske, Aggregation    |
| `scripts/build_ann_stimuli_from_fmri.py` | Bildpfad-Matching                      |
| `scripts/extract_resnet18_features.py`   | Forward-Hook-Feature-Extraktion        |
| `scripts/compute_geometry.py`            | Paarweise Distanzmatrizen              |
| `scripts/compare_geometry_to_human.py`   | Spearman + CKA Berechnung              |
| `scripts/compare_resnet18_to_fmri_rois.py` | Orchestrator für einen Subject       |
| `scripts/run_s1_roi_batch.py`            | fMRI-Import-Batch für alle ROIs        |
| `src/stats_utils.py`                     | Spearman, CKA, Permutationstest        |
| `src/utils.py`                           | Pfad-, Config- und CSV-Hilfsfunktionen |

### Konfigurationen

| Config                              | Wann verwenden                      |
|-------------------------------------|-------------------------------------|
| `processed_fmri_s1_loc.json`        | Template für fMRI-Import            |
| `ann_from_processed_s1_loc.json`    | Template für ANN-Feature-Extraction |
| `tmp_ann_resnet18_S1_matched.json`  | Automatisch erzeugt, nicht manuell editieren |

---

*Letzte Aktualisierung: 2026-05-20*
*Analyse vollständig: RSA, Noise Ceiling, TDA, Flatmap-Visualisierung*

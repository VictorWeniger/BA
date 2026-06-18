# Workflow der aktuellen Analyse

Diese Datei beschreibt den aktuellen Code-Stand der BA-Analyse. Maßgeblich ist die Analyse mit **ResNet-18**, **THINGS-fMRI-Daten**, den ROIs **V1, hV4, LOC, IT** und den Verfahren **RSA**, **CKA**, **Noise Ceiling** und **TDA**.

---

## 1. Gesamtidee

Die Pipeline vergleicht die Repräsentationsstruktur eines ImageNet-vortrainierten ResNet-18 mit fMRI-Repräsentationen des menschlichen visuellen Kortex.

```text
THINGS-fMRI-Betas
  -> ROI-Features pro Subject
  -> per-session Z-Scoring
  -> Mittelung über Stimulus-Wiederholungen
  -> fMRI-RDMs pro Subject × ROI

THINGS-Bilder
  -> ResNet-18 Forward Pass
  -> Layer-Features
  -> ANN-RDMs pro Subject × Layer

ANN-RDMs + fMRI-RDMs
  -> RSA
  -> CKA
  -> Noise Ceiling
  -> Persistente Homologie / TDA
  -> CSV-Ergebnisse + Abbildungen
```

Die zentrale Frage ist nicht, ob ResNet-18 die Bilder korrekt klassifiziert, sondern ob die **Ähnlichkeitsstruktur** der Bilder in ResNet-Schichten der Ähnlichkeitsstruktur in fMRI-ROIs ähnelt.

---

## 2. Zentrale Skripte

| Script | Rolle |
|---|---|
| `import_processed_fmri.py` | Lädt fMRI-Betas, maskiert ROIs, normalisiert pro Session, mittelt Wiederholungen und baut fMRI-RDMs |
| `build_ann_stimuli_from_fmri.py` | Baut aus der fMRI-Stimulusliste eine passende Bildliste für ResNet |
| `extract_resnet18_features.py` | Extrahiert ResNet-18-Aktivierungen mit Forward Hooks |
| `compute_geometry.py` | Berechnet ANN-RDMs aus Layer-Features |
| `compare_resnet18_to_fmri_rois.py` | Orchestriert ResNet-18-vs-fMRI-Vergleiche pro Subject und ROI |
| `compare_geometry_to_human.py` | Berechnet RSA, Permutationstest und CKA zwischen ANN-RDM und fMRI-RDM |
| `compute_noise_ceiling.py` | Berechnet within-subject Noise Ceiling über Sessions |
| `compare_tda_resnet18_to_fmri_rois.py` | Berechnet persistente Homologie auf RDMs und Wasserstein-Distanzen |
| `visualize_flatmap.py` | Erzeugt kortikale Flatmap-Visualisierungen aus Surface-Betas |

Hilfsfunktionen liegen in `src/`, vor allem:

| Datei | Inhalt |
|---|---|
| `stats_utils.py` | Spearman, Permutationstest, CKA, Upper-Triangle-Extraktion |
| `utils.py` | Configs laden, Pfade auflösen, CSV schreiben |
| `pca_utils.py`, `png_utils.py`, `html_plots.py` | ältere/ergänzende Visualisierungs- und Analysehilfen |

---

## 3. Datenbasis

### fMRI-Daten

Lokaler Pfad:

```text
/Volumes/Sonstige Backups/Data/processed
```

Pro Subject liegen u. a. vor:

```text
S1/stimuli/S1_stimuli.csv
S1/betas/vol/S1_betas_vol.npy
S1/rois/localizers/S1_localizers.csv
```

Die Beta-Matrix hat eine Zeile pro Trial. Die Stimulus-CSV enthält die zugehörigen Metadaten, z. B. `Stimulus`, `Concept`, `Session`, `Split` und `Dataset`.

Verwendet werden:

```text
Subjects: S1, S2, S3
ROIs:     V1, hV4, LOC, IT
Split:    test
Dataset:  THINGS
```

### Bilddaten

Lokaler Pfad:

```text
/Volumes/Sonstige Backups/Data/object_images
```

Beispiel:

```text
Concept:  dog
Stimulus: dog_12s.jpg
Bild:     object_images/dog/dog_12s.jpg
```

---

## 4. Schritt 1: fMRI-Import

**Script:** `import_processed_fmri.py`

Beispielconfig:

```text
configs/processed_fmri_s1_loc.json
```

Wichtige Config-Punkte:

```json
"trial_filter": {
  "split": "test",
  "dataset": "THINGS"
},
"normalization": {
  "per_session_zscore": true
},
"aggregation": {
  "group_by": "Stimulus",
  "method": "mean"
}
```

### Was passiert?

1. Stimulus-CSV des Subjects laden.
2. Nur THINGS-Testtrials auswählen.
3. Volumen-Betas laden, z. B. `S1_betas_vol.npy`.
4. ROI-Maske aus `rois/localizers/S1_localizers.csv` laden.
5. Beta-Matrix auf die ROI-Voxel einschränken.
6. **Per-session Z-Scoring** anwenden.
7. Wiederholungen desselben Stimulus mitteln.
8. fMRI-RDM als Cosine-Distanzmatrix speichern.

Wichtig: Die Voxel werden nicht zu einem Skalar gemittelt. Pro Stimulus bleibt ein **ROI-Voxel-Vektor** erhalten.

### Per-session Z-Scoring

Die aktuelle Pipeline normalisiert die Betas innerhalb jeder Session separat:

```python
features_out[idx, :] = (subset - means) / stds
```

Das reduziert Session-spezifische Scannerdrifts und Unterschiede in der Voxel-Skalierung. Erst danach werden die Wiederholungen pro Stimulus gemittelt.

Wenn `per_session_zscore = true`, berechnet der Code die fMRI-Cosine-RDM danach nur noch mit L2-Normalisierung, weil die Feature-Skalierung bereits vorher entfernt wurde.

### Output

Pro Subject und ROI:

```text
data/processed_fmri_{Subject}_{ROI}_stimuli.csv
outputs/human/processed_fmri_{Subject}_{ROI}_features.npy
outputs/human/processed_fmri_{Subject}_{ROI}_distance.npy
```

Typische Shapes:

```text
features: (100, n_roi_voxels)
RDM:      (100, 100)
```

---

## 5. Schritt 2: Bildliste für ResNet bauen

**Script:** `build_ann_stimuli_from_fmri.py`

Dieses Skript nimmt die aggregierte fMRI-Stimulusliste und sucht die passenden THINGS-Bilder.

Beispiel:

```bash
python scripts/build_ann_stimuli_from_fmri.py \
  --fmri-stimuli data/processed_fmri_S1_LOC_stimuli.csv \
  --output data/ann_S1_matched_stimuli.csv
```

Output:

```text
data/ann_{Subject}_matched_stimuli.csv
```

Die Bildliste enthält u. a.:

```text
image_id, concept, category, image_path, fmri_stimulus, fmri_subject, n_aggregated_trials
```

Warum wird LOC als Basis verwendet?  
Alle ROIs verwenden denselben Testtrial-Filter und dieselbe Stimulusaggregation. Daher reicht eine ROI-Stimulusliste als Grundlage für die ANN-Bildreihenfolge. Im Orchestrator wird LOC dafür verwendet.

---

## 6. Schritt 3: ResNet-18-Features extrahieren

**Script:** `extract_resnet18_features.py`

Auch wenn einzelne Kommentare im Skript noch historisch „ResNet-50“ erwähnen, lädt der Code das Modell aus der Config. Für die BA ist das:

```json
"name": "resnet18",
"pretrained": true,
"layers": ["conv1", "layer1", "layer2", "layer3", "layer4", "fc"]
```

### Preprocessing

Die Bilder werden mit ImageNet-Standardpreprocessing vorbereitet:

```python
Resize(256)
CenterCrop(224)
ToTensor()
Normalize(mean=[0.485, 0.456, 0.406],
          std=[0.229, 0.224, 0.225])
```

Das Modell erhält also effektiv:

```text
3 × 224 × 224
```

### Forward Hooks

Für jede Zielschicht wird ein Forward Hook registriert. Der Hook speichert die Ausgabe der Schicht, ohne den normalen Forward Pass zu verändern.

Konvolutionale Aktivierungen haben die Form:

```text
batch × channels × height × width
```

Diese werden mit Global Average Pooling reduziert:

```python
activation.mean(dim=(2, 3))
```

Also:

```text
batch × channels × height × width
-> batch × channels
```

### Output-Shapes

Für 100 Stimuli:

| Layer | Shape |
|---|---|
| `conv1` | `(100, 64)` |
| `layer1` | `(100, 64)` |
| `layer2` | `(100, 128)` |
| `layer3` | `(100, 256)` |
| `layer4` | `(100, 512)` |
| `fc` | `(100, 1000)` |

Output:

```text
outputs/features/ann_resnet18_{Subject}_matched_{layer}.npy
outputs/features/ann_resnet18_{Subject}_matched_image_order.csv
```

---

## 7. Schritt 4: ANN-RDMs berechnen

**Script:** `compute_geometry.py`

Für jede Layer-Feature-Matrix wird eine paarweise Cosine-Distanzmatrix berechnet:

```text
D_ij = 1 - cos(x_i, x_j)
```

Der Code macht:

1. Z-Score pro Feature-Spalte.
2. L2-Normalisierung pro Stimulusvektor.
3. Matrixmultiplikation für alle paarweisen Cosine Similarities.
4. `1 - similarity`.

Output:

```text
outputs/geometry/ann_resnet18_{Subject}_matched_{layer}_cosine.npy
```

Shape:

```text
(100, 100)
```

---

## 8. Schritt 5: RSA und CKA

**Scripts:**

```text
compare_resnet18_to_fmri_rois.py
compare_geometry_to_human.py
stats_utils.py
```

`compare_resnet18_to_fmri_rois.py` ist der Orchestrator. Für ein Subject:

1. ANN-Stimulusliste bauen.
2. ResNet-18-Features extrahieren.
3. ANN-RDMs berechnen.
4. Für jede ROI eine Config erzeugen.
5. `compare_geometry_to_human.py` ausführen.
6. Ergebnisse in einer Subject-ROI-CSV sammeln.

### RSA

Für jede Kombination aus Subject, ROI und Layer:

1. ANN-RDM laden.
2. fMRI-RDM laden.
3. Oberes Dreieck ohne Diagonale extrahieren.
4. Spearman-Korrelation berechnen.

Bei 100 Stimuli:

```text
100 × 99 / 2 = 4950 Paarvergleiche
```

Interpretation:

> RSA fragt, ob dieselben Stimuluspaare im ANN und in der fMRI-ROI ähnlich bzw. unähnlich sind.

### Permutationstest

Der Code nutzt einen zweiseitigen Test:

```python
abs(spearman_corr(x, rng.permutation(y))) >= observed
```

p-Wert:

```text
(count + 1) / (n_permutations + 1)
```

Bei 1000 Permutationen:

```text
Minimum p = 1 / 1001 ≈ 0.001
```

### CKA

Für CKA werden Distanzmatrizen in Ähnlichkeitsmatrizen umgewandelt:

```python
K = 1 - D
```

Dann:

```python
kernel_cka(K_dnn, K_fmri)
```

Interpretation:

> CKA fragt nach der globalen Ähnlichkeitsstruktur der Repräsentationsräume, nicht nur nach der Rangordnung der paarweisen Distanzen.

Hinweis: In `compare_geometry_to_human.py` und `stats_utils.py` stehen noch teilweise historische Kommentare mit „Human“ oder „THINGS“. In der aktuellen BA-Analyse ist diese Matrix eine **fMRI-ROI-RDM**.

### Output

Pro Subject:

```text
outputs/results/ann_resnet18_{Subject}_roi_summary.csv
```

Zusammengeführt:

```text
outputs/results/ann_resnet18_all_subjects_roi_summary.csv
```

Spalten:

```text
subject, roi, layer, metric, spearman_r, p_value, cka
```

---

## 9. Schritt 6: Noise Ceiling

**Script:** `compute_noise_ceiling.py`

Der aktuelle Noise Ceiling ist ein **within-subject Noise Ceiling über Sessions**.

Das heißt: Er fragt nicht primär, wie ähnlich S1, S2 und S3 zueinander sind, sondern:

> Wie stabil ist die fMRI-RDM innerhalb eines Subjects über Sessions hinweg?

### Ablauf pro Subject und ROI

1. Alle THINGS-Testtrials laden.
2. ROI-Betas extrahieren.
3. Pro Session z-standardisieren.
4. Pro Session eine RDM über die 100 Stimuli berechnen.
5. Leave-one-session-out Noise Ceiling berechnen.

### Formeln

Für jede Session `s`:

```text
NC_upper_s = Spearman(RDM_s, mean(RDM_all_sessions))
NC_lower_s = Spearman(RDM_s, mean(RDM_all_except_s))
```

Dann:

```text
NC_upper = Mittelwert über Sessions
NC_lower = Mittelwert über Sessions
```

Am Ende werden die Subject-Werte zusätzlich über S1, S2 und S3 gemittelt.

### Output

```text
outputs/results/noise_ceiling.csv
outputs/figures/rsa_resnet18_fmri_with_noise_ceiling.png
```

Warum machen wir das?

> Damit wir RSA-Werte gegen die Messstabilität der fMRI-Daten einordnen können. Ein niedriger RSA-Wert kann sonst entweder ein schlechtes Modell oder einfach verrauschte fMRI-Daten bedeuten.

---

## 10. Schritt 7: Topologische Analyse

**Script:** `compare_tda_resnet18_to_fmri_rois.py`

Die TDA wird aktuell direkt auf den bereits berechneten RDMs durchgeführt, nicht auf PCA-reduzierten Feature-Vektoren.

### Ablauf

Für jede Kombination aus Subject, ROI und Layer:

1. ANN-RDM laden.
2. fMRI-RDM laden.
3. Beide RDMs auf `[0, 1]` normieren.
4. Mit `ripser(..., distance_matrix=True, maxdim=1)` Persistenzdiagramme berechnen.
5. H0- und H1-Diagramme zwischen ANN und fMRI mit Wasserstein-Distanz vergleichen.

### H0 und H1

```text
H0 = Zusammenhangskomponenten / Clusterstruktur
H1 = Zyklen / Schleifenstruktur
```

Kleinere Wasserstein-Distanz bedeutet:

```text
topologisch ähnlicher
```

### Output

```text
outputs/results/tda_resnet18_all_subjects_roi_summary.csv
outputs/figures/tda_resnet18_fmri_wasserstein_H0.png
outputs/figures/tda_resnet18_fmri_wasserstein_H1.png
outputs/figures/tda_resnet18_fmri_heatmap_H0.png
outputs/figures/tda_resnet18_fmri_heatmap_H1.png
```

---

## 11. Schritt 8: Flatmap-Visualisierung

**Script:** `visualize_flatmap.py`

Dieses Skript dient der Visualisierung, nicht der quantitativen RSA/CKA/TDA-Hauptanalyse.

Es nutzt:

```text
/Volumes/Sonstige Backups/Data/processed/{Subject}/betas/surf
/Volumes/Sonstige Backups/Data/db/{Subject}/cache
```

Es verwendet vorhandene pycortex-Cache-Dateien direkt:

```text
flatverts_1024.npz
flatmask_1024.npz
curvature.npz
overlays.svg
```

Output-Beispiele:

```text
outputs/figures/flatmap_S1_butterfly_avg12.png
outputs/figures/flatmap_S2_butterfly_avg12.png
outputs/figures/flatmap_S3_butterfly_avg12.png
```

Wichtig:

> Die Hauptanalyse verwendet Volumen-Betas und ROI-RDMs. Die Flatmaps sind zur visuellen Plausibilisierung und Illustration gedacht.

---

## 12. Vollständiger aktueller Datenfluss

```text
processed fMRI data
  │
  ├─> import_processed_fmri.py
  │     ├─ per-session z-scoring
  │     ├─ average repetitions by Stimulus
  │     └─ outputs/human/processed_fmri_{S}_{ROI}_distance.npy
  │
  ├─> build_ann_stimuli_from_fmri.py
  │     └─ data/ann_{S}_matched_stimuli.csv
  │
  ├─> extract_resnet18_features.py
  │     └─ outputs/features/ann_resnet18_{S}_matched_{layer}.npy
  │
  ├─> compute_geometry.py
  │     └─ outputs/geometry/ann_resnet18_{S}_matched_{layer}_cosine.npy
  │
  ├─> compare_geometry_to_human.py
  │     └─ RSA + permutation p-value + CKA
  │
  ├─> compute_noise_ceiling.py
  │     └─ within-subject session noise ceiling
  │
  └─> compare_tda_resnet18_to_fmri_rois.py
        └─ H0/H1 Wasserstein distances
```

---

## 13. Funktionen im Detail: Wann wird was wofür benutzt?

Dieser Abschnitt ist die feinere Lesebrille für den Code. Er zeigt pro Skript, welche Funktionen im Workflow tatsächlich aufgerufen werden, welche Daten sie bekommen und was sie zurückgeben.

### 13.1 `import_processed_fmri.py`

Ziel: Aus den rohen THINGS-fMRI-Trials eine saubere ROI-Feature-Matrix und eine fMRI-RDM bauen.

| Funktion | Wird benutzt wann? | Zweck | Input | Output |
|---|---|---|---|---|
| `load_json(path)` | Direkt am Anfang von `main()` | Config laden | Config-Pfad | Python-Dict |
| `read_csv_rows(path)` | Beim Laden der Stimulus-Metadaten | Trial-Tabelle einlesen | `S*_stimuli.csv` | Liste von Zeilen-Dicts |
| `find_column_index(csv_path, column_name)` | Innerhalb von `load_roi_mask()` | ROI-Spalte in CSV finden | ROI-CSV, ROI-Name | Spaltenindex |
| `load_roi_mask(subject_dir, roi_type, roi_name)` | Nach dem Laden der Beta-Matrix | ROI als boolsche Voxelmaske laden | Subject-Ordner, z. B. `localizers`, `LOC` | Bool-Array `(n_voxels,)` |
| `filter_rows(rows, trial_filter)` | Vor dem Beta-Slicing | Nur relevante Trials auswählen | Stimuluszeilen + Filter | Zeilenindizes |
| `per_session_zscore(features, rows)` | Vor der Stimulus-Mittelung, wenn Config es aktiviert | Session-Effekte pro Voxel entfernen | Trial-Features + Metadaten | z-standardisierte Trial-Features |
| `aggregate_features(features, rows, group_by, method)` | Nach der Normalisierung | Wiederholungen desselben Stimulus mitteln | Trial-Features | Stimulus-Features |
| `pairwise_cosine_distance_l2only(features)` | Nach Aggregation, wenn vorher per Session z-standardisiert wurde | fMRI-RDM berechnen | Stimulus-Feature-Matrix | `(100,100)`-RDM |
| `pairwise_cosine_distance(features)` | Alternative, wenn kein per-session Z-Scoring aktiv ist | Z-Score + Cosine-RDM | Feature-Matrix | `(100,100)`-RDM |
| `main()` | CLI-Einstieg | Orchestriert den fMRI-Import | CLI-Argumente/Config | Dateien in `data/` und `outputs/human/` |

Wichtigster Ablauf in `main()`:

```text
Config laden
  -> Stimulus-CSV laden
  -> THINGS-Testtrials filtern
  -> Volumen-Betas laden
  -> ROI-Maske anwenden
  -> per-session z-score
  -> Wiederholungen pro Stimulus mitteln
  -> fMRI-Features speichern
  -> fMRI-RDM speichern
```

Warum ist die Reihenfolge wichtig?

```text
per-session z-score vor Aggregation
```

Wenn zuerst gemittelt würde, wären Session-Effekte bereits in den Stimulusmittelwerten vermischt. Deshalb wird zuerst jede Session separat standardisiert und erst danach über die 12 Wiederholungen gemittelt.

### 13.2 `build_ann_stimuli_from_fmri.py`

Ziel: Sicherstellen, dass ResNet exakt dieselben Bilder in derselben Reihenfolge bekommt wie die fMRI-RDM.

| Funktion | Wird benutzt wann? | Zweck | Input | Output |
|---|---|---|---|---|
| `read_rows(path)` | Anfang von `main()` | fMRI-Stimulusliste laden | `processed_fmri_*_stimuli.csv` | Zeilen-Dicts |
| `main()` | CLI-Einstieg | Bildpfade aus `Concept` + `Stimulus` bauen | fMRI-Stimuli + `object_images` | `ann_*_matched_stimuli.csv` |

Pfadlogik:

```text
Concept = dog
Stimulus = dog_12s.jpg
-> /Volumes/Sonstige Backups/Data/object_images/dog/dog_12s.jpg
```

Wenn ein Bild fehlt, bricht das Skript bewusst ab. Sonst könnten ANN- und fMRI-Reihenfolge unbemerkt auseinanderlaufen.

### 13.3 `extract_resnet18_features.py`

Ziel: Für jedes Bild ResNet-18-Aktivierungen aus mehreren Schichten extrahieren.

| Funktion/Klasse | Wird benutzt wann? | Zweck | Input | Output |
|---|---|---|---|---|
| `ImageTableDataset` | Vor dem DataLoader | Bilder aus CSV laden und transformieren | Stimuluszeilen, Bildpfadspalte | PyTorch-Dataset |
| `ImageTableDataset.__getitem__()` | Bei jedem Batch | Einzelnes Bild laden, RGB-konvertieren, ImageNet-Preprocessing anwenden | Index | `(image_tensor, index)` |
| `get_device()` | Vor Modellladung | Bestes Backend wählen | keine | `mps`, `cuda` oder `cpu` |
| `load_model(model_name, pretrained, ...)` | Vor Feature-Extraktion | ResNet-18 oder ResNet-50 laden | Config-Werte | Torch-Modell |
| `pool_activation(activation)` | Nach jedem Forward Pass pro Layer | 4D-Conv-Aktivierung zu einem Vektor mitteln | Tensor `(B,C,H,W)` | Array `(B,C)` |
| `make_hook(name)` | Beim Registrieren der Hooks | Speichert Layer-Ausgaben im Dict `captured` | Layername | Hook-Funktion |
| `main()` | CLI-Einstieg | Orchestriert Modell, Hooks, Batches und Speichern | Config | `.npy`-Features |

Zentraler Mechanismus:

```text
Forward Hook
  -> hängt an einer ResNet-Schicht
  -> fängt deren Output ab
  -> speichert ihn in captured[layer_name]
```

Warum Global Average Pooling?

```text
Conv-Output: batch × channels × height × width
GAP:         batch × channels
```

Dadurch gibt es pro Bild und Layer einen einzelnen Vektor. Ohne GAP wären die Featurevektoren extrem groß und stark von räumlicher Position abhängig.

### 13.4 `compute_geometry.py`

Ziel: Aus ANN-Feature-Matrizen RDMs berechnen.

| Funktion | Wird benutzt wann? | Zweck | Input | Output |
|---|---|---|---|---|
| `pairwise_cosine_distance(features)` | Für aktuelle ANN-RDMs | Feature-Spalten z-standardisieren, Stimulusvektoren L2-normalisieren, Cosine-Distanz berechnen | `(100,d)` | `(100,100)` |
| `pairwise_euclidean_distance(features)` | Optional, falls Config `euclidean` nutzt | Euklidische Distanzen berechnen | `(100,d)` | `(100,100)` |
| `compute_distance(features, metric)` | Pro Layer in `main()` | Metrik dispatchen | Features + Metrikname | RDM |
| `main()` | CLI-Einstieg | Lädt alle Layer-Features und speichert RDMs | Config | `outputs/geometry/*.npy` |

Aktuelle Standardmetrik:

```text
metric = cosine
```

Die ANN-RDMs werden für RSA, CKA und TDA wiederverwendet.

### 13.5 `compare_resnet18_to_fmri_rois.py`

Ziel: Einen kompletten Subject-Lauf automatisieren.

| Funktion | Wird benutzt wann? | Zweck |
|---|---|---|
| `run_step(command)` | Für jeden Subprozess | Befehl anzeigen und ausführen |
| `patch_config_for_roi(template_path, output_path, roi, subject)` | Vor jedem ROI-Vergleich | Config so anpassen, dass sie auf die passende fMRI-RDM zeigt |
| `main()` | CLI-Einstieg | Bildliste, Feature-Extraktion, RDMs und ROI-Vergleiche orchestrieren |

Was der Orchestrator wirklich macht:

```text
Für ein Subject:
  1. ANN-Bildliste aus fMRI-Stimuli bauen
  2. tmp-Config für dieses Subject schreiben
  3. ResNet-18-Features extrahieren
  4. ANN-RDMs berechnen
  5. Für V1, hV4, LOC, IT:
       - Config auf ROI-RDM patchen
       - RSA/CKA berechnen
       - Ergebniszeilen einsammeln
  6. Subject-ROI-Summary speichern
```

### 13.6 `compare_geometry_to_human.py`

Ziel: Eine ANN-RDM gegen eine fMRI-RDM vergleichen.

| Funktion/Schritt | Wird benutzt wann? | Zweck |
|---|---|---|
| `upper_triangle_values(human_matrix)` | Direkt nach Laden der fMRI-RDM | RDM auf 4950 Paarwerte reduzieren |
| `K_human = 1.0 - human_matrix` | Vor CKA | Distanzmatrix zu Ähnlichkeitskernel machen |
| `upper_triangle_values(ann_matrix)` | Pro Layer | ANN-RDM auf 4950 Paarwerte reduzieren |
| `spearman_corr(ann_values, human_values)` | Pro Layer | RSA berechnen |
| `permutation_test_spearman(...)` | Pro Layer | p-Wert berechnen |
| `kernel_cka(K_dnn, K_human)` | Pro Layer | CKA berechnen |

Historischer Name: Das Skript heißt noch `compare_geometry_to_human.py`. In der aktuellen BA ist `human_matrix` aber nicht THINGS-Similarity, sondern die fMRI-RDM einer ROI.

### 13.7 `stats_utils.py`

Ziel: Kleine Statistikfunktionen ohne schwere Zusatzabhängigkeiten.

| Funktion | Wofür? |
|---|---|
| `upper_triangle_values(matrix)` | Nimmt jedes Stimuluspaar genau einmal aus einer symmetrischen RDM |
| `rankdata_average_ties(values)` | Berechnet Ränge für Spearman, Gleichstände erhalten Durchschnittsränge |
| `pearson_corr(x, y)` | Basis-Korrelation, wird auf Rangwerte angewendet |
| `spearman_corr(x, y)` | RSA-Korrelationsmaß |
| `permutation_test_spearman(x, y, ...)` | Zweiseitiger Permutationstest mit `+1`-Korrektur |
| `kernel_cka(K, L)` | CKA auf vorberechneten Kernelmatrizen |
| `linear_cka(X, Y)` | Alternative CKA direkt auf Feature-Matrizen, vor allem für Layer-zu-Layer-Vergleiche |

### 13.8 `compute_noise_ceiling.py`

Ziel: Abschätzen, wie stabil die fMRI-RDMs überhaupt sind.

| Funktion | Wird benutzt wann? | Zweck |
|---|---|---|
| `load_raw_data(subject, roi_type, roi_name)` | Pro Subject × ROI | Alle THINGS-Testtrials und ROI-Betas laden |
| `per_session_zscore(betas, sessions)` | Direkt nach dem Laden | Session-spezifische Skalierung entfernen |
| `compute_session_rdms(betas, sessions, stimuli)` | Nach Z-Scoring | Eine RDM pro Session bauen |
| `compute_within_subject_nc(rdms)` | Nach Session-RDMs | Leave-one-session-out `NC_upper`/`NC_lower` berechnen |
| `_plot_with_noise_ceiling(nc_results)` | Nach CSV-Speicherung | RSA-Profil mit Noise-Ceiling-Band plotten |

Der aktuelle Noise Ceiling ist:

```text
within-subject, across sessions
```

Er fragt also:

> Wie reproduzierbar ist die Repräsentationsstruktur innerhalb eines Subjects über die Mess-Sessions hinweg?

### 13.9 `compare_tda_resnet18_to_fmri_rois.py`

Ziel: RDM-Topologie von ANN und fMRI vergleichen.

| Funktion | Wird benutzt wann? | Zweck |
|---|---|---|
| `normalize_rdm(rdm)` | Vor Ripser | RDM auf `[0,1]` skalieren |
| `compute_diagram_from_rdm(rdm, maxdim=1)` | Für jede ANN- und fMRI-RDM | Vietoris-Rips-Persistenz berechnen |
| `wasserstein_safe(dgm_a, dgm_b)` | Pro Layer × ROI × Homologiedimension | Wasserstein-Distanz berechnen |
| `_plot_results(...)` | Nach CSV-Speicherung | Linienplots und Heatmaps erzeugen |

Warum TDA auf RDMs?

```text
RSA und TDA verwenden dadurch dieselbe Distanzgrundlage.
```

Das macht die Ergebnisse methodisch vergleichbarer.

### 13.10 `visualize_flatmap.py`

Ziel: fMRI-Aktivität auf kortikale Flatmaps projizieren.

| Funktion | Zweck |
|---|---|
| `load_flatmap_cache(subject)` | pycortex-Cache mit Pixel-Vertex-Zuordnung laden |
| `load_curvature(subject)` | Kortex-Krümmung als Hintergrund laden |
| `load_betas(subject, trial_indices)` | Surface-Betas für Trials laden und mitteln |
| `vertex_to_flatmap(data, flatverts, flatmask)` | Vertexwerte in 2D-Flatmap-Bild umrechnen |
| `load_roi_svg_paths(subject, roi)` | ROI-Konturen aus `overlays.svg` lesen |
| `draw_roi_contour(...)` | ROI-Grenzen auf die Flatmap zeichnen |

Diese Visualisierung ist nicht Teil der quantitativen Hauptanalyse. Die Hauptanalyse nutzt Volumen-Betas; die Flatmap nutzt Surface-Betas für Abbildungen und Plausibilisierung.

---

## 14. Wichtige Unterschiede zur alten Pipeline

| Alt | Aktuell |
|---|---|
| ResNet-50 als Kernmodell | ResNet-18 als Kernmodell |
| THINGS-Similarity/WordNet als Hauptreferenz | fMRI-ROIs aus THINGS-data als Hauptreferenz |
| fMRI war spätere Phase | fMRI ist Kern der Analyse |
| Dimensionalität als eigene Forschungsfrage | nicht mehr Kern der aktuellen BA |
| TDA auf Feature-Punktwolken mit PCA-Modi | TDA direkt auf normierten Cosine-RDMs |
| Noise Ceiling cross-subject beschrieben | aktueller Code: within-subject über Sessions |
| fMRI-Z-Score allgemein | aktueller Code: per-session Z-Scoring vor Aggregation |

---

## 15. Kurzfassung für Erklärung

Wenn man den Code in einem Satz erklären muss:

> Der Code nimmt dieselben 100 THINGS-Bilder für ResNet-18 und fMRI, baut aus beiden Systemen Cosine-RDMs, vergleicht diese geometrisch mit RSA und CKA, ordnet die RSA-Werte über einen within-subject Noise Ceiling ein und vergleicht die globale Topologie der RDMs mit persistenter Homologie in H0 und H1.

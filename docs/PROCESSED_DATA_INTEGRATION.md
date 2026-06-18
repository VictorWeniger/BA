# Integration der processed-fMRI-Daten

## Datenstruktur

Die Daten liegen unter:

```text
/Volumes/Sonstige Backups/Data/processed
```

Gefundene Subjects:

```text
S1
S2
S3
```

Pro Subject gibt es u. a.:

```text
betas/
brainmasks/
neighbors/
prf/
rois/
stimuli/
structural/
```

## Wichtigste Dateien

Für S1:

```text
S1/stimuli/S1_stimuli.csv
S1/betas/vol/S1_betas_vol.npy
S1/betas/surf/S1_betas_surf_lh.npy
S1/betas/surf/S1_betas_surf_rh.npy
S1/rois/localizers/S1_localizers.csv
S1/rois/glasser/S1_glasser.csv
```

Die entscheidende Beziehung:

```text
S1_stimuli.csv: 9840 Trials
S1_betas_vol.npy: (9840, 211339)
```

Das bedeutet:

> Jede Zeile in `S1_betas_vol.npy` entspricht einer Trial-/Stimulus-Zeile in `S1_stimuli.csv`.

## Wie die Daten in unsere Pipeline passen

Unsere bisherige Pipeline arbeitet mit Feature-Matrizen:

```text
n_items x n_features
```

Die fMRI-Betas sind genau so eine Matrix:

```text
Trials x Voxels
```

Damit können fMRI-Betas wie menschliche Repräsentationen behandelt werden:

```text
Stimulus -> fMRI-Beta-Vektor -> Distanzmatrix -> Human-Vergleich
```

## Warum zunächst `vol` statt `surf`

Die ROI-Dateien wie `S1_localizers.csv` haben 211339 Zeilen. Das passt exakt zu:

```text
S1_betas_vol.npy: 211339 Features
```

Deshalb ist der erste Importer auf `space = "vol"` ausgelegt.

Die Surface-Betas haben andere Feature-Anzahlen:

```text
S1_betas_surf_lh.npy: (9840, 138863)
S1_betas_surf_rh.npy: (9840, 140738)
```

Surface-Integration kann später ergänzt werden.

## Aktueller Importer

Skript:

```text
scripts/import_processed_fmri.py
```

Config:

```text
configs/processed_fmri_s1_loc.json
```

Die Beispielconfig macht:

```text
Subject: S1
Space: vol
ROI: LOC aus localizers
Split: test
Dataset: THINGS
Max Trials: 500
Aggregation: Mittelung pro Stimulus
Distanzmetrik: cosine distance
```

## Ausführen

```bash
cd /Users/victorweniger/Desktop/Uni/Goethe-Uni/Sems/BA/ba_analyse_software
PYTHONPATH=/Users/victorweniger/Desktop/Uni/Goethe-Uni/Sems/BA/mnist_tda_start/.venv/lib/python3.11/site-packages \
/opt/homebrew/bin/python3.11 scripts/import_processed_fmri.py --config configs/processed_fmri_s1_loc.json
```

## Outputs

```text
data/processed_fmri_S1_LOC_stimuli.csv
outputs/human/processed_fmri_S1_LOC_features.npy
outputs/human/processed_fmri_S1_LOC_distance.npy
```

Diese Distanzmatrix kann später als menschliche Vergleichsmatrix für ANN-Layer verwendet werden.

## Nächster Schritt

Die ANN-Features müssen auf dieselben Stimuli bezogen werden. Dafür brauchen wir später die tatsächlichen Bilddateien zu den Einträgen in:

```text
Stimulus
Concept
```

Sobald die Bilder verfügbar sind, wird daraus eine `stimuli_subset.csv` für die ANN-Pipeline erzeugt.

## Bilddateien

Die THINGS-Bilder liegen jetzt unter:

```text
/Volumes/Sonstige Backups/Data/object_images
```

Die Struktur passt zu den fMRI-Stimulusnamen:

```text
object_images/dog/dog_12s.jpg
object_images/mango/mango_12s.jpg
```

Aus der aggregierten fMRI-Stimulusliste kann deshalb eine ANN-Stimulusliste erzeugt werden:

```bash
cd /Users/victorweniger/Desktop/Uni/Goethe-Uni/Sems/BA/ba_analyse_software
/opt/homebrew/bin/python3.11 scripts/build_ann_stimuli_from_fmri.py
```

Output:

```text
data/ann_S1_LOC_matched_stimuli.csv
```

Diese Liste hat dieselbe Reihenfolge wie:

```text
outputs/human/processed_fmri_S1_LOC_distance.npy
```

Damit können ANN-Layer-Distanzmatrizen direkt gegen die fMRI-Distanzmatrix verglichen werden.

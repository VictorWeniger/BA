# S1 ResNet18 vs. fMRI Run — VERALTET

> Dieses Dokument beschreibt den ersten Pilotlauf (500 Trials, nur S1,
> CKA noch mit asymmetrischem Kernel). Die aktuellen Ergebnisse für
> S1–S3 mit allen 1200 Test-Trials stehen in
> `docs/ANALYSE_DOKUMENTATION.md` (Abschnitt 5).

## Ziel

Dieser Lauf testet erstmals die echte Pipeline:

```text
THINGS-Bilder
-> ResNet18 Layer-Features
-> ANN-Distanzmatrizen
-> fMRI-ROI-Distanzmatrizen
-> Vergleich ANN-Layer vs. fMRI-ROIs
```

Die zentrale Frage war:

> Welche ResNet18-Schichten stimmen geometrisch am stärksten mit fMRI-Repräsentationen in verschiedenen visuellen ROIs überein?

## Daten

### fMRI-Daten

Pfad:

```text
/Volumes/Sonstige Backups/Data/processed
```

Verwendeter Subject:

```text
S1
```

Verwendete fMRI-Dateien:

```text
/Volumes/Sonstige Backups/Data/processed/S1/stimuli/S1_stimuli.csv
/Volumes/Sonstige Backups/Data/processed/S1/betas/vol/S1_betas_vol.npy
/Volumes/Sonstige Backups/Data/processed/S1/rois/localizers/S1_localizers.csv
```

Die wichtige Struktur:

```text
S1_stimuli.csv: 9840 Trials
S1_betas_vol.npy: (9840, 211339)
```

Das bedeutet:

> Jede Zeile der Beta-Matrix entspricht einer Trial-Zeile in der Stimulus-CSV.

### Bilddaten

Pfad:

```text
/Volumes/Sonstige Backups/Data/object_images
```

Die Bildstruktur passt zu den fMRI-Stimulusnamen:

```text
Stimulus: dog_12s.jpg
Bildpfad: /Volumes/Sonstige Backups/Data/object_images/dog/dog_12s.jpg
```

## Verwendete ROIs

Für S1 wurden vier visuelle ROIs verwendet:

```text
V1
hV4
LOC
IT
```

ROI-Größen:

```text
V1:  1049 Voxels
hV4: 613 Voxels
LOC: 2700 Voxels
IT:  4145 Voxels
```

## fMRI-Import

Für jede ROI wurde aus den fMRI-Betas eine Feature-Matrix erzeugt.

Einstellung:

```text
Subject: S1
Space: vol
Dataset: THINGS
Split: test
Max Trials: 500
Aggregation: Mittelung pro Stimulus
```

Dadurch wurden aus 500 Trials jeweils 100 aggregierte Stimuli.

Erzeugte fMRI-Outputs:

```text
data/processed_fmri_S1_V1_stimuli.csv
data/processed_fmri_S1_hV4_stimuli.csv
data/processed_fmri_S1_LOC_stimuli.csv
data/processed_fmri_S1_IT_stimuli.csv

outputs/human/processed_fmri_S1_V1_features.npy
outputs/human/processed_fmri_S1_hV4_features.npy
outputs/human/processed_fmri_S1_LOC_features.npy
outputs/human/processed_fmri_S1_IT_features.npy

outputs/human/processed_fmri_S1_V1_distance.npy
outputs/human/processed_fmri_S1_hV4_distance.npy
outputs/human/processed_fmri_S1_LOC_distance.npy
outputs/human/processed_fmri_S1_IT_distance.npy
```

Die Distanzmatrizen haben jeweils:

```text
(100, 100)
```

## ANN-Stimulusliste

Aus der aggregierten fMRI-Stimulusliste wurde eine passende ANN-Stimulusliste erzeugt:

```text
data/ann_S1_matched_stimuli.csv
```

Sie enthält 100 Bildpfade und entspricht derselben Stimulus-Reihenfolge wie die fMRI-Distanzmatrizen.

## Modell

Verwendetes Modell:

```text
ResNet18
```

Gewichte:

```text
ImageNet-pretrained
```

Backend:

```text
MPS
```

Verwendete Layer:

```text
conv1
layer1
layer2
layer3
layer4
fc
```

## ResNet18 Feature Extraction

Für die 100 gematchten THINGS-Bilder wurden Layer-Features extrahiert.

Output-Shapes:

```text
conv1:  (100, 64)
layer1: (100, 64)
layer2: (100, 128)
layer3: (100, 256)
layer4: (100, 512)
fc:     (100, 1000)
```

Gespeichert unter:

```text
outputs/features/ann_resnet18_S1_matched_conv1.npy
outputs/features/ann_resnet18_S1_matched_layer1.npy
outputs/features/ann_resnet18_S1_matched_layer2.npy
outputs/features/ann_resnet18_S1_matched_layer3.npy
outputs/features/ann_resnet18_S1_matched_layer4.npy
outputs/features/ann_resnet18_S1_matched_fc.npy
```

## Geometrischer Vergleich

Für jede ResNet18-Schicht wurde eine Cosine-Distanzmatrix berechnet:

```text
outputs/geometry/ann_resnet18_S1_matched_<layer>_cosine.npy
```

Dann wurde jede ANN-Distanzmatrix mit jeder fMRI-ROI-Distanzmatrix verglichen.

Metriken:

```text
Spearman-Korrelation auf Distanzmatrizen
CKA auf Repräsentationen
```

Die zentrale Ergebnisdatei:

```text
outputs/results/ann_resnet18_S1_roi_summary.csv
```

## Ergebnisse

### Spearman-Korrelation

| ROI | Bester Layer | Spearman r |
|---|---:|---:|
| V1 | layer2 | 0.0239 |
| hV4 | conv1 | -0.0007 |
| LOC | layer3 | 0.1536 |
| IT | layer2 | 0.0646 |

Vollständige Werte:

```text
S1,V1,conv1:  -0.0260
S1,V1,layer1:  0.0018
S1,V1,layer2:  0.0239
S1,V1,layer3: -0.1846
S1,V1,layer4: -0.0620
S1,V1,fc:     -0.0921

S1,hV4,conv1:  -0.0007
S1,hV4,layer1: -0.0532
S1,hV4,layer2: -0.0378
S1,hV4,layer3: -0.1559
S1,hV4,layer4: -0.0467
S1,hV4,fc:     -0.0224

S1,LOC,conv1:  -0.0169
S1,LOC,layer1:  0.1005
S1,LOC,layer2:  0.1470
S1,LOC,layer3:  0.1536
S1,LOC,layer4:  0.0989
S1,LOC,fc:      0.0124

S1,IT,conv1:  -0.0160
S1,IT,layer1:  0.0408
S1,IT,layer2:  0.0646
S1,IT,layer3:  0.0624
S1,IT,layer4:  0.0445
S1,IT,fc:     -0.0124
```

### CKA

| ROI | Bester Layer | CKA |
|---|---:|---:|
| V1 | layer4 | 0.4502 |
| hV4 | layer4 | 0.4941 |
| LOC | layer4 | 0.5437 |
| IT | layer4 | 0.5663 |

## Vorläufige Interpretation

Für die Bachelorarbeitsfrage ist der Spearman/RDM-Vergleich besonders relevant, weil er direkt die repräsentationale Geometrie vergleicht.

Der interessanteste Befund:

> Für LOC ist `layer3` am stärksten geometrisch aligned, knapp vor `layer2`.

Das passt grob zur Intermediate-Layer-Idee:

```text
finale Schicht fc: kaum Alignment
mittlere/spätere Schichten layer2/layer3: stärkeres Alignment
```

Für IT ist das Muster schwächer, aber ebenfalls nicht final-layer-dominiert:

```text
layer2/layer3 > layer4 > fc
```

V1 und hV4 zeigen in diesem kleinen Lauf kein klares positives Spearman-Alignment.

CKA zeigt ein anderes Muster:

> In allen ROIs ist `layer4` am höchsten.

Das bedeutet nicht zwingend einen Widerspruch, weil Spearman auf Distanzmatrizen und CKA unterschiedliche Dinge messen:

- Spearman/RDM: Ähnlichkeit der paarweisen Distanzen zwischen Stimuli.
- CKA: Ähnlichkeit der Feature-Strukturen im Repräsentationsraum.

## Einschränkungen

Dieser Lauf ist noch kein finales Ergebnis.

Einschränkungen:

- Nur ein Subject: S1.
- Nur 500 Test-Trials.
- Nach Aggregation nur 100 Stimuli.
- Nur vier ROIs.
- Nur ein Modell: ResNet18.
- Noch keine TDA-Auswertung in diesem Lauf.

## Nächste sinnvolle Schritte

1. Denselben Lauf für S2 und S3 wiederholen.
2. Prüfen, ob LOC/IT-Muster über Subjects stabil bleibt.
3. Größeren Stimulusumfang testen.
4. TDA auf ausgewählten ROIs und Layern ergänzen.
5. Danach ggf. zweites Modell ergänzen, z. B. CLIP oder ResNet50.


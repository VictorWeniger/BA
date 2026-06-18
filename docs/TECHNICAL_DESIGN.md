# Technisches Design

## 1. Analyseziel

Die Software soll pro ANN-Schicht prüfen:

1. Wie sind Objektbilder geometrisch im Feature-Raum angeordnet?
2. Welche topologische Struktur hat diese Punktwolke?
3. Welche Schicht passt am besten zu menschlichen Objekt-Repräsentationen?

Die Kernhypothese ist:

> Intermediäre ANN-Schichten stimmen stärker mit menschlichen Objektstrukturen überein als finale, aufgabenspezifische Schichten.

## 2. Minimale Analyseversion

Die erste Version sollte bewusst klein bleiben:

```text
Datensatz: kleine Bildauswahl
Modell: ResNet18
Layer: conv1, layer1, layer2, layer3, layer4, fc
Feature-Reduktion: Global Average Pooling bei Conv-Layern
Geometrie: cosine distance
TDA: persistent homology H0/H1
Visualisierung: PCA 2D/3D, Persistence Diagrams
Human-Vergleich: zunächst noch optional
```

Warum so klein?

- ResNet18 ist schnell und stabil.
- Global Average Pooling macht Conv-Aktivierungen direkt vergleichbar.
- Cosine Distance ist für neuronale Feature-Vektoren ein guter Startpunkt.
- H0/H1 reicht für den Einstieg in TDA.

## 3. Datenfluss

```text
stimuli_subset.csv
-> load images
-> preprocess images
-> run model
-> collect layer activations
-> pool/flatten activations
-> save feature matrices
-> compute distance matrices
-> compute PCA/TDA
-> compare layers
-> later: compare against human matrix
```

## 4. Stimuli

### Minimaler Start

Für den ersten technischen Lauf reichen Beispielbilder, notfalls auch nicht-THINGS-Bilder. Der wichtige Schritt ist:

> Die Pipeline muss Bilder laden, durch ein Modell schicken und Layer-Features speichern.

### BA-relevanter Start

Danach sollte auf THINGS gewechselt werden:

```text
100 bis 500 Bilder
mehrere Objektkonzepte
mehrere Bilder pro Konzept, wenn möglich
```

Wichtig ist eine eindeutige Tabelle:

```text
image_id, concept, category, image_path
```

## 5. Modell und Layer

### Einstiegsmodell

`ResNet18` ist der beste Start, weil:

- klein genug für Laptop,
- gut dokumentiert,
- in `torchvision` verfügbar,
- klare Layerstruktur.

### Geplante Layer

```text
conv1
layer1
layer2
layer3
layer4
fc
```

Interpretation:

- `conv1`: frühe visuelle Merkmale wie Kanten/Farben.
- `layer1`/`layer2`: mittlere visuelle Merkmale.
- `layer3`/`layer4`: abstraktere Objektmerkmale.
- `fc`: finale Klassifikationsrepräsentation.

## 6. Feature Extraction

Für jeden Layer soll eine Matrix entstehen:

```text
n_images x n_features
```

Bei vollständig verbundenen Layern ist das direkt gegeben.

Bei Convolutional Layers entstehen Aktivierungen wie:

```text
n_images x channels x height x width
```

Diese werden für die Minimalversion durch Global Average Pooling reduziert:

```text
n_images x channels
```

Begründung:

- Die Analyse soll zunächst Objekt-Repräsentationen vergleichen, nicht räumliche Aktivierungskarten.
- Pooling reduziert Speicherverbrauch.
- Pooling passt gut zu fMRI-Vergleichen, wo räumliche Details ebenfalls grob gemessen werden.

## 7. Geometrische Analyse

Pro Layer wird eine Distanzmatrix berechnet:

```text
n_images x n_images
```

Startmetrik:

```text
cosine distance = 1 - cosine similarity
```

Warum Cosine?

- Feature-Vektoren können unterschiedlich skaliert sein.
- Cosine fokussiert stärker auf Richtung/Pattern als auf absolute Aktivierungsgröße.

Später testbar:

- Euclidean distance
- Correlation distance

## 8. Topologische Analyse

Pro Layer wird die Feature-Punktwolke analysiert.

Startvarianten:

```text
original features
PCA 10D
PCA 3D
```

TDA-Ausgabe:

```text
H0: Clusterstruktur
H1: Schleifen/Löcher
```

Für die Bachelorarbeit ist wichtig:

- Nicht jedes Persistence Diagram ist leicht interpretierbar.
- Die zentrale Auswertung sollte Layer vergleichbar machen.
- Dazu braucht man später quantitative Distanzen zwischen Diagrams.

## 9. Menschliche Vergleichsdaten

### Erste sinnvolle Quelle

THINGS-Ähnlichkeitsurteile.

Warum zuerst diese?

- Semantisch nahe an der Forschungsfrage.
- Einfacher als fMRI/MEG.
- Direkt als menschliche Konzeptstruktur interpretierbar.

### Später

THINGS-data fMRI/MEG kann ergänzt werden, wenn die Pipeline stabil ist.

## 10. Vergleich ANN vs. Mensch

### Geometrisch

Vergleich zweier Distanzmatrizen:

```text
ANN layer distance matrix
vs.
human distance matrix
```

Methode:

- obere Dreiecksmatrix extrahieren,
- Spearman-Korrelation berechnen,
- Score pro Layer plotten.

### Topologisch

Vergleich zweier Persistence Diagrams:

```text
ANN layer persistence diagram
vs.
human persistence diagram
```

Mögliche Metriken:

- Wasserstein Distance
- Bottleneck Distance

Startempfehlung:

- Erst Geometrie robust implementieren.
- Dann TDA-Distanzen ergänzen.

## 11. Erwartete Ergebnisse

Mögliche Ergebnisform:

```text
Layer       Geometrie-Score       Topologie-Distanz
conv1       niedrig               hoch
layer1      mittel                mittel
layer2      hoch                  niedrig
layer3      hoch                  niedrig
layer4      mittel                mittel
fc          niedriger             höher
```

Das wäre kompatibel mit der Intermediate-Layer-Hypothese.

Wichtig: Das ist eine Erwartung, kein Ergebnis. Die Software soll genau prüfen, ob dieses Muster tatsächlich auftritt.

## 12. Designprinzipien

- Kleine Tests vor großen Downloads.
- Jeder Zwischenschritt wird gespeichert.
- Keine Analyse ohne klare Bildreihenfolge.
- Layernamen und Parameter müssen in Config-Dateien stehen.
- Plots dienen der Interpretation, CSV/NPY-Dateien der eigentlichen Analyse.


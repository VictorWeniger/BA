# Implementierungsplan

## Ziel

Aus der Agenda wird eine lauffähige Analysepipeline. Die Umsetzung erfolgt in kleinen, prüfbaren Schritten.

## Meilenstein 0: Planungsstruktur

Status: begonnen.

Aufgaben:

- Ordnerstruktur anlegen.
- Agenda schreiben.
- Technisches Design schreiben.
- Offene Fragen sammeln.

Ergebnis:

```text
AGENDA.md
README.md
docs/
```

## Meilenstein 1: Bildliste und Testdaten

Ziel: Die Pipeline braucht eine Tabelle mit Bildpfaden.

Minimaler Start:

```text
data/stimuli_subset.csv
```

Spalten:

```text
image_id, concept, category, image_path
```

Aufgaben:

1. Prüfen, ob THINGS-Bilder lokal vorhanden sind.
2. Falls nein: kleine Dummy-/Beispielbildliste definieren oder Downloadstrategie klären.
3. Loader schreiben, der diese CSV einliest.
4. Prüfen, ob alle Bildpfade existieren.

Akzeptanzkriterium:

> Ein Skript kann die CSV einlesen und die ersten Bilder erfolgreich laden.

## Meilenstein 2: ResNet18 laden

Ziel: Ein vortrainiertes Modell soll ein einzelnes Bild verarbeiten.

Aufgaben:

1. `torchvision.models.resnet18` laden.
2. Passende ImageNet-Transforms definieren.
3. Ein Bild durch das Modell schicken.
4. Output-Shape prüfen.

Akzeptanzkriterium:

> Ein einzelnes Bild erzeugt einen Output-Vektor mit 1000 Klassenwerten.

## Meilenstein 3: Layer Hooks

Ziel: Aktivierungen aus mehreren Schichten abgreifen.

Layer:

```text
conv1
layer1
layer2
layer3
layer4
fc
```

Aufgaben:

1. Forward Hooks registrieren.
2. Aktivierungen beim Forward Pass speichern.
3. Shapes pro Layer ausgeben.
4. Pooling für Conv-Layer implementieren.

Akzeptanzkriterium:

> Für ein Bild entstehen Feature-Vektoren aus allen Ziel-Layern.

## Meilenstein 4: Feature Extraction für Bildsubset

Ziel: Für alle Bilder im Subset Feature-Matrizen speichern.

Aufgaben:

1. DataLoader für Bildliste bauen.
2. Batchweise durch ResNet18 laufen lassen.
3. Features pro Layer sammeln.
4. Features als `.npy` speichern.

Ergebnis:

```text
outputs/features/resnet18_conv1.npy
outputs/features/resnet18_layer1.npy
outputs/features/resnet18_layer2.npy
outputs/features/resnet18_layer3.npy
outputs/features/resnet18_layer4.npy
outputs/features/resnet18_fc.npy
```

Akzeptanzkriterium:

> Jede Feature-Datei hat die Form `n_images x n_features`.

## Meilenstein 5: PCA und visuelle Kontrolle

Ziel: Schnell sehen, ob Layer-Repräsentationen sinnvoll aussehen.

Aufgaben:

1. 2D-PCA pro Layer.
2. 3D-PCA als HTML pro Layer.
3. Punkte nach Konzept/Kategorie färben.

Ergebnis:

```text
outputs/figures/pca2d_resnet18_layer2.png
outputs/figures/pca3d_resnet18_layer2.html
```

Akzeptanzkriterium:

> Die Plots lassen erkennen, ob Konzepte/Kategorien grob gruppieren.

## Meilenstein 6: Distanzmatrizen

Ziel: Geometrische Struktur pro Layer berechnen.

Aufgaben:

1. Feature-Dateien laden.
2. Cosine Distance berechnen.
3. Distanzmatrix speichern.
4. Heatmap plotten.

Ergebnis:

```text
outputs/geometry/resnet18_layer2_cosine.npy
outputs/figures/heatmap_resnet18_layer2_cosine.png
```

Akzeptanzkriterium:

> Jede Distanzmatrix ist quadratisch und entspricht der Bildreihenfolge in `stimuli_subset.csv`.

## Meilenstein 7: Erste TDA pro Layer

Ziel: Persistence Diagrams pro Layer berechnen.

Aufgaben:

1. Feature-Matrizen laden.
2. Optional PCA auf 3D/10D anwenden.
3. `ripser` ausführen.
4. Persistence Diagram speichern und plotten.

Ergebnis:

```text
outputs/tda/resnet18_layer2_pca10.pkl
outputs/figures/persistence_resnet18_layer2_pca10.png
```

Akzeptanzkriterium:

> Für jeden Layer existiert ein Persistence Diagram mit H0 und H1.

## Meilenstein 8: Menschliche Matrix einbauen

Ziel: Human-Vergleich vorbereiten.

Aufgaben:

1. THINGS-Ähnlichkeitsdaten lokalisieren.
2. Matrix oder Triplets einlesen.
3. Auf die verwendeten Konzepte/Bilder filtern.
4. Reihenfolge mit `stimuli_subset.csv` synchronisieren.
5. Menschliche Distanzmatrix speichern.

Ergebnis:

```text
outputs/human/things_similarity_subset.npy
```

Akzeptanzkriterium:

> Human-Matrix und ANN-Matrizen haben dieselbe Reihenfolge und Größe.

## Meilenstein 9: Geometrischer Human-Vergleich

Ziel: Layer nach menschlicher geometrischer Übereinstimmung ranken.

Aufgaben:

1. Upper Triangle der ANN-Matrizen extrahieren.
2. Upper Triangle der Human-Matrix extrahieren.
3. Spearman-Korrelation pro Layer berechnen.
4. Ergebnis als CSV speichern.
5. Layer-Plot erzeugen.

Ergebnis:

```text
outputs/results/geometric_alignment.csv
outputs/figures/geometric_alignment.png
```

Akzeptanzkriterium:

> Es gibt pro Layer einen Alignment-Score.

## Meilenstein 10: Topologischer Human-Vergleich

Ziel: Layer nach topologischer Nähe zu menschlicher Struktur ranken.

Aufgaben:

1. Human-Distanzmatrix in TDA-Eingabe überführen.
2. Persistence Diagram für Human-Daten berechnen.
3. Persistence Diagrams pro Layer vergleichen.
4. Bottleneck/Wasserstein Distance speichern.
5. Layer-Plot erzeugen.

Ergebnis:

```text
outputs/results/topological_alignment.csv
outputs/figures/topological_alignment.png
```

Akzeptanzkriterium:

> Es gibt pro Layer eine topologische Distanz zur Human-Struktur.

Aktueller Implementierungsstand:

- Skript ist angelegt: `scripts/compare_tda_to_human.py`.
- Voraussetzung: `ripser` und `persim` müssen in der aktiven Python-Umgebung sauber importieren.
- In der aktuellen Umgebung blockiert der `ripser`-Import teilweise; daher ist dieser Schritt vorbereitet, aber noch nicht Ende-zu-Ende validiert.

## Vorgeschlagene Reihenfolge ab jetzt

1. `stimuli_subset.csv` definieren.
2. ResNet18-Feature-Extraction auf Bildsubset implementieren.
3. PCA/3D-PCA wie bei MNIST erzeugen.
4. Distanzmatrizen berechnen.
5. TDA pro Layer berechnen.
6. Erst dann Human-Daten einbauen.

## Warum Human-Daten nicht sofort?

Der Human-Vergleich ist fehleranfällig, weil die Reihenfolge der Konzepte/Bilder exakt stimmen muss. Wenn Feature Extraction, Distanzmatrizen und TDA vorher stabil laufen, ist die Fehlersuche deutlich einfacher.

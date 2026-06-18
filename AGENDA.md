# Agenda für die BA-Analyse-Software

## Ziel der Software

Die Software soll untersuchen, ob bestimmte Schichten künstlicher neuronaler Netze eine ähnliche geometrische und topologische Struktur wie menschliche Objekt-Repräsentationen aufweisen.

Die zentrale Analysefrage lautet:

> Stimmen intermediäre ANN-Schichten stärker mit menschlichen Objekt-Repräsentationen überein als finale, aufgabenspezifische Schichten?

Die MNIST-Übung dient dabei als technischer Prototyp. Für die Bachelorarbeit wird MNIST später durch THINGS-Bilder und menschliche Vergleichsdaten ersetzt.

## Grundpipeline

```text
Objektbilder
-> ANN-Modell
-> Feature Extraction aus mehreren Layern
-> Feature-Matrizen pro Layer
-> Distanzmatrizen pro Layer
-> geometrischer Vergleich
-> topologische Analyse
-> Vergleich mit menschlichen Daten
```

## Phase 1: Projektstruktur aufbauen

Ziel: Eine saubere Softwarestruktur schaffen, damit Experimente reproduzierbar bleiben.

Geplante Ordner:

```text
ba_analyse_software/
├── AGENDA.md
├── README.md
├── configs/
├── data/
├── outputs/
├── src/
├── notebooks/
└── scripts/
```

Geplante Aufgaben:

- Projektordner strukturieren.
- Konfigurationsdateien für Experimente anlegen.
- Eine zentrale Stelle für Pfade, Modellnamen und Analyseparameter definieren.
- Klären, welche Daten lokal liegen und welche nur bei Bedarf geladen werden.

Ergebnis:

- Reproduzierbare Grundstruktur.
- Klare Trennung zwischen Daten, Code, Outputs und Notizen.

## Phase 2: Stimuli und Datenauswahl

Ziel: Eine kleine, kontrollierbare Bildauswahl aus THINGS vorbereiten.

Startpunkt:

- Nicht sofort den gesamten THINGS-Datensatz verwenden.
- Zuerst mit einer kleinen Teilmenge arbeiten, z. B. 100 bis 500 Bilder.
- Möglichst mehrere Objektkonzepte verwenden, damit sinnvolle Ähnlichkeitsstrukturen entstehen.

Geplante Aufgaben:

- THINGS-Bilder lokalisieren oder Downloadstrategie festlegen.
- Metadaten zu Bildern und Objektkonzepten einlesen.
- Kleine Testauswahl definieren.
- Bildpfade und Labels in einer Tabelle speichern.

Ergebnis:

```text
data/stimuli_subset.csv
```

Mögliche Spalten:

```text
image_id, concept, category, image_path
```

## Phase 3: ANN-Modell auswählen

Ziel: Ein Modell wählen, aus dem Layer-Aktivierungen extrahiert werden können.

Empfohlener Einstieg:

- `ResNet18`, weil es klein, schnell und gut dokumentiert ist.

Spätere Alternativen:

- `VGG16`
- `ResNet50`
- `CLIP`
- `ViT`

Geplante Aufgaben:

- Vortrainiertes Modell laden.
- Bildvorverarbeitung passend zum Modell definieren.
- Relevante Layer auswählen.
- Testen, ob ein einzelnes Bild korrekt durch das Modell läuft.

Ergebnis:

- Funktionierender Modell-Loader.
- Liste analysierbarer Layer.

## Phase 4: Feature Extraction

Ziel: Für jedes Bild Aktivierungen aus mehreren ANN-Schichten speichern.

Analogie zur MNIST-Übung:

```text
MNIST-Bild -> hidden1 / hidden2 / logits
THINGS-Bild -> layer1 / layer2 / layer3 / layer4 / final
```

Geplante Aufgaben:

- Forward Hooks oder Modell-Wrapper implementieren.
- Aktivierungen pro Layer extrahieren.
- Aktivierungen pro Bild speichern.
- Große Aktivierungstensoren sinnvoll flattenen oder poolen.

Wichtige Entscheidung:

- Für Convolutional Layers entstehen oft Tensoren wie:

```text
channels x height x width
```

Diese müssen für die Analyse in Vektoren überführt werden, z. B. durch:

- Global Average Pooling
- Flattening
- PCA nachträglich

Ergebnis:

```text
outputs/features/
├── layer1.npy
├── layer2.npy
├── layer3.npy
├── layer4.npy
└── final.npy
```

Jede Datei enthält ungefähr:

```text
Anzahl Bilder x Anzahl Features
```

## Phase 5: Geometrische Analyse

Ziel: Pro Layer eine Distanz- oder Ähnlichkeitsmatrix berechnen.

Geplante Aufgaben:

- Feature-Matrix pro Layer laden.
- Distanzen zwischen allen Bildern berechnen.
- Geeignete Metriken testen:

```text
cosine distance
euclidean distance
correlation distance
```

Ergebnis:

```text
outputs/geometry/
├── layer1_distance.npy
├── layer2_distance.npy
├── layer3_distance.npy
├── layer4_distance.npy
└── final_distance.npy
```

Interpretation:

- Wenn zwei Bilder ähnliche Feature-Vektoren haben, liegen sie im Repräsentationsraum nah beieinander.
- Geometrische Analyse fragt: Wie weit sind Konzepte oder Bilder voneinander entfernt?

## Phase 6: Menschliche Vergleichsdaten

Ziel: Eine menschliche Vergleichsstruktur einbauen.

Mögliche Vergleichsdaten:

1. THINGS-Ähnlichkeitsurteile
2. WordNet-Hierarchien
3. THINGS-data fMRI
4. THINGS-data MEG

Empfohlener Einstieg:

- Erst THINGS-Ähnlichkeitsurteile.
- Danach eventuell fMRI/MEG.

Geplante Aufgaben:

- Menschliche Ähnlichkeitsmatrix einlesen.
- Matrix auf dieselben Bilder oder Konzepte wie die ANN-Analyse einschränken.
- Sicherstellen, dass Reihenfolge der Bilder/Konzepte identisch ist.
- Ähnlichkeitswerte gegebenenfalls in Distanzen umwandeln.

Ergebnis:

```text
outputs/human/
└── human_similarity_or_distance.npy
```

## Phase 7: Vergleich Geometrie ANN vs. Mensch

Ziel: Prüfen, welche ANN-Schicht geometrisch am besten mit menschlichen Daten übereinstimmt.

Geplante Aufgaben:

- Pro Layer ANN-Distanzmatrix mit menschlicher Distanzmatrix vergleichen.
- Upper Triangle der Matrizen extrahieren.
- Spearman-Korrelation berechnen.
- Layer-Score speichern.

Ergebnis:

```text
outputs/results/geometric_alignment.csv
```

Mögliche Spalten:

```text
layer, metric, spearman_r, p_value
```

Erwarteter Plot:

```text
x-Achse: Layer
y-Achse: Korrelation mit menschlicher Matrix
```

Interpretation:

- Höhere Korrelation bedeutet stärkere geometrische Übereinstimmung.
- Wenn mittlere Layer höher liegen als finale Layer, unterstützt das die Intermediate-Layer-Hypothese.

## Phase 8: Topologische Datenanalyse

Ziel: Pro Layer die topologische Struktur der Feature-Punktwolke untersuchen.

Geplante Aufgaben:

- Feature-Matrix pro Layer laden.
- Optional PCA auf 3D, 10D oder 50D durchführen.
- Persistente Homologie berechnen.
- Persistence Diagrams speichern.
- H0 und H1 betrachten:

```text
H0: Cluster / verbundene Komponenten
H1: Schleifen / Löcher
```

Ergebnis:

```text
outputs/tda/
├── layer1_persistence.pkl
├── layer2_persistence.pkl
├── layer3_persistence.pkl
├── layer4_persistence.pkl
└── final_persistence.pkl
```

Zusätzliche Plots:

```text
outputs/figures/
├── persistence_layer1.png
├── persistence_layer2.png
├── persistence_layer3.png
├── persistence_layer4.png
└── persistence_final.png
```

Interpretation:

- Punkte weit weg von der Diagonalen zeigen stabile topologische Merkmale.
- Viele kurzlebige Punkte nahe der Diagonalen sprechen eher für Rauschen oder schwache Strukturen.

## Phase 9: Topologischer Vergleich ANN vs. Mensch

Ziel: Prüfen, welche ANN-Schicht topologisch am besten zu menschlichen Daten passt.

Mögliche Vergleichsmethoden:

- Bottleneck Distance
- Wasserstein Distance
- Persistence Images
- Persistence Landscapes

Geplante Aufgaben:

- Persistence Diagram für menschliche Vergleichsdaten berechnen.
- Persistence Diagram pro ANN-Layer berechnen.
- Topologische Distanz zwischen Mensch und jedem Layer bestimmen.
- Layer mit geringster Distanz identifizieren.

Ergebnis:

```text
outputs/results/topological_alignment.csv
```

Mögliche Spalten:

```text
layer, tda_mode, homology_dimension, distance_to_human
```

Erwarteter Plot:

```text
x-Achse: Layer
y-Achse: topologische Distanz zu menschlichen Daten
```

Interpretation:

- Niedrigere Distanz bedeutet stärkere topologische Übereinstimmung.
- Wenn mittlere Layer niedrigere Distanz zeigen als finale Layer, unterstützt das deine zentrale Hypothese.

## Phase 10: Visualisierung und Bericht

Ziel: Ergebnisse so darstellen, dass sie in der Bachelorarbeit verständlich sind.

Benötigte Plots:

- Beispielbilder aus dem Stimulusset.
- 2D-PCA oder UMAP pro ausgewähltem Layer.
- 3D-PCA optional als explorative Visualisierung.
- Distanzmatrix-Heatmaps für ausgewählte Layer.
- Menschliche Distanzmatrix als Heatmap.
- Geometrischer Alignment-Plot über Layer.
- Persistence Diagrams ausgewählter Layer.
- Topologischer Alignment-Plot über Layer.

Ergebnis:

```text
outputs/figures/
```

## Phase 11: Minimal Viable Analysis

Die erste vollständige Version sollte klein bleiben.

Minimaler Umfang:

```text
100 bis 500 THINGS-Bilder
1 Modell: ResNet18
5 Layer: inputnah, layer1, layer2, layer3, final
1 geometrische Metrik: cosine distance
1 menschliche Vergleichsquelle: THINGS similarity judgments
1 TDA-Verfahren: persistent homology mit H0 und H1
```

Erste Forschungsantwort:

> Welche ResNet18-Schicht zeigt die stärkste geometrische und topologische Übereinstimmung mit menschlichen Ähnlichkeitsurteilen?

## Phase 12: Erweiterungen

Erst nach der Minimalversion:

- Mehr Bilder.
- Mehr Modelle.
- Vergleich ResNet vs. CLIP.
- Vergleich unterschiedlicher Distanzmetriken.
- Vergleich unterschiedlicher TDA-Parameter.
- Einbindung von fMRI-Daten.
- Einbindung von MEG-Daten.
- Robustheitsanalysen mit verschiedenen PCA-Dimensionen.

## Konkrete nächste Schritte

1. Projektstruktur anlegen.
2. Kleinen THINGS-Stimulus-Subset definieren.
3. ResNet18 Feature Extraction implementieren.
4. Features für mehrere Layer speichern.
5. PCA- und TDA-Plots wie bei MNIST erzeugen.
6. Danach erst menschliche Ähnlichkeitsdaten einbauen.

## Merksatz

Die MNIST-Pipeline war:

```text
Ziffernbild -> ANN-Schicht -> Feature-Vektor -> PCA/TDA
```

Die BA-Pipeline wird:

```text
Objektbild -> ANN-Schicht -> Feature-Vektor -> Geometrie/TDA -> Vergleich mit menschlicher Objektstruktur
```

## Offene methodische Punkte (Hauptansatz: vertex-weise Oberflächen-Suchlicht)

### Noise Ceiling anpassen

Das bisherige Noise-Ceiling-Skript (`compute_noise_ceiling.py`) ist auf ROI-basierte
Analyse ausgelegt. Es muss geprüft werden, ob und wie es auf den neuen vertex-weisen
Ansatz übertragen werden kann.

Möglichkeiten:
- Noise Ceiling pro Vertex über split-half der drei Probanden berechnen
- Alternativ: Noise Ceiling nur für die Glasser-Arealzusammenfassung (supplementäre Analyse)
- Klären, ob ein vertex-weiser Noise Ceiling bei n=3 Probanden statistisch sinnvoll ist

### Statistische Absicherung von CKA und TDA

Für RSA existiert bereits ein Permutationstest (Shuffle der Stimulusreihenfolge).
Für CKA und TDA fehlt eine gleichwertige statistische Absicherung.

Mögliche Wege:
- Permutationstest analog zu RSA: Stimulusreihenfolge permutieren, Null-Verteilung der
  CKA- bzw. Wasserstein-Werte pro Vertex schätzen
- Bootstrap über Stimuli
- Recherche: Welche statistischen Verfahren für vertex-weise CKA und TDA-Vergleiche
  in der Literatur verwendet werden

### Signifikanztest RSA an Ansatzwechsel anpassen

Der bestehende RSA-Permutationstest wurde für ROI-basierte Analyse entwickelt.
Im vertex-weisen Suchlicht gilt:
- Multiple-Comparisons-Korrektur nötig (z. B. FDR oder Cluster-basiert)
- Permutationstest muss ggf. pro Vertex laufen, was rechenintensiv ist
- Prüfen, ob eine globale Permutation (Stimulusreihenfolge einmal shufflen, alle Vertices
  neu berechnen) ausreicht oder ob vertex-spezifische Null-Verteilungen benötigt werden


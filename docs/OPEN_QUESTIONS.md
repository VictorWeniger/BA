# Offene Fragen und Risiken

## 1. Wo liegen die THINGS-Bilder?

Status: offen.

Die Software braucht später konkrete Bildpfade. Zu klären ist:

- Sind THINGS-Bilder schon lokal vorhanden?
- Müssen sie heruntergeladen werden?
- Welche Version/Struktur hat der Datensatz?

Risiko:

> Ohne klare Bildpfade kann keine echte Feature Extraction auf THINGS laufen.

Vorläufige Lösung:

> Erst die Pipeline mit einer kleinen lokalen Beispielbildliste testen.

## 2. Bild- oder Konzeptniveau?

Es gibt zwei mögliche Analyseebenen:

### Bildniveau

Jedes Bild ist ein Datenpunkt.

Vorteil:

- Direkter Anschluss an ANN-Features.
- Einfacher technisch umzusetzen.

Nachteil:

- Menschliche THINGS-Ähnlichkeitsdaten liegen eventuell eher auf Konzeptniveau vor.

### Konzeptniveau

Mehrere Bilder eines Konzepts werden gemittelt.

Vorteil:

- Besserer Anschluss an semantische Ähnlichkeitsurteile.
- Rauschen einzelner Bilder wird reduziert.

Nachteil:

- Man braucht genügend Bilder pro Konzept.
- Mittelung kann visuelle Varianz verlieren.

Startempfehlung:

> Erst Bildniveau für die technische Pipeline, danach Konzeptmittelung für Human-Vergleiche prüfen.

## 3. Welche Layer sind wirklich intermediär?

Bei ResNet18 ist die grobe Einteilung:

```text
früh: conv1, layer1
intermediär: layer2, layer3
spät: layer4, fc
```

Das ist pragmatisch, aber nicht absolut.

Risiko:

> Die Definition von "intermediär" muss in der Bachelorarbeit sauber begründet werden.

## 4. Pooling oder Flattening?

Convolutional Layers liefern räumliche Aktivierungskarten.

Optionen:

- Global Average Pooling
- Flattening
- Spatial PCA

Startempfehlung:

> Global Average Pooling, weil es robust, klein und gut interpretierbar ist.

Risiko:

> Pooling kann räumliche Information verlieren.

Später:

> Robustheitsanalyse mit Flattening oder anderer Pooling-Strategie.

## 5. Welche Distanzmetrik?

Start:

```text
cosine distance
```

Alternativen:

- Euclidean distance
- Correlation distance

Risiko:

> Unterschiedliche Metriken können unterschiedliche Layer-Rankings erzeugen.

Startempfehlung:

> Erst cosine. Danach prüfen, ob Ergebnisse mit correlation ähnlich bleiben.

## 6. Welche TDA-Dimension?

Optionen:

```text
Original-Features
PCA 3D
PCA 10D
PCA 50D
```

Startempfehlung:

> PCA 10D und Original-Features vergleichen.

Begründung:

- PCA 3D ist gut zum Verstehen, aber wahrscheinlich zu grob.
- PCA 10D ist praktisch und näher an der echten Struktur.
- Original ist konzeptuell sauber, aber rechenintensiver.

## 7. Wie viele Bilder sind praktikabel?

TDA skaliert schlecht mit sehr vielen Punkten.

Start:

```text
100 bis 500 Bilder
```

Später:

- Subsampling
- Konzeptmittelung
- Landmark-basierte Methoden

Risiko:

> Zu viele Bilder machen TDA langsam oder instabil.

## 8. Wie wird Human-TDA definiert?

Für ANN-Layer gibt es Feature-Punktwolken.

Für menschliche Ähnlichkeitsdaten gibt es eher eine Distanzmatrix.

Mögliche Lösung:

> TDA kann auch auf vorberechneten Distanzmatrizen laufen.

Zu klären:

- Welche Form haben die THINGS-Ähnlichkeitsdaten?
- Sind sie vollständig oder lückenhaft?
- Müssen fehlende Werte interpoliert oder ausgeschlossen werden?

## 9. Was gilt als Ergebnis?

Die Software soll am Ende nicht nur schöne Plots erzeugen, sondern klare Tabellen:

```text
geometric_alignment.csv
topological_alignment.csv
```

Wichtig:

> Die Bachelorarbeit braucht eine interpretierbare Layer-Rangfolge.

## 10. Was ist der erste wirklich sinnvolle nächste Schritt?

Nicht sofort fMRI, nicht sofort kompletter THINGS-Datensatz.

Erster sinnvoller Schritt:

> ResNet18-Feature-Extraction auf einer kleinen Bildliste implementieren und die MNIST-Visualisierungen replizieren.

Danach:

> THINGS-Ähnlichkeitsdaten einbauen.


# Aktuelle Learnings für die Bachelorarbeit

Stand: erster echter ResNet18-vs-fMRI-Lauf mit S1, 100 aggregierten THINGS-Stimuli und vier ROIs.

## 1. Die Pipeline funktioniert grundsätzlich

Ein wichtiges methodisches Learning ist:

> Es ist möglich, THINGS-Bilder, ANN-Repräsentationen und fMRI-Repräsentationen stimulusgenau miteinander zu verbinden.

Konkret wurde gezeigt:

```text
fMRI-Stimulusliste
-> passende THINGS-Bilddateien
-> ResNet18 Layer-Features
-> ANN-Distanzmatrizen
-> fMRI-ROI-Distanzmatrizen
-> geometrischer Vergleich
```

Das ist für die Bachelorarbeit wichtig, weil damit die technische Grundlage für den eigentlichen Vergleich steht.

## 2. fMRI-Betas können als menschliche Feature-Repräsentationen behandelt werden

Die processed-Daten haben die Struktur:

```text
S1_betas_vol.npy: Trials x Voxels
```

Für die Analyse bedeutet das:

> Jeder Stimulus kann durch einen fMRI-Beta-Vektor repräsentiert werden.

Damit sind fMRI-Daten formal ähnlich nutzbar wie ANN-Layer-Features:

```text
ANN:  Stimulus -> Layer-Aktivierungsvektor
fMRI: Stimulus -> Voxel-Aktivierungsvektor
```

Aus beiden lassen sich Distanzmatrizen berechnen. Dadurch wird ein direkter geometrischer Vergleich möglich.

## 3. Die Stimulus-Reihenfolge ist entscheidend

Ein zentrales praktisches Learning:

> ANN- und fMRI-Distanzmatrizen dürfen nur verglichen werden, wenn sie exakt dieselben Stimuli in exakt derselben Reihenfolge enthalten.

Deshalb wurde aus der aggregierten fMRI-Stimulusliste eine passende ANN-Stimulusliste erzeugt:

```text
data/ann_S1_matched_stimuli.csv
```

Dieses Matching ist methodisch entscheidend. Ohne korrektes Matching wären alle Korrelationen zwischen ANN und fMRI bedeutungslos.

## 4. Aggregation reduziert Trial-Rauschen

Im aktuellen Lauf wurden 500 Test-Trials auf 100 Stimuli aggregiert:

```text
500 Trials -> 100 Stimuli
```

Dabei wurden wiederholte Messungen desselben Stimulus gemittelt.

Interpretation:

> Die Analyse arbeitet nicht mit einzelnen verrauschten Trial-Messungen, sondern mit stabileren stimulusbezogenen fMRI-Repräsentationen.

Das ist für fMRI sinnvoll, weil einzelne Trial-Betas stark rauschanfällig sein können.

## 5. ResNet18 ist ein geeigneter Baseline-Start

ResNet18 lief erfolgreich auf dem Rechner mit:

```text
ImageNet-pretrained weights
MPS backend
```

Die extrahierten Layer waren:

```text
conv1
layer1
layer2
layer3
layer4
fc
```

Damit ergibt sich eine einfache Hierarchie:

```text
früh: conv1 / layer1
mittel: layer2 / layer3
spät: layer4
final: fc
```

Für die Bachelorarbeit ist ResNet18 ein sinnvoller erster Vergleich, weil es klein, etabliert und interpretierbar ist.

## 6. LOC zeigt das klarste Intermediate-Layer-Muster

Das wichtigste erste Ergebnis:

> Im LOC ist die geometrische Übereinstimmung mit ResNet18 für `layer3` am höchsten, knapp gefolgt von `layer2`.

Spearman-Werte für LOC:

```text
conv1:  -0.0169
layer1:  0.1005
layer2:  0.1470
layer3:  0.1536
layer4:  0.0989
fc:      0.0124
```

Interpretation:

> Die finale Klassifikationsschicht `fc` stimmt kaum mit der LOC-Geometrie überein, während mittlere/spätere Feature-Layer besser passen.

Das ist kompatibel mit der Intermediate-Layer-Hypothese.

## 7. IT zeigt ein schwächeres, aber ähnliches Muster

Für IT ist das Alignment insgesamt schwächer als für LOC.

Spearman-Werte für IT:

```text
conv1:  -0.0160
layer1:  0.0408
layer2:  0.0646
layer3:  0.0624
layer4:  0.0445
fc:     -0.0124
```

Interpretation:

> Auch in IT ist nicht die finale `fc`-Schicht am besten, sondern eher `layer2`/`layer3`.

Das Muster ist aber weniger stark als in LOC und muss über weitere Subjects geprüft werden.

## 8. V1 und hV4 zeigen im aktuellen Lauf kein klares positives Spearman-Alignment

Für V1 und hV4 sind die Spearman-Werte schwach oder negativ.

Das kann mehrere Gründe haben:

- Nur 100 aggregierte Stimuli.
- fMRI-Rauschen.
- ROI-Auswahl.
- ResNet18-Distanzstruktur passt in diesem Setup schlechter zu frühen visuellen ROIs.
- Cosine-Distanz auf gepoolten Features ist möglicherweise nicht optimal für frühe visuelle Ebenen.

Wichtig:

> Daraus sollte noch nicht geschlossen werden, dass V1/hV4 grundsätzlich nicht mit ANN-Schichten alignen.

Dieser Befund ist vorläufig.

## 9. Spearman und CKA erzählen unterschiedliche Geschichten

Nach Spearman/RDM-Vergleich ist für LOC `layer3` am stärksten.

Nach CKA ist dagegen in allen ROIs `layer4` am höchsten:

```text
V1:  layer4
hV4: layer4
LOC: layer4
IT:  layer4
```

Das ist kein direkter Widerspruch, weil beide Maße unterschiedliche Aspekte vergleichen.

### Spearman/RDM

Vergleicht:

```text
paarweise Distanzen zwischen Stimuli
```

Das passt direkt zur repräsentationalen Geometrie.

### CKA

Vergleicht:

```text
Ähnlichkeit der Feature-Räume / Aktivierungsmuster
```

CKA ist also eher ein globales Repräsentationsähnlichkeitsmaß.

Learning:

> Die Wahl des Alignment-Maßes beeinflusst, welche Schicht als am besten passend erscheint.

Für die Bachelorarbeit muss daher klar begründet werden, welches Maß für welche Fragestellung verwendet wird.

## 10. Für die Abstract-Frage ist Spearman/RDM besonders relevant

Der Abstract spricht stark über repräsentationale Geometrie und Distanzen zwischen Konzepten.

Deshalb ist der Spearman-Vergleich zwischen Distanzmatrizen besonders passend:

```text
ANN-Distanzmatrix vs. fMRI-Distanzmatrix
```

Das beantwortet direkt:

> Sind die paarweisen Beziehungen zwischen Stimuli im ANN ähnlich organisiert wie im Gehirn?

## 11. Die finale Schicht ist nicht automatisch die gehirnähnlichste

Ein wichtiges inhaltliches Learning:

> Die finale ResNet18-Schicht `fc` zeigt im aktuellen Lauf kein starkes geometrisches Alignment mit LOC oder IT.

Das passt zur Idee, dass finale Schichten stark auf die Trainingsaufgabe zugeschnitten sind.

Für ResNet18 bedeutet das:

```text
fc = ImageNet-Klassifikationsoutput
```

Diese Schicht ist also wahrscheinlich stärker auf ImageNet-Kategorien optimiert als auf eine flexible menschliche Objektstruktur.

## 12. Der aktuelle Befund unterstützt vorsichtig die Intermediate-Layer-Idee

Der aktuelle Lauf stützt die Hypothese nicht endgültig, ist aber kompatibel mit ihr.

Vorsichtige Formulierung:

> In einem ersten explorativen Vergleich zeigte sich für LOC die höchste geometrische Übereinstimmung mit einer intermediären/späteren ResNet18-Schicht (`layer3`), während die finale Klassifikationsschicht nur schwaches Alignment zeigte.

Nicht schreiben:

> ResNet18 beweist, dass intermediäre Schichten dem Gehirn entsprechen.

Besser:

> Die Ergebnisse liefern erste Hinweise darauf, dass nicht-finale Schichten für die Modellierung visueller fMRI-Repräsentationen relevanter sein könnten als die finale Klassifikationsschicht.

## 13. Die Ergebnisse sind noch explorativ

Die wichtigsten Einschränkungen:

```text
Nur Subject S1
Nur 100 aggregierte Stimuli
Nur vier ROIs
Nur ResNet18
Noch keine TDA
Noch keine Robustheitsanalysen
```

Daraus folgt:

> Die aktuellen Ergebnisse sollten als Pipeline-Validierung und explorativer Vorbefund verstanden werden, nicht als finales Ergebnis der Bachelorarbeit.

## 14. Was als nächstes wissenschaftlich sinnvoll ist

Die nächsten Schritte sollten prüfen, ob das Muster stabil ist:

1. Denselben Lauf für S2 und S3 wiederholen.
2. Prüfen, ob LOC und IT über Subjects ähnliche Layer-Präferenzen zeigen.
3. Mehr Stimuli einbeziehen.
4. TDA auf ausgewählten ROIs und Layern ergänzen.
5. Später ein zweites Modell vergleichen, z. B. ResNet50 oder CLIP.

## 15. Mögliche BA-Formulierung für den Methodenteil

Eine mögliche Formulierung:

> Für jeden Stimulus wurden sowohl künstliche Repräsentationen aus mehreren Schichten eines ImageNet-vortrainierten ResNet18 als auch fMRI-basierte Repräsentationen aus ausgewählten visuellen ROIs extrahiert. Anschließend wurden für jede Repräsentation paarweise Cosine-Distanzmatrizen berechnet. Die geometrische Übereinstimmung zwischen Modell- und Hirnrepräsentationen wurde über Spearman-Korrelationen zwischen den oberen Dreiecksmatrizen der jeweiligen Distanzmatrizen quantifiziert.

## 16. Mögliche BA-Formulierung für erste Ergebnisse

Eine mögliche vorsichtige Formulierung:

> In einem ersten explorativen Lauf mit S1 und 100 aggregierten THINGS-Stimuli zeigte sich im LOC das stärkste geometrische Alignment mit `layer3` von ResNet18, gefolgt von `layer2`. Die finale Klassifikationsschicht zeigte dagegen nur eine sehr geringe Übereinstimmung. Dieses Muster ist konsistent mit der Annahme, dass intermediäre Schichten künstlicher neuronaler Netzwerke visuelle Objektstrukturen erfassen, die stärker mit menschlichen Repräsentationen übereinstimmen als finale, aufgabenspezifische Ausgabeschichten.

## 17. Mögliche BA-Formulierung für Einschränkungen

Eine mögliche Formulierung:

> Da die bisherige Analyse nur auf einem Subject, einer begrenzten Stimulusmenge und einem einzelnen Modell basiert, sind die Ergebnisse als explorativ zu interpretieren. Weitere Analysen über mehrere Subjects, größere Stimulusmengen und zusätzliche Modelle sind notwendig, um die Stabilität des beobachteten Intermediate-Layer-Musters zu prüfen.


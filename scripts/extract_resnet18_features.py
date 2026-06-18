"""Feature-Extraktion aus ResNet-18/ResNet-50.

Für jede Zielschicht wird ein Forward-Hook registriert, der die Aktivierungen
abfängt. Konvolutionale Ausgaben (n, C, H, W) werden per Global Average Pooling
auf (n, C) reduziert. Die resultierenden Matrizen werden als .npy gespeichert.

In der aktuellen BA-Pipeline wird über die Config `resnet18` geladen. Die
Funktion unterstützt `resnet50` nur noch, weil ältere Experimente damit liefen.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.utils import ensure_dir, load_config, project_path, read_stimuli_csv, write_csv


class ImageTableDataset(Dataset):
    """PyTorch-Dataset, das Bilder aus der gematchten Stimulus-CSV lädt.

    Wendet ImageNet-Normalisierung an (Mittelwert und Standardabweichung
    aus dem ImageNet-Trainingsdatensatz). Das ist notwendig, weil die
    torchvision-ResNet-Gewichte mit genau dieser Vorverarbeitung trainiert
    wurden.
    """

    def __init__(self, rows, image_path_column):
        self.rows = rows
        self.image_path_column = image_path_column
        # Standard ImageNet Preprocessing
        self.transform = transforms.Compose(
            [
                transforms.Resize(256),           # Kürzere Seite auf 256px
                transforms.CenterCrop(224),        # Mittleres 224×224 Quadrat
                transforms.ToTensor(),             # [0,255] → [0,1]
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],    # ImageNet RGB-Mittelwerte
                    std=[0.229, 0.224, 0.225],     # ImageNet RGB-Standardabweichungen
                ),
            ]
        )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image_path = project_path(row[self.image_path_column])
        image = Image.open(image_path).convert("RGB")
        return self.transform(image), index


def get_device():
    """Bestes verfügbares Rechengerät wählen: MPS (Apple) > CUDA > CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(model_name, pretrained=True, allow_untrained_fallback=True):
    """ResNet-Modell nach Name laden. Unterstützt resnet18 und resnet50."""

    loaders = {
        "resnet18": (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1),
        "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V2),
    }
    if model_name not in loaders:
        raise ValueError(f"Unsupported model: {model_name}. Choose from {list(loaders)}")

    model_fn, weights = loaders[model_name]

    if pretrained:
        try:
            return model_fn(weights=weights)
        except Exception as exc:
            if not allow_untrained_fallback:
                raise
            print(f"Warning: pretrained weights unavailable, using untrained {model_name}. Reason: {exc}")

    return model_fn(weights=None)


def pool_activation(activation):
    """Global Average Pooling: (n, C, H, W) → (n, C).

    Konvolutionale Schichten geben 4D-Tensoren aus.
    GAP mittelt über alle räumlichen Positionen (H×W) — jeder Kanal
    wird zu einem einzigen Wert. Ergebnis: ein Vektor der Länge C pro Bild.
    """
    if activation.ndim == 4:
        activation = activation.mean(dim=(2, 3))  # Höhe und Breite wegmitteln
    return activation.detach().cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/minimal_resnet18.json")
    args = parser.parse_args()

    config = load_config(args.config)
    data_config = config["data"]
    model_config = config["model"]
    feature_config = config["features"]

    # Die CSV wurde vorher aus den fMRI-Stimuli erzeugt. Dadurch sieht das ANN
    # exakt dieselben Bilder und dieselbe Reihenfolge wie die fMRI-RDM.
    rows = read_stimuli_csv(data_config["stimuli_csv"])
    if not rows:
        raise RuntimeError("stimuli_subset.csv ist leer. Dummy-Stimuli generieren oder echte Bildpfade eintragen.")

    device = get_device()
    print(f"Device: {device}")

    # Modellname und Schichten kommen aus der Config. Für die BA ist das
    # `resnet18` mit ImageNet-Gewichten.
    model = load_model(
        model_config.get("name", "resnet50"),
        pretrained=model_config.get("pretrained", True),
        allow_untrained_fallback=model_config.get("allow_untrained_fallback", True),
    ).to(device)
    model.eval()  # Dropout und BatchNorm auf Inferenz-Modus setzen

    # --- Forward-Hooks registrieren ---
    # Ein Hook fängt die Ausgabe eines Moduls ab, ohne den Vorwärtsdurchlauf zu ändern.
    layer_names = model_config["layers"]
    captured = {}   # Zwischenspeicher: layer_name → aktueller Aktivierungstensor
    hooks = []

    def make_hook(name):
        def hook(_module, _inputs, output):
            captured[name] = output
        return hook

    # `named_modules()` gibt alle Module im ResNet zurück. So können wir per
    # String-Namen wie "layer2" oder "fc" Hooks an genau die Zielschichten
    # hängen, ohne den Modellcode selbst umzubauen.
    module_lookup = dict(model.named_modules())
    for layer_name in layer_names:
        if layer_name not in module_lookup:
            raise KeyError(f"Schicht nicht in ResNet gefunden: {layer_name}")
        hooks.append(module_lookup[layer_name].register_forward_hook(make_hook(layer_name)))

    # --- Bilder in Batches durch das Netz schieben ---
    dataset = ImageTableDataset(rows, data_config["image_path_column"])
    loader = DataLoader(dataset, batch_size=feature_config.get("batch_size", 32), shuffle=False)

    features = {layer_name: [] for layer_name in layer_names}
    image_indices = []

    with torch.no_grad():  # Keine Gradienten nötig — spart Speicher
        for images, indices in loader:
            images = images.to(device)
            captured.clear()
            _ = model(images)  # Vorwärtsdurchlauf — Hooks füllen captured

            for layer_name in layer_names:
                features[layer_name].append(pool_activation(captured[layer_name]))

            image_indices.extend(indices.numpy().tolist())

    # Hooks wieder entfernen
    for hook in hooks:
        hook.remove()

    # --- Aktivierungsmatrizen speichern ---
    output_dir = ensure_dir(feature_config["output_dir"])
    experiment_name = config["experiment_name"]

    for layer_name, chunks in features.items():
        # Batches zusammenkleben: Liste von (batch, C) → (n_stimuli, C)
        matrix = np.concatenate(chunks, axis=0)
        output_path = output_dir / f"{experiment_name}_{layer_name}.npy"
        np.save(output_path, matrix)
        print(f"Gespeichert {layer_name}: {matrix.shape} → {output_path}")

    # Bildreihenfolge für spätere Zuordnung sichern
    ordered_rows = [rows[index] for index in image_indices]
    write_csv(
        ordered_rows,
        output_dir / f"{experiment_name}_image_order.csv",
        fieldnames=list(ordered_rows[0].keys()),
    )


if __name__ == "__main__":
    main()

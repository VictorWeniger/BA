import argparse
import math
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.utils import PROJECT_ROOT, write_csv


CONCEPTS = [
    ("red_circle", "round"),
    ("blue_circle", "round"),
    ("green_square", "angular"),
    ("yellow_square", "angular"),
    ("purple_triangle", "angular"),
    ("orange_triangle", "angular"),
]

COLORS = {
    "red": (220, 70, 70),
    "blue": (70, 120, 220),
    "green": (80, 170, 100),
    "yellow": (220, 190, 60),
    "purple": (150, 90, 190),
    "orange": (230, 140, 60),
}


def jitter(value, amount=12):
    return int(value + random.randint(-amount, amount))


def draw_shape(concept, output_path, size=224):
    """Create one simple image with a colored geometric shape."""

    color_name, shape = concept.split("_", 1)
    color = COLORS[color_name]

    image = Image.new("RGB", (size, size), (245, 245, 245))
    draw = ImageDraw.Draw(image)

    center_x = jitter(size // 2, 18)
    center_y = jitter(size // 2, 18)
    radius = random.randint(55, 78)

    if shape == "circle":
        box = [center_x - radius, center_y - radius, center_x + radius, center_y + radius]
        draw.ellipse(box, fill=color)
    elif shape == "square":
        angle = random.uniform(-0.35, 0.35)
        points = []
        for base_angle in [math.pi / 4, 3 * math.pi / 4, 5 * math.pi / 4, 7 * math.pi / 4]:
            a = base_angle + angle
            points.append((center_x + radius * math.cos(a), center_y + radius * math.sin(a)))
        draw.polygon(points, fill=color)
    elif shape == "triangle":
        angle = random.uniform(-0.5, 0.5)
        points = []
        for base_angle in [-math.pi / 2, math.pi / 6, 5 * math.pi / 6]:
            a = base_angle + angle
            points.append((center_x + radius * math.cos(a), center_y + radius * math.sin(a)))
        draw.polygon(points, fill=color)

    image.save(output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-per-concept", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)

    image_dir = PROJECT_ROOT / "data" / "dummy_images"
    image_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for concept, category in CONCEPTS:
        for index in range(args.images_per_concept):
            image_id = f"{concept}_{index:03d}"
            rel_path = Path("data") / "dummy_images" / f"{image_id}.png"
            abs_path = PROJECT_ROOT / rel_path
            draw_shape(concept, abs_path)
            rows.append(
                {
                    "image_id": image_id,
                    "concept": concept,
                    "category": category,
                    "image_path": str(rel_path),
                }
            )

    write_csv(
        rows,
        "data/stimuli_subset.csv",
        fieldnames=["image_id", "concept", "category", "image_path"],
    )
    print(f"Generated {len(rows)} images and data/stimuli_subset.csv")


if __name__ == "__main__":
    main()

"""Calculate channel statistics using only the training split."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

DEFAULT_DATA_DIR = Path("data/processed/cifar10/v1")


def calculate_statistics(
    annotations_path: Path,
    images_dir: Path,
) -> dict[str, object]:
    """Calculate RGB mean and standard deviation in the [0, 1] range."""
    with annotations_path.open(encoding="utf-8") as file:
        annotations = json.load(file)

    images = annotations["images"]
    if not images:
        raise ValueError("Training split is empty")

    channel_sum = np.zeros(3, dtype=np.float64)
    channel_squared_sum = np.zeros(3, dtype=np.float64)
    pixel_count = 0

    for index, annotation in enumerate(images, start=1):
        image_path = images_dir / annotation["file_name"]

        with Image.open(image_path) as image:
            pixels = np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0

        if pixels.shape != (32, 32, 3):
            raise ValueError(f"Unexpected image shape: {image_path}")

        channel_sum += pixels.sum(axis=(0, 1))
        channel_squared_sum += np.square(pixels).sum(axis=(0, 1))
        pixel_count += pixels.shape[0] * pixels.shape[1]

        if index % 5000 == 0:
            print(f"Processed {index}/{len(images)} images")

    mean = channel_sum / pixel_count
    variance = channel_squared_sum / pixel_count - np.square(mean)
    std = np.sqrt(np.maximum(variance, 0.0))

    return {
        "split": "train",
        "num_images": len(images),
        "pixel_range": [0.0, 1.0],
        "channel_order": ["R", "G", "B"],
        "std_ddof": 0,
        "mean": mean.tolist(),
        "std": std.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Calculate CIFAR-10 training channel statistics.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = parser.parse_args()

    annotations_path = args.data_dir / "annotations/splits/train.json"
    images_dir = args.data_dir / "images/train"
    output_path = args.data_dir / "stats.json"

    statistics = calculate_statistics(annotations_path, images_dir)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(statistics, file, indent=2)
        file.write("\n")

    print(f"Saved statistics to {output_path}")
    print(f"Mean: {statistics['mean']}")
    print(f"Std:  {statistics['std']}")


if __name__ == "__main__":
    main()

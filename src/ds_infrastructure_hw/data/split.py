"""Create a reproducible stratified train/validation split."""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_DATA_DIR = Path("data/processed/cifar10/v1")


def load_annotations(path: Path) -> dict[str, Any]:
    """Load and validate annotations."""
    with path.open(encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict) or "classes" not in data or "images" not in data:
        raise ValueError("Invalid annotation file structure")

    if not isinstance(data["images"], list):
        raise ValueError("The images field must be a list")

    return data


def stratified_split(
    annotations: dict[str, Any],
    val_size: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split images while preserving class proportions."""
    images = annotations["images"]
    classes = annotations["classes"]

    if val_size <= 0 or val_size >= len(images):
        raise ValueError("val_size must be between 1 and the dataset size - 1")

    labels = {int(image["label"]) for image in images}
    if not labels.issubset(set(range(len(classes)))):
        raise ValueError("Annotations contain an unknown class label")

    rng = np.random.default_rng(seed)

    train_images: list[dict[str, Any]] = []
    val_images: list[dict[str, Any]] = []

    for label in range(len(classes)):
        class_images = [image for image in images if int(image["label"]) == label]

        if not class_images:
            raise ValueError(f"Class {label} has no images")

        class_val_size = round(val_size * len(class_images) / len(images))

        if class_val_size <= 0 or class_val_size >= len(class_images):
            raise ValueError(f"Invalid validation size for class {label}")

        indices = rng.permutation(len(class_images))
        val_indices = set(indices[:class_val_size].tolist())

        for index, image in enumerate(class_images):
            if index in val_indices:
                val_images.append(image)
            else:
                train_images.append(image)

    # Preserve original annotation order in both resulting datasets.
    original_order = {image["image_id"]: index for index, image in enumerate(images)}
    train_images.sort(key=lambda image: original_order[image["image_id"]])
    val_images.sort(key=lambda image: original_order[image["image_id"]])

    if len(val_images) != val_size:
        raise ValueError(
            f"Expected {val_size} validation images, got {len(val_images)}. "
            "The requested size may not divide evenly across classes."
        )

    return train_images, val_images


def save_annotations(
    path: Path,
    classes: list[str],
    images: list[dict[str, Any]],
) -> None:
    """Save annotations as deterministic JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            {"classes": classes, "images": images},
            file,
            ensure_ascii=False,
            indent=2,
        )
        file.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Split CIFAR-10 train data.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--val-size", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_path = args.data_dir / "annotations/train.json"
    output_dir = args.data_dir / "annotations/splits"

    annotations = load_annotations(input_path)

    train_images, val_images = stratified_split(
        annotations,
        val_size=args.val_size,
        seed=args.seed,
    )

    save_annotations(output_dir / "train.json", annotations["classes"], train_images)
    save_annotations(output_dir / "val.json", annotations["classes"], val_images)

    print(f"Train: {len(train_images)} images")
    print(f"Validation: {len(val_images)} images")
    print(f"Seed: {args.seed}")


if __name__ == "__main__":
    main()

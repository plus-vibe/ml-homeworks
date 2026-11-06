"""Convert the verified CIFAR-10 archive into PNG images and JSON annotations."""

import argparse
import hashlib
import json
import pickle
import tarfile
from pathlib import Path
from typing import Any

from PIL import Image

ARCHIVE_PATH = Path("data/raw/cifar-10-python.tar.gz")
OUTPUT_DIR = Path("data/processed/cifar10/v1")

ARCHIVE_ROOT = "cifar-10-batches-py"
TRAIN_BATCHES = [f"data_batch_{index}" for index in range(1, 6)]
TEST_BATCH = "test_batch"
METADATA_FILE = "batches.meta"


def load_pickle(archive: tarfile.TarFile, member_name: str) -> dict[bytes, Any]:
    """Load a CIFAR-10 pickle file from the archive."""
    member_path = f"{ARCHIVE_ROOT}/{member_name}"
    member = archive.extractfile(member_path)

    if member is None:
        raise FileNotFoundError(f"Missing archive member: {member_path}")

    with member:
        data = pickle.load(member, encoding="bytes")

    if not isinstance(data, dict):
        raise ValueError(f"Unexpected data format in {member_name}")

    return data


def calculate_sha256(path: Path) -> str:
    """Calculate SHA-256 checksum of a file."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def convert_batch(
    archive: tarfile.TarFile,
    batch_name: str,
    split_name: str,
    classes: list[str],
    start_index: int,
    output_dir: Path,
) -> list[dict[str, Any]]:
    """Convert one CIFAR-10 batch into PNG files and annotations."""
    batch = load_pickle(archive, batch_name)

    images = batch[b"data"]
    labels = batch[b"labels"]

    if images.shape != (len(labels), 3072):
        raise ValueError(f"Unexpected image array shape in {batch_name}")

    split_dir = output_dir / "images" / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    annotations: list[dict[str, Any]] = []

    for index, (image_data, label) in enumerate(zip(images, labels, strict=True)):
        image_id = f"{split_name}_{start_index + index:05d}"
        filename = f"{image_id}.png"
        image_path = split_dir / filename

        image_array = image_data.reshape(3, 32, 32).transpose(1, 2, 0)
        image = Image.fromarray(image_array)
        image.save(image_path, format="PNG")

        annotations.append(
            {
                "image_id": image_id,
                "file_name": filename,
                "label": int(label),
                "label_name": classes[int(label)],
                "sha256": calculate_sha256(image_path),
            }
        )

    return annotations


def convert_archive(archive_path: Path, output_dir: Path) -> None:
    """Convert the complete CIFAR-10 archive."""
    if not archive_path.is_file():
        raise FileNotFoundError(f"Archive not found: {archive_path}")

    with tarfile.open(archive_path, mode="r:gz") as archive:
        members = {member.name: member for member in archive.getmembers() if member.isfile()}

        expected_files = {
            f"{ARCHIVE_ROOT}/{name}" for name in [*TRAIN_BATCHES, TEST_BATCH, METADATA_FILE]
        }

        missing_files = expected_files - members.keys()
        if missing_files:
            raise ValueError(f"Archive is missing files: {sorted(missing_files)}")

        for name in expected_files:
            if members[name].size > 100 * 1024 * 1024:
                raise ValueError(f"Unexpectedly large archive member: {name}")

        metadata = load_pickle(archive, METADATA_FILE)
        classes = [name.decode("utf-8") for name in metadata[b"label_names"]]

        if len(classes) != 10:
            raise ValueError("Expected exactly 10 CIFAR-10 classes")

        train_annotations: list[dict[str, Any]] = []

        for batch_name in TRAIN_BATCHES:
            annotations = convert_batch(
                archive,
                batch_name,
                "train",
                classes,
                start_index=len(train_annotations),
                output_dir=output_dir,
            )
            train_annotations.extend(annotations)
            print(f"Converted {batch_name}: {len(annotations)} images")

        test_annotations = convert_batch(
            archive,
            TEST_BATCH,
            "test",
            classes,
            start_index=0,
            output_dir=output_dir,
        )
        print(f"Converted {TEST_BATCH}: {len(test_annotations)} images")

    annotations_dir = output_dir / "annotations"
    annotations_dir.mkdir(parents=True, exist_ok=True)

    for split_name, annotations in [
        ("train", train_annotations),
        ("test", test_annotations),
    ]:
        output_path = annotations_dir / f"{split_name}.json"

        with output_path.open("w", encoding="utf-8") as file:
            json.dump(
                {"classes": classes, "images": annotations},
                file,
                ensure_ascii=False,
                indent=2,
            )
            file.write("\n")

        print(f"Saved {output_path}: {len(annotations)} annotations")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert CIFAR-10 archive to PNG.")
    parser.add_argument("--archive", type=Path, default=ARCHIVE_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    convert_archive(args.archive, args.output_dir)


if __name__ == "__main__":
    main()

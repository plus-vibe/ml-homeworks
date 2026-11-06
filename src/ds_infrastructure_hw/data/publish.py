"""Publish a versioned CIFAR-10 dataset to S3-compatible storage."""

import argparse
import hashlib
import json
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from botocore.exceptions import ClientError

from ds_infrastructure_hw.s3_client import get_bucket_name, get_s3_client

DEFAULT_DATA_DIR = Path("data/processed/cifar10")
DEFAULT_ARCHIVE = Path("data/raw/cifar-10-python.tar.gz")
EXPECTED_ARCHIVE_MD5 = "c58f30108f718f92721af3b95e74349a"

ANNOTATION_FILES = {
    "train": Path("annotations/splits/train.json"),
    "val": Path("annotations/splits/val.json"),
    "test": Path("annotations/test.json"),
}

IMAGE_DIRECTORIES = {
    "train": Path("images/train"),
    "val": Path("images/train"),
    "test": Path("images/test"),
}


def calculate_sha256(path: Path) -> str:
    """Calculate SHA-256 checksum of a file."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def calculate_md5(path: Path) -> str:
    """Calculate MD5 checksum of the original archive."""
    digest = hashlib.md5(usedforsecurity=False)

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def get_git_info() -> dict[str, Any]:
    """Get Git commit and working tree status."""
    commit_result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )

    status_result = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=False,
    )

    return {
        "commit": commit_result.stdout.strip() if commit_result.returncode == 0 else None,
        "working_tree_dirty": (
            bool(status_result.stdout.strip()) if status_result.returncode == 0 else None
        ),
    }


def load_annotations(path: Path) -> dict[str, Any]:
    """Load and validate an annotation file."""
    with path.open(encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict) or "classes" not in data or "images" not in data:
        raise ValueError(f"Invalid annotation file: {path}")

    if not isinstance(data["images"], list):
        raise ValueError(f"Invalid images field: {path}")

    return data


def file_metadata(path: Path) -> dict[str, Any]:
    """Return file size and SHA-256 checksum."""
    return {
        "size_bytes": path.stat().st_size,
        "sha256": calculate_sha256(path),
    }


def acquire_lock(client: Any, bucket: str, key: str, version: str) -> None:
    """Acquire an exclusive publication lock using a conditional S3 write."""
    lock_data = json.dumps(
        {
            "version": version,
            "created_at": datetime.now(UTC).isoformat(),
        }
    ).encode("utf-8")

    try:
        client.put_object(
            Bucket=bucket,
            Key=key,
            Body=lock_data,
            ContentType="application/json",
            IfNoneMatch="*",
        )
    except ClientError as error:
        code = error.response.get("Error", {}).get("Code", "Unknown")

        if code in {"PreconditionFailed", "412", "ConditionalRequestConflict"}:
            raise RuntimeError(
                f"Another publication is in progress or a lock already exists: {key}"
            ) from error

        raise RuntimeError(
            "Could not acquire the S3 publication lock. "
            "Check whether the S3-compatible storage supports conditional writes."
        ) from error


def upload_images(
    client: Any,
    bucket: str,
    prefix: str,
    data_dir: Path,
    annotations: dict[str, list[dict[str, Any]]],
    workers: int,
) -> int:
    """Upload all images concurrently."""
    tasks: list[tuple[Path, str]] = []

    for split_name, images in annotations.items():
        image_dir = data_dir / IMAGE_DIRECTORIES[split_name]

        for image in images:
            filename = image["file_name"]

            if Path(filename).name != filename:
                raise ValueError(f"Invalid image filename: {filename}")

            image_path = image_dir / filename

            if not image_path.is_file():
                raise FileNotFoundError(f"Image not found: {image_path}")

            tasks.append(
                (
                    image_path,
                    f"{prefix}/images/{split_name}/{filename}",
                )
            )

    uploaded = 0
    batch_size = 500

    def upload_one(task: tuple[Path, str]) -> None:
        path, key = task
        client.upload_file(str(path), bucket, key)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        for start in range(0, len(tasks), batch_size):
            batch = tasks[start : start + batch_size]
            futures = [executor.submit(upload_one, task) for task in batch]

            for future in futures:
                future.result()
                uploaded += 1

            if uploaded % 5000 == 0 or uploaded == len(tasks):
                print(f"Uploaded {uploaded}/{len(tasks)} images")

    return uploaded


def publish(
    data_dir: Path,
    archive_path: Path,
    version: str,
    seed: int,
    val_size: int,
    workers: int,
) -> None:
    """Publish an immutable dataset version to S3."""
    if not re.fullmatch(r"v[1-9]\d*", version):
        raise ValueError("Version must have the format v1, v2, v3, ...")

    if workers < 1:
        raise ValueError("workers must be at least 1")

    version_dir = data_dir / version
    if not version_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {version_dir}")

    if not archive_path.is_file():
        raise FileNotFoundError(f"Original archive not found: {archive_path}")

    archive_md5 = calculate_md5(archive_path)
    if archive_md5 != EXPECTED_ARCHIVE_MD5:
        raise ValueError(
            f"Original archive MD5 mismatch: expected {EXPECTED_ARCHIVE_MD5}, got {archive_md5}"
        )

    archive_sha256 = calculate_sha256(archive_path)

    annotations: dict[str, list[dict[str, Any]]] = {}
    classes: list[str] | None = None

    for split_name, relative_path in ANNOTATION_FILES.items():
        annotation_path = version_dir / relative_path
        data = load_annotations(annotation_path)

        if classes is None:
            classes = data["classes"]
        elif data["classes"] != classes:
            raise ValueError("Class lists differ between annotation files")

        annotations[split_name] = data["images"]

    expected_counts = {
        "train": 50000 - val_size,
        "val": val_size,
        "test": 10000,
    }

    for split_name, expected_count in expected_counts.items():
        actual_count = len(annotations[split_name])
        if actual_count != expected_count:
            raise ValueError(f"{split_name}: expected {expected_count} images, got {actual_count}")

    all_ids = [
        image["image_id"] for split_images in annotations.values() for image in split_images
    ]

    if len(all_ids) != len(set(all_ids)):
        raise ValueError("Duplicate image IDs found between dataset splits")

    stats_path = version_dir / "stats.json"
    if not stats_path.is_file():
        raise FileNotFoundError(f"Statistics file not found: {stats_path}")

    stats = json.loads(stats_path.read_text(encoding="utf-8"))

    if stats.get("split") != "train" or stats.get("num_images") != len(annotations["train"]):
        raise ValueError("Statistics do not match the training split")

    artifact_paths = {
        f"annotations/{split_name}.json": version_dir / relative_path
        for split_name, relative_path in ANNOTATION_FILES.items()
    }
    artifact_paths["stats.json"] = stats_path

    artifacts = {name: file_metadata(path) for name, path in artifact_paths.items()}

    git_info = get_git_info()

    manifest = {
        "dataset": "CIFAR-10",
        "version": version,
        "created_at": datetime.now(UTC).isoformat(),
        "source": {
            "url": "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
            "filename": archive_path.name,
            "size_bytes": archive_path.stat().st_size,
            "md5": archive_md5,
            "sha256": archive_sha256,
        },
        "counts": {name: len(images) for name, images in annotations.items()},
        "classes": classes,
        "processing": {
            "validation_size": val_size,
            "seed": seed,
            "image_format": "PNG",
            "image_dimensions": [32, 32],
            "channel_order": "RGB",
            "statistics_split": "train",
        },
        "statistics": stats,
        "artifacts": artifacts,
        "git": git_info,
    }

    manifest_path = version_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    client = get_s3_client()
    bucket = get_bucket_name()
    prefix = f"processed/cifar10/{version}"
    lock_key = f"processed/cifar10/_locks/{version}.lock"

    existing = client.list_objects_v2(Bucket=bucket, Prefix=f"{prefix}/")
    if existing.get("Contents"):
        raise FileExistsError(f"Dataset version already contains objects: s3://{bucket}/{prefix}/")

    acquire_lock(client, bucket, lock_key, version)
    print(f"Acquired publication lock: {lock_key}")

    try:
        for name, path in artifact_paths.items():
            key = f"{prefix}/{name}"
            client.upload_file(str(path), bucket, key)
            print(f"Uploaded: {key}")

        uploaded_images = upload_images(
            client,
            bucket,
            prefix,
            version_dir,
            annotations,
            workers,
        )

        if uploaded_images != sum(expected_counts.values()):
            raise RuntimeError("Uploaded image count does not match expected count")

        # The manifest is the final commit marker for this dataset version.
        client.upload_file(
            str(manifest_path),
            bucket,
            f"{prefix}/manifest.json",
        )
        print(f"Uploaded manifest: s3://{bucket}/{prefix}/manifest.json")

    except Exception:
        print(
            "Publication failed. The lock has been retained to prevent "
            "accidental reuse of this version. Use a new version or inspect "
            "the incomplete prefix before recovery."
        )
        raise

    try:
        client.delete_object(Bucket=bucket, Key=lock_key)
    except ClientError as error:
        print(f"Warning: could not remove publication lock: {error}")

    print(f"Publication completed: s3://{bucket}/{prefix}/")
    print(f"Images uploaded: {uploaded_images}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish CIFAR-10 to S3.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--version", default="v1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-size", type=int, default=5000)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    publish(
        data_dir=args.data_dir,
        archive_path=args.archive,
        version=args.version,
        seed=args.seed,
        val_size=args.val_size,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()

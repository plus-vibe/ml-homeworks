"""CIFAR-10 dataset loader backed by versioned S3 storage."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypedDict, cast

import numpy as np
import torch

if TYPE_CHECKING:
    from mypy_boto3_s3.client import S3Client
from PIL import Image
from torch.utils.data import Dataset

from ds_infrastructure_hw.s3_client import get_bucket_name, get_s3_client

Split = Literal["train", "val", "test"]


class ImageRecord(TypedDict):
    image_id: str
    file_name: str
    label: int
    label_name: str
    sha256: str


class AnnotationFile(TypedDict):
    classes: list[str]
    images: list[ImageRecord]


class StatisticsFile(TypedDict):
    mean: list[float]
    std: list[float]


def _validate_s3_path(s3_path: str) -> str:
    """Validate the relative prefix of a versioned dataset."""
    prefix = s3_path.strip("/")

    if not prefix or not re.fullmatch(r"[A-Za-z0-9._/-]+", prefix):
        raise ValueError(f"Invalid S3 dataset path: {s3_path}")

    if any(part in ("", ".", "..") for part in prefix.split("/")):
        raise ValueError(f"Unsafe S3 dataset path: {s3_path}")

    return prefix


def _download_if_missing(
    client: S3Client,
    bucket: str,
    key: str,
    destination: Path,
    expected_sha256: str | None = None,
) -> None:
    """Download an S3 object and optionally verify its SHA-256."""
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists():
        if expected_sha256 is None:
            return

        actual = hashlib.sha256(destination.read_bytes()).hexdigest()
        if actual == expected_sha256:
            return

        destination.unlink()

    temporary = destination.with_name(destination.name + ".download")

    try:
        client.download_file(bucket, key, str(temporary))

        if expected_sha256 is not None:
            actual = hashlib.sha256(temporary.read_bytes()).hexdigest()

            if actual != expected_sha256:
                raise ValueError(f"SHA-256 mismatch for s3://{bucket}/{key}")

        os.replace(temporary, destination)

    finally:
        temporary.unlink(missing_ok=True)


class CIFAR10S3Dataset(Dataset[tuple[torch.Tensor, int]]):
    """CIFAR-10 split downloaded and cached from S3."""

    def __init__(
        self,
        s3_path: str,
        split: Split,
        max_samples: int | None = None,
        cache_root: Path = Path("data/cache/cifar10"),
    ) -> None:
        super().__init__()

        if max_samples is not None and max_samples <= 0:
            raise ValueError("max_samples must be positive or None")

        prefix = _validate_s3_path(s3_path)
        client = get_s3_client()
        bucket = get_bucket_name()

        self.split = split
        self.local_root = cache_root / bucket / prefix

        annotation_path = self.local_root / "annotations" / f"{split}.json"
        statistics_path = self.local_root / "stats.json"

        _download_if_missing(
            client,
            bucket,
            f"{prefix}/annotations/{split}.json",
            annotation_path,
        )

        _download_if_missing(
            client,
            bucket,
            f"{prefix}/stats.json",
            statistics_path,
        )

        annotations = cast(
            AnnotationFile,
            json.loads(annotation_path.read_text(encoding="utf-8")),
        )

        statistics = cast(
            StatisticsFile,
            json.loads(statistics_path.read_text(encoding="utf-8")),
        )

        records = annotations["images"]

        if max_samples is not None:
            records = records[:max_samples]

        if not records:
            raise ValueError(f"Empty dataset split: {split}")

        self.records = records

        mean = statistics["mean"]
        std = statistics["std"]

        if len(mean) != 3 or len(std) != 3:
            raise ValueError("Expected three RGB mean/std values")

        if any(value <= 0 for value in std):
            raise ValueError("RGB standard deviations must be positive")

        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)

        print(f"Preparing {split}: {len(self.records)} images from s3://{bucket}/{prefix}")

        for index, record in enumerate(self.records, start=1):
            file_name = record["file_name"]

            if Path(file_name).name != file_name:
                raise ValueError(f"Unsafe image filename: {file_name}")

            destination = self.local_root / "images" / split / file_name

            _download_if_missing(
                client,
                bucket,
                f"{prefix}/images/{split}/{file_name}",
                destination,
                expected_sha256=record["sha256"],
            )

            if index % 500 == 0 or index == len(self.records):
                print(f"{split}: prepared {index}/{len(self.records)}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        record = self.records[index]

        path = self.local_root / "images" / self.split / record["file_name"]

        with Image.open(path) as image:
            rgb = image.convert("RGB")

            if rgb.size != (32, 32):
                raise ValueError(f"Unexpected image size: {path}")

            pixels = np.asarray(rgb, dtype=np.float32)

        channels_first = np.transpose(pixels, (2, 0, 1)).copy()
        tensor = torch.from_numpy(channels_first)

        tensor = tensor / 255.0
        tensor = (tensor - self.mean) / self.std

        return tensor, record["label"]

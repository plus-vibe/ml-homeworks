"""Download and verify the original CIFAR-10 archive from S3."""

import argparse
import hashlib
from pathlib import Path

from botocore.exceptions import ClientError

from ds_infrastructure_hw.s3_client import get_bucket_name, get_s3_client

RAW_KEY = "raw/cifar-10-python.tar.gz"
EXPECTED_MD5 = "c58f30108f718f92721af3b95e74349a"


def calculate_md5(path: Path) -> str:
    """Calculate the MD5 checksum of a file."""
    digest = hashlib.md5(usedforsecurity=False)

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def download_raw(destination: Path) -> None:
    """Download CIFAR-10 from S3 and verify its integrity."""
    client = get_s3_client()
    bucket = get_bucket_name()

    destination.parent.mkdir(parents=True, exist_ok=True)

    try:
        client.download_file(bucket, RAW_KEY, str(destination))
    except ClientError:
        destination.unlink(missing_ok=True)
        raise

    actual_md5 = calculate_md5(destination)

    if actual_md5 != EXPECTED_MD5:
        destination.unlink(missing_ok=True)
        raise ValueError(
            f"Invalid CIFAR-10 archive MD5: expected {EXPECTED_MD5}, got {actual_md5}"
        )

    print(f"Downloaded: {destination}")
    print(f"Verified MD5: {actual_md5}")


def main() -> None:
    """Run the download command."""
    parser = argparse.ArgumentParser(description="Download CIFAR-10 archive from S3")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/cifar-10-python.tar.gz"),
    )
    args = parser.parse_args()

    download_raw(args.output)


if __name__ == "__main__":
    main()

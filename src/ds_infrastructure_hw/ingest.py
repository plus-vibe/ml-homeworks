import argparse
import hashlib
import tempfile
from pathlib import Path

import requests
from botocore.exceptions import ClientError

from ds_infrastructure_hw.s3_client import get_bucket_name, get_s3_client

CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
RAW_KEY = "raw/cifar-10-python.tar.gz"


def object_exists(bucket: str, key: str) -> bool:
    """Check whether an object already exists in S3."""
    client = get_s3_client()

    try:
        client.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as exc:
        status = exc.response["ResponseMetadata"]["HTTPStatusCode"]
        if status == 404:
            return False
        raise


def download_archive(destination: Path) -> str:
    """Download CIFAR-10 and calculate its SHA-256 checksum."""
    digest = hashlib.sha256()

    with requests.get(CIFAR10_URL, stream=True, timeout=120) as response:
        response.raise_for_status()

        with destination.open("wb") as output:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    output.write(chunk)
                    digest.update(chunk)

    return digest.hexdigest()


def ingest() -> None:
    """Upload the original CIFAR-10 archive without overwriting it."""
    bucket = get_bucket_name()
    client = get_s3_client()

    if object_exists(bucket, RAW_KEY):
        print(f"Already exists: s3://{bucket}/{RAW_KEY}")
        print("Skipping download and upload.")
        return

    with tempfile.TemporaryDirectory() as directory:
        archive = Path(directory) / "cifar-10-python.tar.gz"

        print("Downloading official CIFAR-10 archive...")
        checksum = download_archive(archive)

        print(f"SHA-256: {checksum}")
        print(f"Size: {archive.stat().st_size} bytes")

        if object_exists(bucket, RAW_KEY):
            print("Object appeared during download; skipping upload.")
            return

        with archive.open("rb") as source:
            client.put_object(
                Bucket=bucket,
                Key=RAW_KEY,
                Body=source,
                IfNoneMatch="*",
                Metadata={"sha256": checksum},
            )

        print(f"Uploaded: s3://{bucket}/{RAW_KEY}")


def main() -> None:
    """Run the ingestion command."""
    parser = argparse.ArgumentParser(description="Upload CIFAR-10 to S3")
    parser.parse_args()
    ingest()


if __name__ == "__main__":
    main()

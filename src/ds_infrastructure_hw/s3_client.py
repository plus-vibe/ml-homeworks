import os

import boto3
from botocore.config import Config
from mypy_boto3_s3.client import S3Client


def get_s3_client() -> S3Client:
    endpoint = os.environ["S3_ENDPOINT_URL"]
    access_key = os.environ["RUSTFS_ACCESS_KEY"]
    secret_key = os.environ["RUSTFS_SECRET_KEY"]

    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name="us-east-1",
        config=Config(
            s3={"addressing_style": "path"},
            request_checksum_calculation="when_required",
            response_checksum_validation="when_required",
        ),
    )


def get_bucket_name() -> str:
    return os.environ["S3_BUCKET"]

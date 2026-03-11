"""S3-compatible storage helpers for MinIO."""

import io
import json
import os
import pickle

import boto3
from botocore.client import Config


def get_s3_client():
    """Create an S3 client configured for the local MinIO instance."""
    return boto3.client(
        "s3",
        endpoint_url=os.getenv("S3_ENDPOINT_URL", "http://localhost:9000"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID", "minioadmin"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin"),
        config=Config(signature_version="s3v4"),
    )


def ensure_bucket(client, bucket: str) -> None:
    """Create the bucket if it doesn't already exist."""
    try:
        client.head_bucket(Bucket=bucket)
    except client.exceptions.ClientError:
        client.create_bucket(Bucket=bucket)
        print(f"Created bucket: {bucket}")


def upload_json(client, bucket: str, key: str, data: dict) -> None:
    """Serialize a dict as JSON and upload to S3."""
    body = json.dumps(data).encode("utf-8")
    client.put_object(Bucket=bucket, Key=key, Body=body, ContentType="application/json")
    print(f"Uploaded s3://{bucket}/{key} ({len(body)} bytes)")


def download_json(client, bucket: str, key: str) -> dict:
    """Download a JSON object from S3 and return as dict."""
    response = client.get_object(Bucket=bucket, Key=key)
    return json.loads(response["Body"].read().decode("utf-8"))


def upload_pickle(client, bucket: str, key: str, obj) -> None:
    """Pickle an object and upload to S3."""
    buf = io.BytesIO()
    pickle.dump(obj, buf)
    buf.seek(0)
    client.put_object(Bucket=bucket, Key=key, Body=buf.getvalue(), ContentType="application/octet-stream")
    print(f"Uploaded s3://{bucket}/{key} ({buf.getbuffer().nbytes} bytes)")


def upload_text(client, bucket: str, key: str, text: str) -> None:
    """Upload a text string to S3."""
    body = text.encode("utf-8")
    client.put_object(Bucket=bucket, Key=key, Body=body, ContentType="text/plain")
    print(f"Uploaded s3://{bucket}/{key} ({len(body)} bytes)")

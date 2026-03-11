"""
Download the 20 Newsgroups dataset and upload the raw data to MinIO.
Uses scikit-learn's fetch_20newsgroups which provides the same dataset
as https://www.kaggle.com/datasets/au1206/20-newsgroup-original
"""

import argparse

from sklearn.datasets import fetch_20newsgroups

from src.storage import ensure_bucket, get_s3_client, upload_json

DATA_BUCKET = "data"


def download(bucket: str = DATA_BUCKET) -> None:
    """Download train and test splits and upload to MinIO."""
    client = get_s3_client()
    ensure_bucket(client, bucket)

    for subset in ("train", "test"):
        data = fetch_20newsgroups(subset=subset, remove=())
        records = {
            "data": data.data,
            "target": data.target.tolist(),
            "target_names": data.target_names,
        }
        key = f"raw/raw_{subset}.json"
        upload_json(client, bucket, key, records)
        print(f"Saved {len(data.data)} documents to s3://{bucket}/{key}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download 20 Newsgroups dataset")
    parser.add_argument(
        "--bucket",
        default=DATA_BUCKET,
        help=f"S3 bucket for raw data (default: {DATA_BUCKET})",
    )
    args = parser.parse_args()
    download(args.bucket)

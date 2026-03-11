"""
Preprocess the 20 Newsgroups dataset.

Steps:
  1. Load raw JSON data
  2. Strip email headers, footers, and quoting artifacts
  3. Lowercase and remove non-alphabetic characters
  4. Remove short documents (< 10 chars after cleaning)
  5. Save cleaned data as JSON
"""

import argparse
import re

import mlflow

from src.storage import (
    download_json,
    ensure_bucket,
    get_s3_client,
    upload_json,
)

DATA_BUCKET = "data"


def clean_text(text: str) -> str:
    """Clean a single document."""
    # Remove header lines (From:, Subject:, etc.) up to the first blank line
    parts = text.split("\n\n", 1)
    body = parts[1] if len(parts) > 1 else parts[0]

    # Remove quoted lines (lines starting with > or |)
    lines = [line for line in body.split("\n") if not line.strip().startswith(">")]
    body = "\n".join(lines)

    # Remove email addresses
    body = re.sub(r"\S+@\S+", " ", body)

    # Remove non-alphabetic characters (keep spaces)
    body = re.sub(r"[^a-zA-Z\s]", " ", body)

    # Collapse whitespace and lowercase
    body = re.sub(r"\s+", " ", body).strip().lower()

    return body


def preprocess(bucket: str = DATA_BUCKET) -> None:
    """Load raw data from MinIO, clean it, and upload back."""
    client = get_s3_client()
    ensure_bucket(client, bucket)

    mlflow.set_experiment("20newsgroups-tfidf")

    with mlflow.start_run(run_name="preprocessing"):
        mlflow.log_param("min_doc_length", 10)

        for subset in ("train", "test"):
            raw = download_json(client, bucket, f"raw/raw_{subset}.json")

            cleaned_data = []
            cleaned_targets = []

            for text, target in zip(raw["data"], raw["target"]):
                cleaned = clean_text(text)
                if len(cleaned) >= 10:
                    cleaned_data.append(cleaned)
                    cleaned_targets.append(target)

            records = {
                "data": cleaned_data,
                "target": cleaned_targets,
                "target_names": raw["target_names"],
            }

            key = f"clean/clean_{subset}.json"
            upload_json(client, bucket, key, records)

            dropped = len(raw["data"]) - len(cleaned_data)
            mlflow.log_metric(f"{subset}_docs_kept", len(cleaned_data))
            mlflow.log_metric(f"{subset}_docs_dropped", dropped)
            print(
                f"[{subset}] {len(cleaned_data)} docs kept, {dropped} dropped → s3://{bucket}/{key}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess 20 Newsgroups data")
    parser.add_argument(
        "--bucket",
        default=DATA_BUCKET,
        help=f"S3 bucket for data (default: {DATA_BUCKET})",
    )
    args = parser.parse_args()
    preprocess(args.bucket)

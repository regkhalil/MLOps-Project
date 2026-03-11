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
import json
import re
from pathlib import Path

import mlflow


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


def preprocess(data_dir: Path, output_dir: Path) -> None:
    """Load raw data, clean it, and save."""
    output_dir.mkdir(parents=True, exist_ok=True)

    mlflow.set_experiment("20newsgroups-tfidf")

    with mlflow.start_run(run_name="preprocessing"):
        mlflow.log_param("min_doc_length", 10)

        for subset in ("train", "test"):
            raw_path = data_dir / f"raw_{subset}.json"
            with open(raw_path, "r", encoding="utf-8") as f:
                raw = json.load(f)

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

            out_path = output_dir / f"clean_{subset}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(records, f)

            dropped = len(raw["data"]) - len(cleaned_data)
            mlflow.log_metric(f"{subset}_docs_kept", len(cleaned_data))
            mlflow.log_metric(f"{subset}_docs_dropped", dropped)
            print(
                f"[{subset}] {len(cleaned_data)} docs kept, {dropped} dropped → {out_path}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess 20 Newsgroups data")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory with raw data (default: data/)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Directory for cleaned data (default: data/)",
    )
    args = parser.parse_args()
    preprocess(args.data_dir, args.output_dir)

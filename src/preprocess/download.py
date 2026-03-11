"""
Download the 20 Newsgroups dataset and save the raw data to disk.
Uses scikit-learn's fetch_20newsgroups which provides the same dataset
as https://www.kaggle.com/datasets/au1206/20-newsgroup-original
"""

import argparse
import json
from pathlib import Path

from sklearn.datasets import fetch_20newsgroups


def download(data_dir: Path) -> None:
    """Download train and test splits and persist as JSON."""
    data_dir.mkdir(parents=True, exist_ok=True)

    for subset in ("train", "test"):
        data = fetch_20newsgroups(subset=subset, remove=())
        records = {
            "data": data.data,
            "target": data.target.tolist(),
            "target_names": data.target_names,
        }
        out_path = data_dir / f"raw_{subset}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(records, f)
        print(f"Saved {len(data.data)} documents to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download 20 Newsgroups dataset")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory to save raw data (default: data/)",
    )
    args = parser.parse_args()
    download(args.data_dir)

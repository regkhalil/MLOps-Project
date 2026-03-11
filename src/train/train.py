"""
Train a TF-IDF + classifier pipeline on the preprocessed 20 Newsgroups data.

Pipeline:
  1. Load cleaned train/test JSON
  2. Fit TF-IDF vectorizer on training data
  3. Train a classifier (default: LinearSVC via SGDClassifier for scalability)
  4. Evaluate on the test set
  5. Save the trained pipeline to disk
"""

import argparse
import json
import pickle
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline


def load_split(path: Path) -> tuple[list[str], list[int], list[str]]:
    """Load a cleaned JSON split and return (texts, targets, target_names)."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["data"], data["target"], data["target_names"]


def train(data_dir: Path, model_dir: Path) -> None:
    """Train and evaluate the TF-IDF classification pipeline."""
    train_texts, train_targets, target_names = load_split(
        data_dir / "clean_train.json"
    )
    test_texts, test_targets, _ = load_split(data_dir / "clean_test.json")

    print(f"Training samples: {len(train_texts)}")
    print(f"Test samples:     {len(test_texts)}")
    print(f"Classes:          {len(target_names)}")

    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=50_000,
                    sublinear_tf=True,
                    ngram_range=(1, 2),
                ),
            ),
            (
                "clf",
                SGDClassifier(
                    loss="hinge",
                    alpha=1e-4,
                    max_iter=100,
                    random_state=42,
                ),
            ),
        ]
    )

    print("\nTraining...")
    pipeline.fit(train_texts, train_targets)

    print("\nEvaluation on test set:")
    predictions = pipeline.predict(test_texts)
    print(classification_report(test_targets, predictions, target_names=target_names))

    # Save model
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "tfidf_classifier.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train TF-IDF classifier on 20 Newsgroups"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory with cleaned data (default: data/)",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("models"),
        help="Directory to save trained model (default: models/)",
    )
    args = parser.parse_args()
    train(args.data_dir, args.model_dir)

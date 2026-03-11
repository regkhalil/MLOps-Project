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

import mlflow
import mlflow.sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

from src.storage import (
    download_json,
    ensure_bucket,
    get_s3_client,
    upload_pickle,
    upload_text,
)

DATA_BUCKET = "data"
MODEL_BUCKET = "models"


def load_split(client, bucket: str, key: str) -> tuple[list[str], list[int], list[str]]:
    """Load a cleaned JSON split from S3 and return (texts, targets, target_names)."""
    data = download_json(client, bucket, key)
    return data["data"], data["target"], data["target_names"]


def train(data_bucket: str = DATA_BUCKET, model_bucket: str = MODEL_BUCKET) -> None:
    """Train and evaluate the TF-IDF classification pipeline."""
    client = get_s3_client()
    ensure_bucket(client, model_bucket)

    train_texts, train_targets, target_names = load_split(
        client, data_bucket, "clean/clean_train.json"
    )
    test_texts, test_targets, _ = load_split(
        client, data_bucket, "clean/clean_test.json"
    )

    print(f"Training samples: {len(train_texts)}")
    print(f"Test samples:     {len(test_texts)}")
    print(f"Classes:          {len(target_names)}")

    ensure_bucket(client, "mlflow-artifacts")

    mlflow.set_experiment("20newsgroups-tfidf")

    with mlflow.start_run(run_name="training"):
        # Log dataset info
        mlflow.log_param("train_samples", len(train_texts))
        mlflow.log_param("test_samples", len(test_texts))
        mlflow.log_param("num_classes", len(target_names))

        # Log TF-IDF params
        mlflow.log_param("max_features", 50_000)
        mlflow.log_param("ngram_range", "(1, 2)")
        mlflow.log_param("sublinear_tf", True)

        # Log classifier params
        mlflow.log_param("clf_loss", "hinge")
        mlflow.log_param("clf_alpha", 1e-4)
        mlflow.log_param("clf_max_iter", 100)

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
        report = classification_report(
            test_targets, predictions, target_names=target_names, output_dict=True
        )
        print(
            classification_report(
                test_targets, predictions, target_names=target_names
            )
        )

        # Log aggregate metrics
        mlflow.log_metric("accuracy", report["accuracy"])
        mlflow.log_metric("macro_f1", report["macro avg"]["f1-score"])
        mlflow.log_metric("weighted_f1", report["weighted avg"]["f1-score"])
        mlflow.log_metric("macro_precision", report["macro avg"]["precision"])
        mlflow.log_metric("macro_recall", report["macro avg"]["recall"])

        # Log per-class F1
        for name in target_names:
            safe_name = name.replace(".", "_")
            mlflow.log_metric(f"f1_{safe_name}", report[name]["f1-score"])

        # Log the sklearn pipeline as an MLflow model
        mlflow.sklearn.log_model(pipeline, "model")

        # Upload model to MinIO
        upload_pickle(client, model_bucket, "classifier.pkl", pipeline)

        # Upload classification report to MinIO and log as MLflow artifact
        report_text = classification_report(
            test_targets, predictions, target_names=target_names
        )
        upload_text(client, model_bucket, "classification_report.txt", report_text)

        # Also log report as MLflow artifact (write temp file)
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            report_path = Path(tmp) / "classification_report.txt"
            report_path.write_text(report_text, encoding="utf-8")
            mlflow.log_artifact(str(report_path))

        print(f"\nMLflow run ID: {mlflow.active_run().info.run_id}")
        print(f"Model saved to s3://{model_bucket}/classifier.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train TF-IDF classifier on 20 Newsgroups"
    )
    parser.add_argument(
        "--data-bucket",
        default=DATA_BUCKET,
        help=f"S3 bucket for cleaned data (default: {DATA_BUCKET})",
    )
    parser.add_argument(
        "--model-bucket",
        default=MODEL_BUCKET,
        help=f"S3 bucket for trained model (default: {MODEL_BUCKET})",
    )
    args = parser.parse_args()
    train(args.data_bucket, args.model_bucket)

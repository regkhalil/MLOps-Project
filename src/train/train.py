"""
Train multiple models on 20 Newsgroups (text classification - 20 categories).
All experiments tracked with MLflow for model comparison and management.

Models: SGDClassifier, MultinomialNB, LogisticRegression
"""

import time

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, f1_score
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


# ============== CONFIGURATIONS ==============
# Multiple hyperparameter configurations for comparison in MLflow
MODELS = [
    # SGDClassifier with different alpha values
    {"name": "SGD_alpha1e-4", "class": SGDClassifier, 
     "params": {"loss": "hinge", "alpha": 1e-4, "max_iter": 100, "random_state": 42}},
    {"name": "SGD_alpha1e-3", "class": SGDClassifier, 
     "params": {"loss": "hinge", "alpha": 1e-3, "max_iter": 100, "random_state": 42}},
    
    # MultinomialNB with different alpha (smoothing)
    {"name": "NaiveBayes_alpha0.1", "class": MultinomialNB, 
     "params": {"alpha": 0.1}},
    {"name": "NaiveBayes_alpha1.0", "class": MultinomialNB, 
     "params": {"alpha": 1.0}},
    
    # LogisticRegression with different C (regularization)
    {"name": "LogReg_C1", "class": LogisticRegression, 
     "params": {"C": 1.0, "max_iter": 200, "random_state": 42}},
    {"name": "LogReg_C10", "class": LogisticRegression, 
     "params": {"C": 10.0, "max_iter": 200, "random_state": 42}},
]

TFIDF_PARAMS = {"max_features": 30000, "ngram_range": (1, 2), "sublinear_tf": True}


def load_split(client, bucket: str, key: str) -> tuple[list[str], list[int], list[str]]:
    """Load a cleaned JSON split from S3 and return (texts, targets, target_names)."""
    data = download_json(client, bucket, key)
    return data["data"], data["target"], data["target_names"]


def train_model(model_cfg, tfidf_params, train_texts, train_targets, 
                test_texts, test_targets, target_names, s3_client, model_bucket):
    """Train a single model and log everything to MLflow."""
    
    run_name = model_cfg["name"]
    print(f"\n{'='*50}")
    print(f"Training: {run_name}")
    print(f"{'='*50}")
    
    with mlflow.start_run(run_name=run_name):
        # Log parameters
        mlflow.log_param("model", run_name)
        mlflow.log_param("train_samples", len(train_texts))
        mlflow.log_param("test_samples", len(test_texts))
        mlflow.log_param("num_classes", len(target_names))
        
        for k, v in tfidf_params.items():
            mlflow.log_param(f"tfidf_{k}", str(v))
        for k, v in model_cfg["params"].items():
            mlflow.log_param(f"clf_{k}", v)
        
        # Build pipeline
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(**tfidf_params)),
            ("clf", model_cfg["class"](**model_cfg["params"]))
        ])
        
        # Train
        start = time.time()
        pipeline.fit(train_texts, train_targets)
        train_time = time.time() - start
        mlflow.log_metric("train_time_sec", round(train_time, 2))
        
        # Evaluate
        preds = pipeline.predict(test_texts)
        acc = accuracy_score(test_targets, preds)
        macro_f1 = f1_score(test_targets, preds, average="macro")
        weighted_f1 = f1_score(test_targets, preds, average="weighted")
        
        # Log metrics
        mlflow.log_metric("accuracy", round(acc, 4))
        mlflow.log_metric("macro_f1", round(macro_f1, 4))
        mlflow.log_metric("weighted_f1", round(weighted_f1, 4))
        
        # Log per-class F1
        report = classification_report(test_targets, preds, target_names=target_names, output_dict=True)
        for name in target_names:
            safe = name.replace(".", "_").replace("-", "_")
            mlflow.log_metric(f"f1_{safe}", round(report[name]["f1-score"], 4))
        
        # Log and REGISTER model in Model Registry
        model_info = mlflow.sklearn.log_model(pipeline, "model", registered_model_name="20newsgroups-classifier")
        
        # Save report to S3 and as MLflow artifact
        report_text = classification_report(test_targets, preds, target_names=target_names)
        upload_text(s3_client, model_bucket, f"reports/report_{run_name}.txt", report_text)
        
        # Also log as MLflow artifact (write temp file)
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            report_path = Path(tmp) / f"report_{run_name}.txt"
            report_path.write_text(report_text, encoding="utf-8")
            mlflow.log_artifact(str(report_path))
        
        run_id = mlflow.active_run().info.run_id
        print(f"  Accuracy: {acc:.4f} | Macro F1: {macro_f1:.4f} | Time: {train_time:.1f}s")
        return {"name": run_name, "accuracy": acc, "macro_f1": macro_f1, "run_id": run_id}


def train(data_bucket: str = DATA_BUCKET, model_bucket: str = MODEL_BUCKET) -> None:
    """Train all models and track with MLflow."""
    client = get_s3_client()
    ensure_bucket(client, model_bucket)
    
    # Load data from S3
    train_texts, train_targets, target_names = load_split(client, data_bucket, "clean/clean_train.json")
    test_texts, test_targets, _ = load_split(client, data_bucket, "clean/clean_test.json")
    
    print(f"Training samples: {len(train_texts)}")
    print(f"Test samples:     {len(test_texts)}")
    print(f"Classes:          {len(target_names)}")
    print(f"\nModels to train: {len(MODELS)}")
    
    ensure_bucket(client, "mlflow-artifacts")
    mlflow.set_experiment("20newsgroups-classification")
    
    results = []
    for i, model_cfg in enumerate(MODELS, 1):
        print(f"\n[{i}/{len(MODELS)}]", end="")
        result = train_model(
            model_cfg, TFIDF_PARAMS,
            train_texts, train_targets,
            test_texts, test_targets,
            target_names, client, model_bucket
        )
        results.append(result)
    
    # Summary
    print("\n" + "="*50)
    print("RESULTS SUMMARY")
    print("="*50)
    sorted_results = sorted(results, key=lambda x: x["accuracy"], reverse=True)
    for r in sorted_results:
        print(f"  {r['name']}: Accuracy={r['accuracy']:.4f}, F1={r['macro_f1']:.4f}")
    
    # Save best model to S3 as well
    best = sorted_results[0]
    best_run = mlflow.get_run(best["run_id"])
    best_model = mlflow.sklearn.load_model(f"runs:/{best['run_id']}/model")
    upload_pickle(client, model_bucket, "classifier.pkl", best_model)
    
    # Set best model as "champion" alias in Model Registry
    mlflow_client = MlflowClient()
    model_name = "20newsgroups-classifier"
    versions = mlflow_client.search_model_versions(f"name='{model_name}'")
    
    for v in versions:
        if v.run_id == best["run_id"]:
            mlflow_client.set_registered_model_alias(model_name, "champion", v.version)
            print(f"\n*** BEST MODEL: {best['name']} (version {v.version}) -> alias 'champion' ***")
            break
    
    print("\nAll models registered in MLflow. API can load best with alias 'champion'.")
    print(f"Best model saved to s3://{model_bucket}/classifier.pkl")


if __name__ == "__main__":
    train()

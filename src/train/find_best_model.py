"""
Load the best model from MLflow Model Registry.
The API uses this to get the champion model.
"""

import mlflow
from mlflow.tracking import MlflowClient

MODEL_NAME = "20newsgroups-classifier"


def get_best_model():
    """
    Load the best model (champion alias) from MLflow Model Registry.
    This is what the API should call to get the model for inference.
    
    Returns:
        Loaded sklearn pipeline (TF-IDF + Classifier)
    """
    model_uri = f"models:/{MODEL_NAME}@champion"
    
    try:
        model = mlflow.sklearn.load_model(model_uri)
        print(f"Loaded champion model from: {model_uri}")
        return model
    except Exception as e:
        print(f"Could not load champion model: {e}")
        print("Falling back to latest version...")
        
        # Fallback: load latest version
        model_uri = f"models:/{MODEL_NAME}/latest"
        model = mlflow.sklearn.load_model(model_uri)
        return model


def get_model_info():
    """Get information about the champion model."""
    client = MlflowClient()
    
    try:
        version = client.get_model_version_by_alias(MODEL_NAME, "champion")
        run = client.get_run(version.run_id)
        
        return {
            "model_name": MODEL_NAME,
            "version": version.version,
            "run_id": version.run_id,
            "accuracy": run.data.metrics.get("accuracy"),
            "macro_f1": run.data.metrics.get("macro_f1"),
            "model_type": run.data.params.get("model"),
        }
    except Exception as e:
        return {"error": str(e)}


def list_all_models():
    """List all registered model versions."""
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    
    models = []
    for v in versions:
        run = client.get_run(v.run_id)
        models.append({
            "version": v.version,
            "model_type": run.data.params.get("model"),
            "accuracy": run.data.metrics.get("accuracy"),
            "aliases": v.aliases,
        })
    
    return sorted(models, key=lambda x: x["macro_f1"] or 0, reverse=True)


if __name__ == "__main__":
    print("="*60)
    print("ALL REGISTERED MODELS")
    print("="*60)
    
    for m in list_all_models():
        alias_str = f" [{', '.join(m['aliases'])}]" if m['aliases'] else ""
        print(f"  v{m['version']}: {m['model_type']:25} Acc={m['accuracy']:.4f}{alias_str}")
    
    print("\n" + "="*60)
    print("CHAMPION MODEL INFO")
    print("="*60)
    info = get_model_info()
    for k, v in info.items():
        print(f"  {k}: {v}")
    
    print("\n" + "="*60)
    print("TESTING MODEL LOAD")
    print("="*60)
    model = get_best_model()
    
    # Quick test
    test_texts = ["NASA launched a new satellite into space orbit"]
    pred = model.predict(test_texts)
    print(f"\nTest: '{test_texts[0]}'")
    print(f"Predicted class: {pred[0]}")

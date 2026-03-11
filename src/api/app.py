"""
FastAPI service for 20 Newsgroups text classification.

Loads the BEST model (champion alias) from MLflow Model Registry.
"""

import os

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME = "20newsgroups-classifier"  # Registered model name in Model Registry

TARGET_NAMES = [
    "alt.atheism",
    "comp.graphics",
    "comp.os.ms-windows.misc",
    "comp.sys.ibm.pc.hardware",
    "comp.sys.mac.hardware",
    "comp.windows.x",
    "misc.forsale",
    "rec.autos",
    "rec.motorcycles",
    "rec.sport.baseball",
    "rec.sport.hockey",
    "sci.crypt",
    "sci.electronics",
    "sci.med",
    "sci.space",
    "soc.religion.christian",
    "talk.politics.guns",
    "talk.politics.mideast",
    "talk.politics.misc",
    "talk.religion.misc",
]

DISPLAY_NAMES = {
    "alt.atheism": "Atheism & Secularism",
    "comp.graphics": "Computer Graphics",
    "comp.os.ms-windows.misc": "Windows OS",
    "comp.sys.ibm.pc.hardware": "PC Hardware",
    "comp.sys.mac.hardware": "Mac Hardware",
    "comp.windows.x": "X Window System",
    "misc.forsale": "For Sale",
    "rec.autos": "Automobiles",
    "rec.motorcycles": "Motorcycles",
    "rec.sport.baseball": "Baseball",
    "rec.sport.hockey": "Hockey",
    "sci.crypt": "Cryptography",
    "sci.electronics": "Electronics",
    "sci.med": "Medicine & Health",
    "sci.space": "Space & Astronomy",
    "soc.religion.christian": "Christianity",
    "talk.politics.guns": "Gun Politics",
    "talk.politics.mideast": "Middle East Politics",
    "talk.politics.misc": "General Politics",
    "talk.religion.misc": "Religion & Beliefs",
}

app = FastAPI(title="20 Newsgroups Classifier", version="1.0.0")

model = None


class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    label: str
    display_name: str
    class_id: int


def load_model():
    """Load the BEST model using the 'champion' alias from MLflow Model Registry."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    # Load model with "champion" alias (set during training to best accuracy)
    model_uri = f"models:/{MODEL_NAME}@champion"
    
    try:
        loaded = mlflow.sklearn.load_model(model_uri)
        client = MlflowClient()
        alias_info = client.get_model_version_by_alias(MODEL_NAME, "champion")
        print(f"Loaded model: {MODEL_NAME} version {alias_info.version} (champion)")
        return loaded
    except Exception as e:
        print(f"Warning: Could not load champion alias: {e}")
        print("Falling back to latest version...")
        try:
            model_uri = f"models:/{MODEL_NAME}/latest"
            return mlflow.sklearn.load_model(model_uri)
        except Exception as e2:
            print(f"Warning: No model available yet: {e2}")
            return None


@app.on_event("startup")
def startup():
    global model
    model = load_model()


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.get("/model-info")
def model_info():
    """Get information about the currently loaded model."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()
    try:
        alias_info = client.get_model_version_by_alias(MODEL_NAME, "champion")
        run = client.get_run(alias_info.run_id)
        return {
            "model_name": MODEL_NAME,
            "version": alias_info.version,
            "alias": "champion",
            "run_id": alias_info.run_id,
            "accuracy": run.data.metrics.get("accuracy"),
            "macro_f1": run.data.metrics.get("macro_f1"),
            "model_type": run.data.params.get("model"),
        }
    except Exception as e:
        return {"error": str(e)}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    class_id = int(model.predict([request.text])[0])
    label = TARGET_NAMES[class_id]
    return PredictResponse(
        label=label,
        display_name=DISPLAY_NAMES[label],
        class_id=class_id,
    )

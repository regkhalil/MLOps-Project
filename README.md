w# MLOps Project — 20 Newsgroups Text Classification

End-to-end MLOps pipeline that downloads the [20 Newsgroups](https://scikit-learn.org/stable/datasets/real_world.html#the-20-newsgroups-text-dataset) dataset, preprocesses it, trains multiple TF-IDF classifiers, tracks experiments with MLflow, and serves predictions through a FastAPI backend and Streamlit frontend. The full stack runs on **Docker Compose** (local dev) or **Kubernetes via Kind** (production-like).

---

## Architecture

```
                        ┌───────────────────────────────────────────────┐
                        │              Airflow (scheduler)              │
                        │         Weekly DAG trigger                    │
                        └──────────┬────────────────────────────────────┘
                                   │ Runs 3 tasks (pods / containers)
                     ┌─────────────┼─────────────────┐
                     ▼             ▼                  ▼
               ┌──────────-┐ ┌────────────┐   ┌─────────────┐
               │ Download  │ │ Preprocess │   │   Train     │
               │ (sklearn) │ │ (cleaning) │   │ (models)    │
               └─────┬─────┘ └─────┬──────┘   └──┬──────────┘
                     │             │             │
              write raw data  write clean data   │  log metrics, params,
                     │             │             │  models & artifacts
                     ▼             ▼             ▼
               ┌──────────────────────┐   ┌───────────────-─┐
               │   MinIO (S3)         │   │   MLflow Server │
               │  buckets: data,      │   │   + Model       │
               │  models, mlflow-     │   │     Registry    │
               │  artifacts           │   └───────┬─────────┘
               └──────────────────────┘           │
                                                  │ champion alias
                                                  ▼
                                          ┌───────────────┐
                                          │  FastAPI (API)│
                                          │  POST /predict│
                                          └───────┬───────┘
                                                  │
                                                  ▼
                                          ┌───────────────┐
                                          │ Streamlit (UI)│
                                          └───────────────┘
```

---

## Project Structure

```
MLOps-Project/
├── src/
│   ├── storage.py                  # S3/MinIO helper functions
│   ├── preprocess/
│   │   ├── download.py             # Download 20 Newsgroups → MinIO
│   │   └── preprocess.py           # Text cleaning pipeline
│   ├── train/
│   │   └── train.py                # Train 6 models, champion promotion
│   ├── api/
│   │   └── app.py                  # FastAPI prediction service
│   └── ui/
│       └── app.py                  # Streamlit web interface
├── dags/
│   ├── pipeline_dag.py             # Airflow DAG (Docker Compose)
│   └── pipeline_dag_k8s.py         # Airflow DAG (Kubernetes)
├── k8s/
│   ├── namespace.yaml              # mlops namespace
│   ├── minio.yaml                  # MinIO deployment + PVC + service
│   ├── mlflow.yaml                 # MLflow server deployment
│   ├── airflow.yaml                # Airflow webserver + scheduler
│   ├── api.yaml                    # FastAPI deployment
│   ├── ui.yaml                     # Streamlit deployment
│   └── pipeline-job.yaml           # One-shot pipeline K8s Job
├── Dockerfile.pipeline             # ML pipeline image
├── Dockerfile.mlflow               # MLflow server + boto3
├── Dockerfile.api                  # FastAPI serving image
├── Dockerfile.ui                   # Streamlit UI image
├── Dockerfile.airflow              # Airflow + Docker provider
├── Dockerfile.airflow-k8s          # Airflow + Kubernetes provider
├── docker-compose.yml              # Full local stack
├── kind-config.yaml                # Kind cluster config
├── Makefile                        # Dev & deployment commands
└── pyproject.toml                  # Python dependencies
```

---

## Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| ML Pipeline | scikit-learn 1.8, Python 3.12 | TF-IDF vectorization + classification |
| Experiment Tracking | MLflow 2.x | Parameters, metrics, artifacts, model registry |
| Object Storage | MinIO | S3-compatible storage for data, models, artifacts |
| Orchestration | Apache Airflow 2.10.5 | Weekly DAG scheduling |
| Serving API | FastAPI + Uvicorn | REST prediction endpoint |
| Web UI | Streamlit | Interactive classification interface |
| Containers | Docker / Docker Compose | Local development |
| Kubernetes | Kind | Production-like deployment |

---

## ML Pipeline

### 1. Download
Downloads the 20 Newsgroups dataset (train + test splits) from scikit-learn and uploads raw JSON to `s3://data/raw/`.

### 2. Preprocess
- Strips email headers, footers, and quoting artifacts
- Removes email addresses and non-alphabetic characters
- Lowercases text and collapses whitespace
- Drops documents shorter than 10 characters
- Uploads cleaned data to `s3://data/clean/`
- Logs preprocessing metrics to MLflow

### 3. Train
Trains **6 model configurations** and compares them:

| Model | Variant | Key Hyperparameters |
|-------|---------|-------------------|
| SGDClassifier | `SGD_alpha1e-4` | loss=hinge, alpha=1e-4 |
| SGDClassifier | `SGD_alpha1e-3` | loss=hinge, alpha=1e-3 |
| MultinomialNB | `NaiveBayes_alpha0.1` | alpha=0.1 |
| MultinomialNB | `NaiveBayes_alpha1.0` | alpha=1.0 |
| LogisticRegression | `LogReg_C1` | C=1.0 |
| LogisticRegression | `LogReg_C10` | C=10.0 |

All models use TF-IDF vectorization (30k features, bigrams, sublinear TF) and are wrapped in an sklearn `Pipeline`.

### Champion Model Promotion
After training, the best model (by macro F1) is compared against the current **champion** in the MLflow Model Registry. The champion alias is only updated if the new model is strictly better, preventing regressions across pipeline runs.

---

## Getting Started

### Prerequisites
- Docker & Docker Compose
- Make
- (For K8s) [Kind](https://kind.sigs.k8s.io/) and `kubectl`

### Docker Compose (Local Dev)

```bash
# Build images and start all services
make dev

# Run the ML pipeline (first time or manually)
make pipeline-run

# Stop services
make down

# Stop and remove volumes
make clean
```

Once running:

| Service | URL | Credentials |
|---------|-----|-------------|
| Airflow UI | http://localhost:8080 | admin / admin |
| MLflow UI | http://localhost:5000 | — |
| MinIO Console | http://localhost:9001 | minioadmin / minioadmin |
| Prediction API | http://localhost:8000 | — |
| Streamlit UI | http://localhost:8501 | — |

### Kubernetes (Kind)

```bash
# Full setup: create cluster, build images, load into Kind, deploy manifests
make k8s-dev

# Run the pipeline as a K8s Job
make k8s-pipeline-run

# Tear down resources (keep cluster)
make k8s-down

# Delete the Kind cluster entirely
make k8s-clean
```

Same URLs apply — Kind maps NodePort services to the same host ports.

---

## API Usage

### Health Check
```bash
curl http://localhost:8000/health
# {"status": "ok", "model_loaded": true}
```

### Predict
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "NASA launched a new telescope into orbit"}'
# {"label": "sci.space", "display_name": "Space & Astronomy", "class_id": 14}
```

### Model Info
```bash
curl http://localhost:8000/model-info
# {"model_name": "20newsgroups-classifier", "version": "3", "alias": "champion", ...}
```

> **Note:** The API starts gracefully without a model (returns `model_loaded: false`) and responds with HTTP 503 on `/predict` until the pipeline has run at least once.

---

## Orchestration

Airflow runs a `newsgroups_tfidf_pipeline` DAG scheduled `@weekly` with three sequential tasks:

```
download → preprocess → train
```

- **Docker Compose**: Uses `DockerOperator` — each task spawns a container from the pipeline image on the Docker socket.
- **Kubernetes**: Uses `KubernetesPodOperator` — each task spawns a Pod in the `mlops` namespace.

---

## Categories

The classifier predicts across 20 newsgroup categories:

| Internal Label | Display Name |
|---------------|-------------|
| alt.atheism | Atheism & Secularism |
| comp.graphics | Computer Graphics |
| comp.os.ms-windows.misc | Windows OS |
| comp.sys.ibm.pc.hardware | PC Hardware |
| comp.sys.mac.hardware | Mac Hardware |
| comp.windows.x | X Window System |
| misc.forsale | For Sale |
| rec.autos | Automobiles |
| rec.motorcycles | Motorcycles |
| rec.sport.baseball | Baseball |
| rec.sport.hockey | Hockey |
| sci.crypt | Cryptography |
| sci.electronics | Electronics |
| sci.med | Medicine & Health |
| sci.space | Space & Astronomy |
| soc.religion.christian | Christianity |
| talk.politics.guns | Gun Politics |
| talk.politics.mideast | Middle East Politics |
| talk.politics.misc | General Politics |
| talk.religion.misc | Religion & Beliefs |
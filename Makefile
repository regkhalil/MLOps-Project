.PHONY: dev down build pipeline-build pipeline-run clean \
       k8s-create k8s-build k8s-load k8s-deploy k8s-dev k8s-pipeline-run k8s-down k8s-clean

# Build the pipeline image and start all services
dev: pipeline-build build
	docker compose up -d
	@echo ""
	@echo "Services started:"
	@echo "  Airflow UI:    http://localhost:8080  (admin / admin)"
	@echo "  MinIO Console: http://localhost:9001  (minioadmin / minioadmin)"
	@echo "  MLflow UI:     http://localhost:5000"
	@echo ""

# Build the standalone pipeline image
pipeline-build:
	docker build -t mlops-pipeline:latest -f Dockerfile.pipeline .

# Build Airflow images
build:
	docker compose build

# Trigger a manual pipeline run via Airflow
pipeline-run:
	docker exec mlops-airflow-scheduler airflow dags trigger newsgroups_tfidf_pipeline

# Stop all services
down:
	docker compose down

# Stop all services and remove volumes
clean:
	docker compose down -v

# ── Kubernetes (Kind) ────────────────────────────────────────────

CLUSTER_NAME := mlops

# Create the Kind cluster
k8s-create:
	kind create cluster --name $(CLUSTER_NAME) --config kind-config.yaml

# Build all Docker images for K8s
k8s-build: pipeline-build
	docker build -t mlops-mlflow:latest -f Dockerfile.mlflow .
	docker build -t mlops-api:latest -f Dockerfile.api .
	docker build -t mlops-ui:latest -f Dockerfile.ui .
	docker build -t mlops-airflow-k8s:latest -f Dockerfile.airflow-k8s .

# Load images into the Kind cluster
k8s-load:
	kind load docker-image mlops-pipeline:latest --name $(CLUSTER_NAME)
	kind load docker-image mlops-mlflow:latest --name $(CLUSTER_NAME)
	kind load docker-image mlops-api:latest --name $(CLUSTER_NAME)
	kind load docker-image mlops-ui:latest --name $(CLUSTER_NAME)
	kind load docker-image mlops-airflow-k8s:latest --name $(CLUSTER_NAME)

# Deploy all K8s manifests
k8s-deploy:
	kubectl apply -f k8s/namespace.yaml
	kubectl apply -f k8s/minio.yaml
	kubectl apply -f k8s/mlflow.yaml
	kubectl apply -f k8s/airflow.yaml
	kubectl apply -f k8s/api.yaml
	kubectl apply -f k8s/ui.yaml
	@echo ""
	@echo "Waiting for pods to be ready..."
	kubectl -n mlops wait --for=condition=ready pod -l app=minio --timeout=120s
	kubectl -n mlops wait --for=condition=ready pod -l app=mlflow --timeout=120s
	@echo ""
	@echo "Services:"
	@echo "  MinIO Console: http://localhost:9001  (minioadmin / minioadmin)"
	@echo "  MinIO API:     http://localhost:9000"
	@echo "  MLflow UI:     http://localhost:5000"
	@echo "  Airflow UI:    http://localhost:8080  (admin / admin)"
	@echo "  API:           http://localhost:8000"
	@echo "  Streamlit UI:  http://localhost:8501"

# Full K8s dev setup: create cluster, build, load, deploy
k8s-dev: k8s-create k8s-build k8s-load k8s-deploy

# Run the pipeline as a K8s Job
k8s-pipeline-run:
	-kubectl -n mlops delete job pipeline-run --ignore-not-found
	kubectl apply -f k8s/pipeline-job.yaml
	kubectl -n mlops wait --for=condition=complete job/pipeline-run --timeout=600s
	kubectl -n mlops logs job/pipeline-run --all-containers

# Delete all K8s resources but keep the cluster
k8s-down:
	-kubectl delete -f k8s/ --ignore-not-found

# Delete the Kind cluster entirely
k8s-clean:
	kind delete cluster --name $(CLUSTER_NAME)

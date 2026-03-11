.PHONY: dev down build pipeline-build pipeline-run clean

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

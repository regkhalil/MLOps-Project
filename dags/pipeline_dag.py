"""
Airflow DAG: 20 Newsgroups TF-IDF Pipeline

Runs weekly: download → preprocess → train

Each task runs in an isolated Docker container built from Dockerfile.pipeline.
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator

PIPELINE_IMAGE = "mlops-pipeline:latest"

# Environment variables passed to the pipeline containers for MinIO access
PIPELINE_ENV = {
    "S3_ENDPOINT_URL": "http://minio:9000",
    "AWS_ACCESS_KEY_ID": "minioadmin",
    "AWS_SECRET_ACCESS_KEY": "minioadmin",
}

default_args = {
    "owner": "mlops",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="newsgroups_tfidf_pipeline",
    default_args=default_args,
    description="Weekly 20 Newsgroups TF-IDF classification pipeline",
    schedule="@weekly",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["mlops", "nlp"],
) as dag:

    download = DockerOperator(
        task_id="download",
        image=PIPELINE_IMAGE,
        command="python -m src.preprocess.download",
        environment=PIPELINE_ENV,
        network_mode="mlops-project_default",
        auto_remove="success",
        docker_url="unix://var/run/docker.sock",
        mount_tmp_dir=False,
    )

    preprocess = DockerOperator(
        task_id="preprocess",
        image=PIPELINE_IMAGE,
        command="python -m src.preprocess.preprocess",
        environment=PIPELINE_ENV,
        network_mode="mlops-project_default",
        auto_remove="success",
        docker_url="unix://var/run/docker.sock",
        mount_tmp_dir=False,
    )

    train = DockerOperator(
        task_id="train",
        image=PIPELINE_IMAGE,
        command="python -m src.train.train",
        environment=PIPELINE_ENV,
        network_mode="mlops-project_default",
        auto_remove="success",
        docker_url="unix://var/run/docker.sock",
        mount_tmp_dir=False,
    )

    download >> preprocess >> train


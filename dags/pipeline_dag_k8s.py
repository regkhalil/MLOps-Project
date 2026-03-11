"""
Airflow DAG: 20 Newsgroups TF-IDF Pipeline (Kubernetes)

Runs weekly: download → preprocess → train

Each task runs as a Kubernetes Pod using KubernetesPodOperator.
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator

PIPELINE_IMAGE = "mlops-pipeline:latest"
NAMESPACE = "mlops"

PIPELINE_ENV_VARS = {
    "S3_ENDPOINT_URL": "http://minio.mlops.svc.cluster.local:9000",
    "AWS_ACCESS_KEY_ID": "minioadmin",
    "AWS_SECRET_ACCESS_KEY": "minioadmin",
    "MLFLOW_TRACKING_URI": "http://mlflow.mlops.svc.cluster.local:5000",
    "GIT_PYTHON_REFRESH": "quiet",
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

    download = KubernetesPodOperator(
        task_id="download",
        name="pipeline-download",
        namespace=NAMESPACE,
        image=PIPELINE_IMAGE,
        cmds=["python", "-m", "src.preprocess.download"],
        env_vars=PIPELINE_ENV_VARS,
        image_pull_policy="Never",
        is_delete_operator_pod=True,
        get_logs=True,
        startup_timeout_seconds=300,
    )

    preprocess = KubernetesPodOperator(
        task_id="preprocess",
        name="pipeline-preprocess",
        namespace=NAMESPACE,
        image=PIPELINE_IMAGE,
        cmds=["python", "-m", "src.preprocess.preprocess"],
        env_vars=PIPELINE_ENV_VARS,
        image_pull_policy="Never",
        is_delete_operator_pod=True,
        get_logs=True,
        startup_timeout_seconds=300,
    )

    train = KubernetesPodOperator(
        task_id="train",
        name="pipeline-train",
        namespace=NAMESPACE,
        image=PIPELINE_IMAGE,
        cmds=["python", "-m", "src.train.train"],
        env_vars=PIPELINE_ENV_VARS,
        image_pull_policy="Never",
        is_delete_operator_pod=True,
        get_logs=True,
        startup_timeout_seconds=300,
    )

    download >> preprocess >> train

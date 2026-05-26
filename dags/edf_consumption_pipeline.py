"""
Airflow DAG - EDF electricity consumption ML pipeline.

Steps: extract -> clean -> aggregate -> features -> splits -> train ->
validate -> register -> predict -> write Snowflake.

Default schedule: weekly Monday 02:00 UTC.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from pipeline.steps import (  # noqa: E402
    run_aggregate,
    run_clean,
    run_extract,
    run_features,
    run_load_splits,
    run_predictions,
    run_register_model,
    run_train,
    run_validate,
    run_write_snowflake,
)

default_args = {
    "owner": "mspr3-edf",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
}


def _runner(fn):
    """Build an Airflow callable that passes dag run_id to pipeline steps."""

    def _callable(**context):
        dag_run = context.get("dag_run")
        run_id = dag_run.run_id if dag_run else None
        return fn(run_id=run_id)

    return _callable


with DAG(
    dag_id="edf_consumption_ml_pipeline",
    default_args=default_args,
    description="EDF ML pipeline: ETL, training, predictions, Snowflake",
    schedule_interval=os.getenv("AIRFLOW_SCHEDULE", "0 2 * * 1"),
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["edf", "ml", "eco2mix", "snowflake"],
    max_active_runs=1,
) as dag:

    extract_data = PythonOperator(
        task_id="extract_data",
        python_callable=_runner(run_extract),
    )

    clean_data = PythonOperator(
        task_id="clean_data",
        python_callable=_runner(run_clean),
    )

    aggregate_daily_task = PythonOperator(
        task_id="aggregate_daily",
        python_callable=_runner(run_aggregate),
    )

    feature_engineering = PythonOperator(
        task_id="feature_engineering",
        python_callable=_runner(run_features),
    )

    load_splits = PythonOperator(
        task_id="load_splits",
        python_callable=_runner(run_load_splits),
    )

    train_models = PythonOperator(
        task_id="train_models",
        python_callable=_runner(run_train),
    )

    validate_models = PythonOperator(
        task_id="validate_and_test",
        python_callable=_runner(run_validate),
    )

    register_model = PythonOperator(
        task_id="register_model",
        python_callable=_runner(run_register_model),
    )

    generate_predictions = PythonOperator(
        task_id="generate_predictions",
        python_callable=_runner(run_predictions),
    )

    write_snowflake = PythonOperator(
        task_id="write_forecasts_to_snowflake",
        python_callable=_runner(run_write_snowflake),
    )

    (
        extract_data
        >> clean_data
        >> aggregate_daily_task
        >> feature_engineering
        >> load_splits
        >> train_models
        >> validate_models
        >> register_model
        >> generate_predictions
        >> write_snowflake
    )

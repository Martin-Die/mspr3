FROM apache/airflow:2.9.3-python3.11

USER root

COPY requirements.txt /tmp/
COPY requirements-airflow.txt /tmp/

RUN python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir -r /tmp/requirements-airflow.txt

USER airflow

ENV PYTHONPATH=/opt/airflow/mspr3
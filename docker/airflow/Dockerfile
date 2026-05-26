FROM apache/airflow:2.9.3-python3.11

# pip doit tourner en utilisateur airflow (image officielle Apache Airflow)
USER airflow

COPY --chown=airflow:root requirements.txt requirements-airflow.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements-airflow.txt

ENV PYTHONPATH=/opt/airflow/mspr3

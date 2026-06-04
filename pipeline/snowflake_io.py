"""
Write forecasts to Snowflake.

Variables d'environnement :
  SNOWFLAKE_ACCOUNT, SNOWFLAKE_USER, SNOWFLAKE_PASSWORD
  SNOWFLAKE_WAREHOUSE, SNOWFLAKE_DATABASE, SNOWFLAKE_SCHEMA
  SNOWFLAKE_TABLE (defaut : EDF_CONSUMPTION_FORECASTS)
  SNOWFLAKE_ROLE (optionnel)
  SNOWFLAKE_SKIP_WRITE=1 pour ignorer l'ecriture en local
"""

from __future__ import annotations

import logging
import os

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_TABLE = "EDF_CONSUMPTION_FORECASTS"


def _snowflake_config() -> dict:
    required = ("SNOWFLAKE_ACCOUNT", "SNOWFLAKE_USER", "SNOWFLAKE_PASSWORD")
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        raise EnvironmentError(
            f"Variables Snowflake manquantes : {', '.join(missing)}. "
            "En dev local, definissez SNOWFLAKE_SKIP_WRITE=1 pour ignorer l'ecriture."
        )
    return {
        "account": os.environ["SNOWFLAKE_ACCOUNT"],
        "user": os.environ["SNOWFLAKE_USER"],
        "password": os.environ["SNOWFLAKE_PASSWORD"],
        "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH"),
        "database": os.getenv("SNOWFLAKE_DATABASE", "EDF_ML"),
        "schema": os.getenv("SNOWFLAKE_SCHEMA", "PUBLIC"),
        "role": os.getenv("SNOWFLAKE_ROLE"),
    }


def _ensure_table(cursor, database: str, schema: str, table: str) -> None:
    cursor.execute(f"USE DATABASE {database}")
    cursor.execute(f"USE SCHEMA {schema}")
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {table} (
            prediction_date     TIMESTAMP_NTZ,
            prediction_mw       FLOAT,
            actual_mw           FLOAT,
            model_name          VARCHAR(128),
            pipeline_run_id     VARCHAR(256),
            predicted_at        TIMESTAMP_NTZ,
            prevision_rte_j1_mw FLOAT,
            loaded_at           TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
        )
    """)


def write_predictions_to_snowflake(df: pd.DataFrame) -> int:
    """Insere les previsions ; retourne le nombre de lignes ecrites."""
    if os.getenv("SNOWFLAKE_SKIP_WRITE", "").lower() in ("1", "true", "yes"):
        logger.warning("[Snowflake] Ecriture ignoree (SNOWFLAKE_SKIP_WRITE)")
        return len(df)

    import snowflake.connector

    cfg = _snowflake_config()
    table = os.getenv("SNOWFLAKE_TABLE", DEFAULT_TABLE)
    database = cfg["database"]
    schema = cfg["schema"]

    out = df.copy()
    if "prediction_date" in out.columns:
        out["prediction_date"] = pd.to_datetime(out["prediction_date"])

    connect_kwargs = {
        "account": cfg["account"],
        "user": cfg["user"],
        "password": cfg["password"],
        "warehouse": cfg["warehouse"],
        "database": database,
        "schema": schema,
    }
    if cfg.get("role"):
        connect_kwargs["role"] = cfg["role"]

    logger.info("[Snowflake] Connexion -> %s.%s.%s", database, schema, table)
    conn = snowflake.connector.connect(**connect_kwargs)
    try:
        cur = conn.cursor()
        _ensure_table(cur, database, schema, table)

        cols = [
            "prediction_date",
            "prediction_mw",
            "actual_mw",
            "model_name",
            "pipeline_run_id",
            "predicted_at",
            "prevision_rte_j1_mw",
        ]
        present = [c for c in cols if c in out.columns]
        rows = [tuple(r) for r in out[present].itertuples(index=False, name=None)]

        placeholders = ", ".join(["%s"] * len(present))
        col_list = ", ".join(present)
        sql = f"INSERT INTO {table} ({col_list}) VALUES ({placeholders})"
        cur.executemany(sql, rows)
        conn.commit()
        logger.info("[Snowflake] %d ligne(s) inseree(s)", len(rows))
        return len(rows)
    finally:
        conn.close()

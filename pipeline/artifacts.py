"""Artifact storage between Airflow tasks (avoids large XCom payloads)."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

import joblib
import pandas as pd

ARTIFACTS_DIR = Path(os.getenv("PIPELINE_ARTIFACTS_DIR", "artifacts"))


def artifact_path(name: str, run_id: str | None = None) -> Path:
    base = ARTIFACTS_DIR
    if run_id:
        base = base / run_id
    base.mkdir(parents=True, exist_ok=True)
    return base / name


def _parquet_safe(df: pd.DataFrame) -> pd.DataFrame:
    """Colonnes object (texte melange) -> str pour eviter les erreurs PyArrow."""
    out = df.copy()
    for col in out.select_dtypes(include=["object"]).columns:
        out[col] = out[col].astype(str)
    return out


def save_parquet(df: pd.DataFrame, name: str, run_id: str | None = None) -> Path:
    path = artifact_path(name, run_id)
    _parquet_safe(df).to_parquet(path)
    return path


def load_parquet(name: str, run_id: str | None = None) -> pd.DataFrame:
    path = artifact_path(name, run_id)
    if not path.exists():
        raise FileNotFoundError(f"Artefact manquant : {path}")
    return pd.read_parquet(path)


def save_splits(
    x_train,
    x_val,
    x_test,
    y_train,
    y_val,
    y_test,
    feats: list[str],
    run_id: str | None = None,
) -> Path:
    path = artifact_path("splits.joblib", run_id)
    joblib.dump(
        {
            "x_train": x_train,
            "x_val": x_val,
            "x_test": x_test,
            "y_train": y_train,
            "y_val": y_val,
            "y_test": y_test,
            "feats": feats,
        },
        path,
    )
    return path


def load_splits(run_id: str | None = None) -> dict:
    path = artifact_path("splits.joblib", run_id)
    if not path.exists():
        raise FileNotFoundError(f"Artefact manquant : {path}")
    return joblib.load(path)


def save_meta(data: dict, name: str = "run_meta.json", run_id: str | None = None) -> Path:
    path = artifact_path(name, run_id)
    payload = {**data, "updated_at": datetime.now(UTC).isoformat()}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_meta(name: str = "run_meta.json", run_id: str | None = None) -> dict:
    path = artifact_path(name, run_id)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))

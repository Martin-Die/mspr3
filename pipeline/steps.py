"""Airflow step functions (one per ML pipeline phase)."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import joblib
import pandas as pd

from models import (
    R2_THRESHOLD,
    benchmark_rte,
    evaluate_all,
    save_model_if_better,
    select_best_model,
    train_all,
)
from pipeline.artifacts import (
    artifact_path,
    load_meta,
    load_parquet,
    load_splits,
    save_meta,
    save_parquet,
    save_splits,
)
from pipeline.snowflake_io import write_predictions_to_snowflake
from preprocessing.extract import extract
from preprocessing.load import load as load_splits_fn
from preprocessing.transform import aggregate_daily, clean, engineer_features

logger = logging.getLogger(__name__)

# Repo root (…/mspr3). Airflow tasks often run with cwd=/opt/airflow,
# so we must resolve data paths relative to the project, not the cwd.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _resolve_run_id(run_id: str | None) -> str | None:
    return run_id or os.getenv("AIRFLOW_CTX_DAG_RUN_ID")


_DATA_EXTENSIONS = (".xls", ".xlsx", ".tsv", ".csv")
_DEFAULT_DATA_DIR = _PROJECT_ROOT / "data"


def _files_in_data_dir(directory: Path) -> list[Path]:
    """Liste tous les fichiers de donnees dans un dossier (ignore .gitkeep, etc.)."""
    if not directory.is_dir():
        return []
    files = [
        p
        for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in _DATA_EXTENSIONS
    ]
    return sorted(files, key=lambda p: p.name.lower())


def _resolve_path(raw: str) -> Path:
    p = Path(raw.strip())
    if not p.is_absolute():
        p = (_PROJECT_ROOT / p).resolve()
    return p


def _data_paths() -> list[Path]:
    """
    Chemins des fichiers eCO2mix.

    - ECO2MIX_DATA_PATHS vide ou ``data`` / ``data/`` : tous les fichiers dans data/
    - valeur = dossier : tous les fichiers de ce dossier
    - valeur = fichier : ce fichier uniquement
    - plusieurs chemins separes par des virgules
    """
    env = os.getenv("ECO2MIX_DATA_PATHS", "").strip()
    if not env or env.rstrip("/").replace("\\", "/") in {"data", "."}:
        return _files_in_data_dir(_DEFAULT_DATA_DIR)

    paths: list[Path] = []
    for raw in (p.strip() for p in env.split(",") if p.strip()):
        p = _resolve_path(raw)
        if p.is_dir():
            paths.extend(_files_in_data_dir(p))
        else:
            paths.append(p)

    dedup: list[Path] = []
    seen: set[str] = set()
    for p in paths:
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        dedup.append(p)
    return dedup


def run_extract(run_id: str | None = None) -> dict:
    """Extraction : lecture et fusion des fichiers RTE eCO2mix."""
    run_id = _resolve_run_id(run_id)
    paths = _data_paths()
    if not paths:
        raise FileNotFoundError(
            f"Aucun fichier de donnees dans {_DEFAULT_DATA_DIR} "
            f"(extensions: {', '.join(_DATA_EXTENSIONS)}). "
            "Placez vos fichiers eCO2mix dans data/ ou definissez ECO2MIX_DATA_PATHS=data"
        )
    logger.info("[Extract] %d fichier(s)", len(paths))
    raw = extract(paths)
    save_parquet(raw, "raw.parquet", run_id)
    meta = {"data_files": [str(p) for p in paths], "rows_raw": len(raw)}
    save_meta(meta, run_id=run_id)
    return meta


def run_clean(run_id: str | None = None) -> dict:
    """Preparation / nettoyage des donnees brutes."""
    run_id = _resolve_run_id(run_id)
    raw = load_parquet("raw.parquet", run_id)
    cleaned = clean(raw)
    save_parquet(cleaned, "cleaned.parquet", run_id)
    return {"rows_cleaned": len(cleaned)}


def run_aggregate(run_id: str | None = None) -> dict:
    """Agregation journaliere."""
    run_id = _resolve_run_id(run_id)
    cleaned = load_parquet("cleaned.parquet", run_id)
    daily = aggregate_daily(cleaned)
    save_parquet(daily, "daily.parquet", run_id)
    return {"days": len(daily)}


def run_features(run_id: str | None = None) -> dict:
    """Feature engineering."""
    run_id = _resolve_run_id(run_id)
    daily = load_parquet("daily.parquet", run_id)
    featured = engineer_features(daily)
    save_parquet(featured, "featured.parquet", run_id)
    return {"rows_featured": len(featured), "columns": list(featured.columns)}


def run_load_splits(run_id: str | None = None) -> dict:
    """Decoupage train / validation / test."""
    run_id = _resolve_run_id(run_id)
    featured = load_parquet("featured.parquet", run_id)
    x_train, x_val, x_test, y_train, y_val, y_test, feats = load_splits_fn(featured)
    save_splits(x_train, x_val, x_test, y_train, y_val, y_test, feats, run_id)
    return {
        "n_train": len(x_train),
        "n_val": len(x_val),
        "n_test": len(x_test),
        "n_features": len(feats),
    }


def run_train(run_id: str | None = None) -> dict:
    """Entrainement des 4 modeles."""
    run_id = _resolve_run_id(run_id)
    splits = load_splits(run_id)
    trained = train_all(splits["x_train"], splits["y_train"])
    path = artifact_path("trained_models.joblib", run_id)
    joblib.dump(trained, path)
    return {"models": list(trained.keys())}


def run_validate(run_id: str | None = None) -> dict:
    """Evaluation, tests unitaires et seuils de qualite."""
    run_id = _resolve_run_id(run_id)
    splits = load_splits(run_id)
    trained = joblib.load(artifact_path("trained_models.joblib", run_id))

    featured = load_parquet("featured.parquet", run_id)
    benchmark = {}
    if "prevision_j1" in featured.columns:
        test_idx = featured.index >= (
            "2024-10-01" if featured.index.max().year > 2023 else "2023-10-01"
        )
        test_df = featured[test_idx]
        if len(test_df) > 0 and "consommation" in test_df.columns:
            benchmark = benchmark_rte(test_df["consommation"], test_df["prevision_j1"])

    results = evaluate_all(
        trained,
        splits["x_val"],
        splits["y_val"],
        splits["x_test"],
        splits["y_test"],
    )
    results.to_csv(artifact_path("evaluation_results.csv", run_id), index=False)

    best_row = results.iloc[0]
    test_metrics = {
        "r2_test": float(best_row["r2_test"]),
        "rmse_test": float(best_row["rmse_test"]),
        "mape_test": float(best_row["mape_test"]),
        "model": str(best_row["model"]),
    }

    if test_metrics["r2_test"] < R2_THRESHOLD:
        logger.warning(
            "[Validate] R2 test %.4f < seuil %.2f",
            test_metrics["r2_test"],
            R2_THRESHOLD,
        )

    selected = select_best_model(results, benchmark)
    save_meta(
        {
            "evaluation": test_metrics,
            "benchmark": benchmark,
            "selected_model": selected or test_metrics["model"],
            "beats_benchmark": selected is not None,
        },
        run_id=run_id,
    )

    root = Path(__file__).resolve().parent.parent
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"],
        cwd=root,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        logger.error("[Validate] pytest echoue:\n%s", proc.stderr)
        raise RuntimeError("Tests automatises en echec (pytest)")

    return {
        "best_model": test_metrics["model"],
        "r2_test": test_metrics["r2_test"],
        "pytest": "ok",
    }


def run_register_model(run_id: str | None = None) -> dict:
    """Enregistrement du meilleur modele dans models/."""
    run_id = _resolve_run_id(run_id)
    meta = load_meta(run_id=run_id)
    model_name = meta.get("selected_model") or meta.get("evaluation", {}).get("model")
    if not model_name:
        raise ValueError("No model selected; run run_validate first.")

    trained = joblib.load(artifact_path("trained_models.joblib", run_id))
    test_metrics = meta.get("evaluation", {})
    path = save_model_if_better(
        trained[model_name]["pipeline"],
        model_name,
        test_metrics,
    )
    return {
        "model_name": model_name,
        "saved": path is not None,
        "path": str(path) if path else None,
    }


def run_predictions(run_id: str | None = None) -> dict:
    """Predictions batch sur le jeu de test."""
    run_id = _resolve_run_id(run_id)
    meta = load_meta(run_id=run_id)
    model_name = meta.get("selected_model") or meta.get("evaluation", {}).get("model")
    if not model_name:
        raise ValueError("Modele non defini dans run_meta.json")

    trained = joblib.load(artifact_path("trained_models.joblib", run_id))
    pipe = trained[model_name]["pipeline"]
    splits = load_splits(run_id)

    y_pred = pipe.predict(splits["x_test"])
    df = pd.DataFrame(
        {
            "prediction_date": splits["x_test"].index,
            "prediction_mw": y_pred.round(1),
            "actual_mw": splits["y_test"].values,
            "model_name": model_name,
            "pipeline_run_id": run_id or "local",
            "predicted_at": datetime.now(UTC).isoformat(),
        }
    )
    if "prevision_j1" in splits["x_test"].columns:
        df["prevision_rte_j1_mw"] = splits["x_test"]["prevision_j1"].values

    save_parquet(df, "predictions.parquet", run_id)
    return {"n_predictions": len(df), "model_name": model_name}


def run_write_snowflake(run_id: str | None = None) -> dict:
    """Ecriture des previsions dans Snowflake."""
    run_id = _resolve_run_id(run_id)
    df = load_parquet("predictions.parquet", run_id)
    n = write_predictions_to_snowflake(df)
    return {"rows_written": n}

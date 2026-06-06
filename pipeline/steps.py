"""Airflow step functions (one per ML pipeline phase)."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
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

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ─────────────────────────────────────────────────────────────────────────────
# Config MLflow
# ─────────────────────────────────────────────────────────────────────────────

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MLFLOW_EXPERIMENT   = os.getenv("MLFLOW_EXPERIMENT", "edf-consumption")
REGISTERED_MODEL    = "edf-random-forest"
MAPE_PROD_THRESHOLD = float(os.getenv("MAPE_PRODUCTION_THRESHOLD", "5.0"))

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


def _get_or_create_experiment() -> str:
    exp = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT)
    if exp is None:
        exp_id = mlflow.create_experiment(
            MLFLOW_EXPERIMENT,
            tags={"project": "mspr3-edf", "team": "epsi-rncp36582"},
        )
        logger.info("[MLflow] Experiment créé : %s (id=%s)", MLFLOW_EXPERIMENT, exp_id)
        return exp_id
    return exp.experiment_id


# ─────────────────────────────────────────────────────────────────────────────
# Helpers (inchangés)
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_run_id(run_id: str | None) -> str | None:
    return run_id or os.getenv("AIRFLOW_CTX_DAG_RUN_ID")


_DATA_EXTENSIONS = (".xls", ".xlsx", ".tsv", ".csv")
_DEFAULT_DATA_DIR = _PROJECT_ROOT / "data"


def _files_in_data_dir(directory: Path) -> list[Path]:
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


# ─────────────────────────────────────────────────────────────────────────────
# Étapes ETL — inchangées
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# run_train — MLflow : log params + artefact + modèle sklearn
# ─────────────────────────────────────────────────────────────────────────────

def run_train(run_id: str | None = None) -> dict:
    """Entrainement des 4 modeles. Log MLflow : params, durée, artefact joblib."""
    run_id = _resolve_run_id(run_id)
    splits = load_splits(run_id)

    exp_id = _get_or_create_experiment()

    with mlflow.start_run(
        experiment_id=exp_id,
        run_name=f"train_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}",
        tags={
            "airflow_run_id": run_id or "manual",
            "step": "train",
        },
    ) as active_run:
        mlflow_run_id = active_run.info.run_id
        logger.info("[MLflow] Run démarré : %s", mlflow_run_id)

        mlflow.log_param("n_train", len(splits["x_train"]))
        mlflow.log_param("n_features", len(splits.get("feature_names", [])))

        import time
        t0 = time.time()
        trained = train_all(splits["x_train"], splits["y_train"])
        train_duration = round(time.time() - t0, 2)

        mlflow.log_metric("train_duration_s", train_duration)
        mlflow.log_param("models_trained", ",".join(trained.keys()))

        # Sauvegarder l'artefact joblib (chemin standard du projet)
        path = artifact_path("trained_models.joblib", run_id)
        joblib.dump(trained, path)
        mlflow.log_artifact(str(path), artifact_path="joblib_models")

        # Log chaque sous-modèle sklearn dans MLflow
        for model_name, model_info in trained.items():
            try:
                mlflow.sklearn.log_model(
                    sk_model=model_info["pipeline"],
                    artifact_path=f"sklearn_{model_name}",
                )
            except Exception as exc:
                logger.warning("[MLflow] Impossible de logger %s : %s", model_name, exc)

        # Sauvegarder le mlflow_run_id pour les étapes suivantes
        _save_mlflow_run_id(mlflow_run_id, run_id)

    logger.info("[MLflow] run_train terminé | mlflow_run_id=%s", mlflow_run_id)
    return {"models": list(trained.keys()), "mlflow_run_id": mlflow_run_id}


# ─────────────────────────────────────────────────────────────────────────────
# run_validate — MLflow : log métriques val/test sur le run existant
# ─────────────────────────────────────────────────────────────────────────────

def run_validate(run_id: str | None = None) -> dict:
    """Evaluation, tests unitaires et seuils de qualite. Log MLflow : métriques."""
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
        "r2_test":   float(best_row["r2_test"]),
        "rmse_test": float(best_row["rmse_test"]),
        "mape_test": float(best_row["mape_test"]),
        "model":     str(best_row["model"]),
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

    # ── Log MLflow sur le run ouvert par run_train ──────────────────────────
    mlflow_run_id = _load_mlflow_run_id(run_id)
    if mlflow_run_id:
        with mlflow.start_run(run_id=mlflow_run_id):
            # Métriques du meilleur modèle sur val et test
            mlflow.log_metric("val_r2",        float(best_row.get("r2_val",   0)))
            mlflow.log_metric("val_mape_pct",  float(best_row.get("mape_val", 0)))
            mlflow.log_metric("test_r2",       test_metrics["r2_test"])
            mlflow.log_metric("test_rmse_mw",  test_metrics["rmse_test"])
            mlflow.log_metric("test_mape_pct", test_metrics["mape_test"])
            mlflow.log_param("selected_model", selected or test_metrics["model"])
            mlflow.log_param("beats_rte_benchmark", str(selected is not None))

            # Log le CSV des résultats comme artefact
            mlflow.log_artifact(
                str(artifact_path("evaluation_results.csv", run_id)),
                artifact_path="evaluation",
            )
        logger.info("[MLflow] Métriques de validation loggées sur run=%s", mlflow_run_id)
    else:
        logger.warning("[MLflow] Pas de mlflow_run_id trouvé — métriques non loggées")

    # ── Tests automatisés (pytest) ───────────────────────────────────────────
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
        "r2_test":    test_metrics["r2_test"],
        "pytest":     "ok",
        "mlflow_run_id": mlflow_run_id,
    }


# ─────────────────────────────────────────────────────────────────────────────
# run_register_model — MLflow Model Registry + metrics.json pour l'API
# ─────────────────────────────────────────────────────────────────────────────

def run_register_model(run_id: str | None = None) -> dict:
    """Enregistrement du meilleur modele dans models/ + MLflow Registry."""
    run_id = _resolve_run_id(run_id)
    meta = load_meta(run_id=run_id)
    model_name = meta.get("selected_model") or meta.get("evaluation", {}).get("model")
    if not model_name:
        raise ValueError("No model selected; run run_validate first.")

    trained = joblib.load(artifact_path("trained_models.joblib", run_id))
    test_metrics = meta.get("evaluation", {})

    # Sauvegarde joblib standard (utilisée par l'API FastAPI)
    path = save_model_if_better(
        trained[model_name]["pipeline"],
        model_name,
        test_metrics,
    )

    # ── MLflow Model Registry ────────────────────────────────────────────────
    mlflow_run_id = _load_mlflow_run_id(run_id)
    registry_version = None
    registry_stage   = None

    if mlflow_run_id:
        model_uri = f"runs:/{mlflow_run_id}/sklearn_{model_name}"
        try:
            mv = mlflow.register_model(
                model_uri=model_uri,
                name=REGISTERED_MODEL,
                tags={
                    "airflow_run_id": run_id or "manual",
                    "mape_pct":       str(round(test_metrics.get("mape_test", 999), 4)),
                    "selected_model": model_name,
                },
            )
            registry_version = mv.version

            client = mlflow.MlflowClient()
            client.transition_model_version_stage(
                name=REGISTERED_MODEL,
                version=registry_version,
                stage="Staging",
                archive_existing_versions=False,
            )
            registry_stage = "Staging"
            logger.info("[MLflow] %s v%s → Staging", REGISTERED_MODEL, registry_version)

            mape = test_metrics.get("mape_test", 999)
            if mape <= MAPE_PROD_THRESHOLD:
                client.transition_model_version_stage(
                    name=REGISTERED_MODEL,
                    version=registry_version,
                    stage="Production",
                    archive_existing_versions=True,
                )
                registry_stage = "Production"
                logger.info(
                    "[MLflow] %s v%s → Production (MAPE=%.2f%% ≤ %.2f%%)",
                    REGISTERED_MODEL, registry_version, mape, MAPE_PROD_THRESHOLD,
                )
            else:
                logger.warning(
                    "[MLflow] MAPE=%.2f%% > seuil — modèle resté en Staging", mape
                )

        except Exception as exc:
            logger.error("[MLflow] Erreur lors de l'enregistrement registry : %s", exc)

    # Écrire le fichier metrics.json lu par l'API FastAPI (/health, /models, /metrics)
    if path:
        metrics_path = Path(path).parent / f"{Path(path).stem}.metrics.json"
        metrics_payload = {
            "r2_score":            test_metrics.get("r2_test"),
            "mape_percent":        test_metrics.get("mape_test"),
            "rmse_mw":             test_metrics.get("rmse_test"),
            "mlflow_run_id":       mlflow_run_id,
            "mlflow_tracking_uri": MLFLOW_TRACKING_URI,
            "registry_version":    registry_version,
            "registry_stage":      registry_stage,
            "registered_model":    REGISTERED_MODEL,
            "validated_at":        datetime.now(UTC).isoformat(),
        }
        metrics_path.write_text(json.dumps(metrics_payload, indent=2))
        logger.info("[API] metrics.json écrit : %s", metrics_path)

    return {
        "model_name":       model_name,
        "saved":            path is not None,
        "path":             str(path) if path else None,
        "mlflow_run_id":    mlflow_run_id,
        "registry_version": registry_version,
        "registry_stage":   registry_stage,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Étapes PREDICT & SNOWFLAKE — inchangées
# ─────────────────────────────────────────────────────────────────────────────

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
            "prediction_mw":   y_pred.round(1),
            "actual_mw":       splits["y_test"].values,
            "model_name":      model_name,
            "pipeline_run_id": run_id or "local",
            "predicted_at":    datetime.now(UTC).isoformat(),
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


# ─────────────────────────────────────────────────────────────────────────────
# Helpers internes — persistance du mlflow_run_id entre steps Airflow
# ─────────────────────────────────────────────────────────────────────────────

def _mlflow_run_id_path(run_id: str | None) -> Path:
    """Chemin du fichier qui stocke le mlflow_run_id pour un dag run donné."""
    return Path(artifact_path("mlflow_run_id.txt", run_id))


def _save_mlflow_run_id(mlflow_run_id: str, run_id: str | None) -> None:
    path = _mlflow_run_id_path(run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(mlflow_run_id)


def _load_mlflow_run_id(run_id: str | None) -> str | None:
    path = _mlflow_run_id_path(run_id)
    if path.exists():
        return path.read_text().strip()
    logger.warning("[MLflow] mlflow_run_id.txt introuvable : %s", path)
    return None
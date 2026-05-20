"""
Transform — nettoyage, agrégation journalière et feature engineering.
"""

import logging

import numpy as np
import pandas as pd

from preprocessing.constants import (
    BENCHMARK_COL,
    JOURS_FERIES_FR,
    NUMERIC_FEATURES,
    RENAME_MAP,
    TARGET,
)

logger = logging.getLogger(__name__)


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Renommage, datetime, interpolation, filtrage des valeurs aberrantes."""
    logger.info("[ETL - Transform] Étape 1/3 : nettoyage des données brutes")
    df = df.copy()

    existing = {k: v for k, v in RENAME_MAP.items() if k in df.columns}
    df = df.rename(columns=existing)

    if TARGET not in df.columns:
        raise ValueError(
            f"Colonne « {TARGET} » absente. Colonnes lues : {list(df.columns)[:8]}…"
        )

    if "date" in df.columns and "heures" in df.columns:
        df["datetime"] = pd.to_datetime(
            df["date"].astype(str) + " " + df["heures"].astype(str),
            format="%Y-%m-%d %H:%M",
            errors="coerce",
        )
        df = df.dropna(subset=["datetime"])
        df = df.sort_values("datetime").reset_index(drop=True)
        df = df[df["datetime"].dt.minute.isin([0, 30])].reset_index(drop=True)

    cols_num = list(dict.fromkeys(
        c for c in [TARGET, BENCHMARK_COL, *NUMERIC_FEATURES] if c in df.columns
    ))
    for col in cols_num:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df[cols_num] = df[cols_num].interpolate(method="linear", limit_direction="both")
    df = df.dropna(subset=[TARGET])
    df = df[df[TARGET] > 0].reset_index(drop=True)

    logger.info("[ETL - Transform] Nettoyage terminé : %s lignes conservées", f"{len(df):,}")
    return df


def _aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Agrège les mesures demi-horaires en journalier."""
    logger.info("[ETL - Transform] Étape 2/3 : agrégation journalière")
    if "datetime" not in df.columns:
        raise ValueError("Colonne 'datetime' absente : exécutez le nettoyage d'abord.")

    df = df.copy()
    df["date_only"] = df["datetime"].dt.date

    grouped = df.groupby("date_only")
    consumption = grouped[TARGET].agg(["mean", "max", "min"]).rename(
        columns={"mean": TARGET, "max": f"{TARGET}_max", "min": f"{TARGET}_min"}
    )

    feature_cols = [c for c in NUMERIC_FEATURES if c in df.columns]
    daily = (
        consumption.join(grouped[feature_cols].mean())
        if feature_cols
        else consumption
    )
    daily.index = pd.to_datetime(daily.index)
    daily = daily.sort_index()

    logger.info("[ETL - Transform] Agrégation terminée : %d jours", len(daily))
    return daily


def _add_features(daily: pd.DataFrame) -> pd.DataFrame:
    """Calendrier, encodage cyclique et lags sur le DataFrame journalier."""
    logger.info("[ETL - Transform] Étape 3/3 : création des variables (features)")
    df = daily.copy()
    dt = pd.Series(pd.DatetimeIndex(df.index), index=df.index).dt

    df["day_of_week"] = dt.dayofweek
    df["month"] = dt.month
    df["year"] = dt.year
    df["day_of_year"] = dt.dayofyear
    df["is_weekend"] = (dt.dayofweek >= 5).astype(int)
    df["is_holiday"] = dt.strftime("%Y-%m-%d").isin(JOURS_FERIES_FR).astype(int)

    def saison(m: int) -> int:
        if m in (12, 1, 2):
            return 0
        if m in (3, 4, 5):
            return 1
        if m in (6, 7, 8):
            return 2
        return 3

    df["saison"] = dt.month.map(saison)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["lag_1"] = df[TARGET].shift(1)
    df["lag_7"] = df[TARGET].shift(7)
    df["prevision_j1_lag1"] = (
        df[BENCHMARK_COL].shift(1) if BENCHMARK_COL in df.columns else np.nan
    )

    df = df.dropna()
    logger.info(
        "[ETL - Transform] Features terminées : %d jours, %d colonnes",
        len(df),
        df.shape[1],
    )
    return df


def transform(raw: pd.DataFrame) -> pd.DataFrame:
    """Enchaîne nettoyage → agrégation → feature engineering."""
    logger.info("[ETL - Transform] Début de la transformation (%s lignes en entrée)", f"{len(raw):,}")
    cleaned = _clean(raw)
    daily = _aggregate_daily(cleaned)
    featured = _add_features(daily)
    logger.info("[ETL - Transform] Transformation terminée")
    return featured

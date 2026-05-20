"""
Preprocessing des données RTE éCO2mix pour la prédiction de consommation électrique.
MSPR Bloc 3 – EDF | Arthur Méry, Martin Dié, Imed Eddine Zeroual
"""

from collections.abc import Sequence

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Chargement brut                                                              #
# --------------------------------------------------------------------------- #

def load_rte_file(filepath: str | Path) -> pd.DataFrame:
    """
    Charge un fichier RTE éCO2mix (.xls / .tsv, séparateur tabulation).
    Retourne un DataFrame brut avec les colonnes d'origine.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Fichier introuvable : {filepath}")

    logger.info(f"Chargement du fichier : {filepath.name}")
    df = pd.read_csv(
        filepath,
        sep="\t",
        encoding="latin-1",
        low_memory=False,
        index_col=False,
    )
    logger.info(f"  {len(df):,} lignes chargées, {df.shape[1]} colonnes")
    return df


def _is_eco2mix_file(df: pd.DataFrame) -> bool:
    cols = set(df.columns)
    return "Consommation" in cols or "Consommation (MW)" in cols


def load_and_merge(paths: Sequence[str | Path]) -> pd.DataFrame:
    """Charge et concatène plusieurs fichiers éCO2mix (ignore les autres exports RTE)."""
    frames = []
    for path in paths:
        df = load_rte_file(path)
        if not _is_eco2mix_file(df):
            logger.warning("Fichier ignoré (pas éCO2mix) : %s", Path(path).name)
            continue
        frames.append(df)

    if not frames:
        raise ValueError(
            "Aucun fichier éCO2mix valide. "
            "Ex. data/eCO2mix_RTE_Annuel-Definitif_2023.xls"
        )

    merged = pd.concat(frames, ignore_index=True)
    logger.info(f"Dataset fusionné : {len(merged):,} lignes")
    return merged


# --------------------------------------------------------------------------- #
# Nettoyage                                                                    #
# --------------------------------------------------------------------------- #

RENAME_MAP = {
    # Export RTE actuel (colonnes sans suffixe MW)
    "Consommation": "consommation",
    "Prévision J-1": "prevision_j1",
    "Nucléaire": "nucleaire",
    "Eolien": "eolien",
    "Solaire": "solaire",
    "Hydraulique": "hydraulique",
    "Gaz": "gaz",
    "Taux de Co2": "co2",
    # Ancien format avec suffixe (MW)
    "Consommation (MW)": "consommation",
    "Prévision J-1 (MW)": "prevision_j1",
    "Nucléaire (MW)": "nucleaire",
    "Eolien (MW)": "eolien",
    "Solaire (MW)": "solaire",
    "Hydraulique (MW)": "hydraulique",
    "Gaz (MW)": "gaz",
    "Taux de CO2 (g/kWh)": "co2",
    "Date": "date",
    "Heures": "heures",
}

TARGET = "consommation"
BENCHMARK_COL = "prevision_j1"
NUMERIC_FEATURES = ["prevision_j1", "nucleaire", "eolien", "solaire", "hydraulique", "gaz", "co2"]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    1. Renomme les colonnes.
    2. Parse la date + heure en datetime.
    3. Supprime les lignes au pas de 15 min (garder 30 min uniquement).
    4. Remplace les valeurs manquantes par interpolation linéaire.
    5. Supprime les valeurs aberrantes (consommation < 0).
    """
    df = df.copy()

    # ---- renommage ----
    existing = {k: v for k, v in RENAME_MAP.items() if k in df.columns}
    df = df.rename(columns=existing)

    if TARGET not in df.columns:
        raise ValueError(
            f"Colonne « {TARGET} » absente. Colonnes lues : {list(df.columns)[:8]}…"
        )

    # ---- datetime ----
    if "date" in df.columns and "heures" in df.columns:
        df["datetime"] = pd.to_datetime(
            df["date"].astype(str) + " " + df["heures"].astype(str),
            format="%Y-%m-%d %H:%M",
            errors="coerce",
        )
        df = df.dropna(subset=["datetime"])
        df = df.sort_values("datetime").reset_index(drop=True)

        # garder uniquement le pas 30 min (minutes == 0 ou 30)
        df = df[df["datetime"].dt.minute.isin([0, 30])].reset_index(drop=True)

    # ---- numériques ----
    cols_num = list(dict.fromkeys(
        c for c in [TARGET, BENCHMARK_COL, *NUMERIC_FEATURES] if c in df.columns
    ))
    for col in cols_num:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # interpolation des NaN
    df[cols_num] = df[cols_num].interpolate(method="linear", limit_direction="both")
    df = df.dropna(subset=[TARGET])

    # valeurs aberrantes
    df = df[df[TARGET] > 0].reset_index(drop=True)

    logger.info(f"Après nettoyage : {len(df):,} lignes")
    return df


# --------------------------------------------------------------------------- #
# Agrégation journalière                                                       #
# --------------------------------------------------------------------------- #

def aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrège les mesures demi-horaires en journalier.
    Produit : moyenne, max, min de la consommation + moyennes des features.
    """
    if "datetime" not in df.columns:
        raise ValueError("La colonne 'datetime' est absente. Exécutez clean() d'abord.")

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

    logger.info(f"Agrégation journalière : {len(daily)} jours")
    return daily


# --------------------------------------------------------------------------- #
# Feature engineering                                                          #
# --------------------------------------------------------------------------- #

JOURS_FERIES_FR = {
    "2023-01-01", "2023-04-10", "2023-05-01", "2023-05-08", "2023-05-18",
    "2023-05-29", "2023-07-14", "2023-08-15", "2023-11-01", "2023-11-11", "2023-12-25",
    "2024-01-01", "2024-04-01", "2024-05-01", "2024-05-08", "2024-05-09",
    "2024-05-20", "2024-07-14", "2024-08-15", "2024-11-01", "2024-11-11", "2024-12-25",
}


def add_features(daily: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering sur le DataFrame journalier indexé par date."""
    df = daily.copy()
    dt = pd.Series(pd.DatetimeIndex(df.index), index=df.index).dt

    df["day_of_week"] = dt.dayofweek
    df["month"] = dt.month
    df["year"] = dt.year
    df["day_of_year"] = dt.dayofyear
    df["is_weekend"] = (dt.dayofweek >= 5).astype(int)
    df["is_holiday"] = dt.strftime("%Y-%m-%d").isin(JOURS_FERIES_FR).astype(int)

    # saison : 0=hiver, 1=printemps, 2=été, 3=automne
    def saison(m: int) -> int:
        if m in (12, 1, 2): return 0
        if m in (3, 4, 5):  return 1
        if m in (6, 7, 8):  return 2
        return 3
    df["saison"] = dt.month.map(saison)

    # encodage cyclique mois et jour de la semaine
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    # lags (J-1 et J-7)
    df["lag_1"] = df[TARGET].shift(1)
    df["lag_7"] = df[TARGET].shift(7)
    df["prevision_j1_lag1"] = df[BENCHMARK_COL].shift(1) if BENCHMARK_COL in df.columns else np.nan

    # suppression des NaN engendrés par les lags
    df = df.dropna()
    logger.info(f"Après feature engineering : {len(df)} jours, {df.shape[1]} colonnes")
    return df


# --------------------------------------------------------------------------- #
# Découpage train / validation / test                                          #
# --------------------------------------------------------------------------- #

FEATURE_COLS = [
    "prevision_j1", "nucleaire", "eolien", "solaire", "hydraulique", "gaz", "co2",
    "consommation_max", "consommation_min",
    "day_of_week", "month", "day_of_year", "is_weekend", "is_holiday", "saison",
    "month_sin", "month_cos", "dow_sin", "dow_cos",
    "lag_1", "lag_7", "prevision_j1_lag1",
]


def split_dataset(df: pd.DataFrame):
    """
    Découpage temporel :
      train      : janv. 2023 → juin 2024
      validation : juil. 2024 → sept. 2024
      test       : oct. 2024  → déc. 2024
    """
    feats = [c for c in FEATURE_COLS if c in df.columns]

    if df.index.max().year <= 2023:
        train_end, val_end = "2023-07-01", "2023-10-01"
    else:
        train_end, val_end = "2024-07-01", "2024-10-01"

    train = df[df.index < train_end]
    val   = df[(df.index >= train_end) & (df.index < val_end)]
    test  = df[df.index >= val_end]

    if len(train) == 0 or len(test) == 0:
        raise ValueError(
            f"Split invalide (train={len(train)}, val={len(val)}, test={len(test)}). "
            "Vérifiez la période couverte par les fichiers."
        )

    logger.info(f"Train: {len(train)} jours | Val: {len(val)} jours | Test: {len(test)} jours")

    x_train, y_train = train[feats], train[TARGET]
    x_val, y_val = val[feats], val[TARGET]
    x_test, y_test = test[feats], test[TARGET]

    return x_train, x_val, x_test, y_train, y_val, y_test, feats


# --------------------------------------------------------------------------- #
# Pipeline complet                                                             #
# --------------------------------------------------------------------------- #

def run_pipeline(paths: Sequence[str | Path]):
    """Enchaîne toutes les étapes et retourne les splits prêts à l'emploi."""
    raw   = load_and_merge(paths)
    clean_df = clean(raw)
    daily = aggregate_daily(clean_df)
    featured = add_features(daily)
    return split_dataset(featured)
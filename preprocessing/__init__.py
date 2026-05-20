"""
Pipeline ETL des données RTE éCO2mix — prédiction de consommation électrique.
MSPR Bloc 3 – EDF | Arthur Méry, Martin Dié, Imed Eddine Zeroual

  extract.py   → lecture des fichiers bruts
  transform.py → nettoyage, agrégation, features
  load.py      → splits train / val / test
"""

import logging

from preprocessing.constants import (
    BENCHMARK_COL,
    FEATURE_COLS,
    NUMERIC_FEATURES,
    RENAME_MAP,
    TARGET,
)
from preprocessing.extract import extract, load_rte_file
from preprocessing.load import load
from preprocessing.pipeline import run_etl, run_pipeline
from preprocessing.transform import transform

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

__all__ = [
    "TARGET",
    "BENCHMARK_COL",
    "RENAME_MAP",
    "NUMERIC_FEATURES",
    "FEATURE_COLS",
    "extract",
    "transform",
    "load",
    "load_rte_file",
    "run_etl",
    "run_pipeline",
]

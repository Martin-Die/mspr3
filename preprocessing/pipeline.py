"""
Orchestration ETL : Extract → Transform → Load.
"""

import logging
from collections.abc import Sequence
from pathlib import Path

from preprocessing.extract import extract
from preprocessing.load import load
from preprocessing.transform import transform

logger = logging.getLogger(__name__)


def run_etl(paths: Sequence[str | Path]):
    """Exécute la pipeline ETL complète."""
    path_list = [Path(p) for p in paths]
    logger.info("[ETL] Démarrage de la pipeline (%d fichier(s))", len(path_list))
    for i, p in enumerate(path_list, start=1):
        logger.info("[ETL] Fichier %d/%d : %s", i, len(path_list), p.name)

    raw = extract(paths)
    featured = transform(raw)
    result = load(featured)
    logger.info("[ETL] Pipeline terminée avec succès")
    return result


def run_pipeline(paths: Sequence[str | Path]):
    """Alias conservé pour compatibilité avec models.py."""
    return run_etl(paths)

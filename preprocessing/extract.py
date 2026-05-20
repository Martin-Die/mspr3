"""
Extract — lecture des fichiers RTE éCO2mix bruts.
"""

from collections.abc import Sequence
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def load_rte_file(filepath: str | Path) -> pd.DataFrame:
    """Charge un fichier RTE éCO2mix (.xls / .tsv, séparateur tabulation)."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Fichier introuvable : {filepath}")

    logger.info("[ETL - Extract] Chargement du fichier : %s", filepath.name)
    df = pd.read_csv(
        filepath,
        sep="\t",
        encoding="latin-1",
        low_memory=False,
        index_col=False,
    )
    logger.info("  %s lignes, %s colonnes", f"{len(df):,}", df.shape[1])
    return df


def _is_eco2mix_file(df: pd.DataFrame) -> bool:
    cols = set(df.columns)
    return "Consommation" in cols or "Consommation (MW)" in cols


def extract(paths: Sequence[str | Path]) -> pd.DataFrame:
    """
    Extrait et fusionne les fichiers éCO2mix.
    Ignore les exports RTE qui ne sont pas au format éCO2mix.
    """
    logger.info("[ETL - Extract] Début de l'extraction (%d fichier(s))", len(paths))
    frames = []
    for path in paths:
        df = load_rte_file(path)
        if not _is_eco2mix_file(df):
            logger.warning(
                "[ETL - Extract] Fichier ignoré (format non éCO2mix) : %s", Path(path).name
            )
            continue
        frames.append(df)

    if not frames:
        raise ValueError(
            "Aucun fichier éCO2mix valide. "
            "Ex. data/eCO2mix_RTE_Annuel-Definitif_2023.xls"
        )

    merged = pd.concat(frames, ignore_index=True)
    logger.info(
        "[ETL - Extract] Terminé : %d fichier(s) retenu(s), %s lignes fusionnées",
        len(frames),
        f"{len(merged):,}",
    )
    return merged

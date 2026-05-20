"""
Load — découpage temporel et préparation des jeux train / val / test.
"""

import logging

import pandas as pd

from preprocessing.constants import FEATURE_COLS, TARGET

logger = logging.getLogger(__name__)


def load(featured: pd.DataFrame):
    """
    Charge le dataset transformé en splits prêts pour l'entraînement.

    Découpage temporel :
      train      : janv. 2023 → juin 2024 (ou équivalent 2023 seul)
      validation : juil. → sept.
      test       : oct. → déc.
    """
    logger.info("[ETL - Load] Début du découpage train / validation / test")
    feats = [c for c in FEATURE_COLS if c in featured.columns]
    logger.info("[ETL - Load] %d variables utilisées pour l'entraînement", len(feats))

    if featured.index.max().year <= 2023:
        train_end, val_end = "2023-07-01", "2023-10-01"
    else:
        train_end, val_end = "2024-07-01", "2024-10-01"

    train = featured[featured.index < train_end]
    val = featured[(featured.index >= train_end) & (featured.index < val_end)]
    test = featured[featured.index >= val_end]

    if len(train) == 0 or len(test) == 0:
        raise ValueError(
            f"Split invalide (train={len(train)}, val={len(val)}, test={len(test)}). "
            "Vérifiez la période couverte par les fichiers."
        )

    logger.info(
        "[ETL - Load] Découpage terminé : train=%d jours, validation=%d jours, test=%d jours",
        len(train),
        len(val),
        len(test),
    )
    logger.info(
        "[ETL - Load] Bornes temporelles : train < %s, validation < %s, test >= %s",
        train_end,
        val_end,
        val_end,
    )

    x_train, y_train = train[feats], train[TARGET]
    x_val, y_val = val[feats], val[TARGET]
    x_test, y_test = test[feats], test[TARGET]

    return x_train, x_val, x_test, y_train, y_val, y_test, feats

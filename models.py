"""
Entraînement et évaluation des 4 modèles de prédiction.
MSPR Bloc 3 – EDF | Arthur Méry, Martin Dié, Imed Eddine Zeroual

Modèles :
  - Arbre de décision  (DecisionTreeRegressor)
  - Forêt aléatoire    (RandomForestRegressor)
  - KNN                (KNeighborsRegressor)
  - Réseau de neurones (MLPRegressor, architecture RBF-like)
"""

import json
import time
import logging
import joblib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error

logger = logging.getLogger(__name__)

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

# --------------------------------------------------------------------------- #
# Définition des modèles                                                       #
# --------------------------------------------------------------------------- #

def build_models() -> dict:
    """
    Retourne un dictionnaire nom → pipeline scikit-learn.
    Chaque pipeline intègre un StandardScaler (obligatoire pour KNN et MLP).
    """
    return {
        "decision_tree": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", DecisionTreeRegressor(
                    max_depth=8,
                    min_samples_leaf=5,
                    ccp_alpha=0.0,
                    random_state=42,
                )),
            ],
            memory=None,
        ),
        "random_forest": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", RandomForestRegressor(
                    n_estimators=200,
                    max_depth=12,
                    max_features=1.0,
                    min_samples_leaf=3,
                    ccp_alpha=0.0,
                    n_jobs=-1,
                    random_state=42,
                )),
            ],
            memory=None,
        ),
        "knn": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", KNeighborsRegressor(
                    n_neighbors=7,
                    weights="distance",
                    metric="minkowski",
                    n_jobs=-1,
                )),
            ],
            memory=None,
        ),
        # MLPRegressor avec architecture à couches cachées (RBF-like via hidden layers)
        "neural_network_rbf": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", MLPRegressor(
                    hidden_layer_sizes=(128, 64, 32),
                    activation="relu",
                    solver="adam",
                    learning_rate_init=0.001,
                    max_iter=500,
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=20,
                    random_state=42,
                )),
            ],
            memory=None,
        ),
    }


# --------------------------------------------------------------------------- #
# Métriques                                                                    #
# --------------------------------------------------------------------------- #

def _to_ndarray(values: pd.Series | np.ndarray) -> np.ndarray:
    """Convertit Series / ndarray pandas en ndarray numpy pour les métriques."""
    if isinstance(values, pd.Series):
        return values.to_numpy(dtype=np.float64)
    return np.ascontiguousarray(values, dtype=np.float64)


def compute_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    label: str = "",
) -> dict:
    """Calcule R², RMSE, MAPE pour un vecteur de prédictions."""
    y_true_arr = _to_ndarray(y_true)
    y_pred_arr = _to_ndarray(y_pred)
    r2   = r2_score(y_true_arr, y_pred_arr)
    rmse = np.sqrt(mean_squared_error(y_true_arr, y_pred_arr))
    mape = mean_absolute_percentage_error(y_true_arr, y_pred_arr) * 100  # en %

    logger.info(
        "[Évaluation] %s : R²=%.4f, RMSE=%s MW, MAPE=%.2f %%",
        label,
        r2,
        f"{rmse:,.0f}",
        mape,
    )
    return {"r2": r2, "rmse": rmse, "mape": mape}


def benchmark_rte(y_true: pd.Series, y_prevision_j1: pd.Series) -> dict:
    """Évalue la prévision RTE J-1 comme baseline."""
    mask = y_prevision_j1.notna()
    metrics = compute_metrics(y_true[mask], y_prevision_j1[mask], "RTE_J-1_benchmark")
    return metrics


# --------------------------------------------------------------------------- #
# Entraînement                                                                 #
# --------------------------------------------------------------------------- #

def train_model(name: str, pipeline: Pipeline, x_train, y_train) -> tuple[Pipeline, float]:
    """Entraîne un pipeline et retourne (pipeline_entraîné, durée_sec)."""
    logger.info("[Entraînement] Modèle en cours : %s", name)
    t0 = time.perf_counter()
    pipeline.fit(x_train, y_train)
    elapsed = time.perf_counter() - t0
    logger.info("[Entraînement] Modèle %s terminé en %.2f s", name, elapsed)
    return pipeline, elapsed


def train_all(x_train, y_train) -> dict:
    """Entraîne les 4 modèles et retourne un dict nom → (pipeline, durée)."""
    models = build_models()
    logger.info(
        "[Entraînement] Début : %d modèle(s), %d échantillon(s)",
        len(models),
        len(x_train),
    )
    trained = {}
    for i, (name, pipe) in enumerate(models.items(), start=1):
        logger.info("[Entraînement] Progression : %d/%d", i, len(models))
        pipe, elapsed = train_model(name, pipe, x_train, y_train)
        trained[name] = {"pipeline": pipe, "train_time": elapsed}
    logger.info("[Entraînement] Tous les modèles ont été entraînés")
    return trained


# --------------------------------------------------------------------------- #
# Évaluation et comparaison                                                    #
# --------------------------------------------------------------------------- #

def evaluate_all(trained: dict, x_val, y_val, x_test, y_test) -> pd.DataFrame:
    """
    Évalue chaque modèle sur val + test.
    Retourne un DataFrame de comparaison trié par R² test décroissant.
    """
    logger.info(
        "[Évaluation] Début : %d modèle(s), validation=%d, test=%d échantillon(s)",
        len(trained),
        len(x_val),
        len(x_test),
    )
    rows = []
    for i, (name, info) in enumerate(trained.items(), start=1):
        logger.info("[Évaluation] Modèle %d/%d : %s", i, len(trained), name)
        pipe = info["pipeline"]

        y_pred_val  = pipe.predict(x_val)
        y_pred_test = pipe.predict(x_test)

        m_val  = compute_metrics(y_val, y_pred_val, f"{name}/val")
        m_test = compute_metrics(y_test, y_pred_test, f"{name}/test")

        rows.append({
            "model":          name,
            "r2_val":         round(m_val["r2"],   4),
            "rmse_val":       round(m_val["rmse"],  0),
            "mape_val":       round(m_val["mape"],  2),
            "r2_test":        round(m_test["r2"],   4),
            "rmse_test":      round(m_test["rmse"], 0),
            "mape_test":      round(m_test["mape"], 2),
            "train_time_s":   round(info["train_time"], 2),
        })

    df = pd.DataFrame(rows).sort_values("r2_test", ascending=False).reset_index(drop=True)
    best = df.iloc[0]
    logger.info(
        "[Évaluation] Terminée. Meilleur modèle : %s (R² test=%.4f)",
        best["model"],
        best["r2_test"],
    )
    return df


# --------------------------------------------------------------------------- #
# Sélection du meilleur modèle vs benchmark                                   #
# --------------------------------------------------------------------------- #

R2_THRESHOLD   = 0.85
MAPE_MIN_DELTA = 0.10   # 10 % d'amélioration relative sur MAPE
R2_MIN_DELTA   = 0.05


def select_best_model(results: pd.DataFrame, benchmark: dict) -> str | None:
    """
    Retourne le nom du meilleur modèle ML si au moins 2 métriques battent le benchmark.
    Sinon retourne None (utiliser le benchmark RTE J-1 comme baseline).
    """
    bm_mape = benchmark.get("mape", np.inf)
    bm_r2   = benchmark.get("r2",   -np.inf)

    best = results.iloc[0]  # meilleur R² test

    beats_mape = best["mape_test"] < bm_mape * (1 - MAPE_MIN_DELTA)
    beats_r2   = best["r2_test"]   > bm_r2   + R2_MIN_DELTA
    beats_rmse = best["rmse_test"] < benchmark.get("rmse", np.inf)

    wins = sum([beats_mape, beats_r2, beats_rmse])
    model_name = best["model"]

    logger.info("[Sélection] Comparaison au benchmark RTE J-1")
    if wins >= 2:
        logger.info(
            "[Sélection] Modèle retenu : %s (R²=%.4f, MAPE=%.2f %%, bat le benchmark sur %d/3 métriques)",
            model_name,
            best["r2_test"],
            best["mape_test"],
            wins,
        )
        return model_name
    else:
        logger.warning(
            "[Sélection] Aucun modèle ne bat le benchmark sur au moins 2 métriques. "
            "Baseline RTE J-1 recommandée."
        )
        return None


# --------------------------------------------------------------------------- #
# Sauvegarde / chargement                                                      #
# --------------------------------------------------------------------------- #

def _model_paths(name: str, version: str = "latest") -> tuple[Path, Path]:
    stem = f"{name}_{version}"
    return MODELS_DIR / f"{stem}.joblib", MODELS_DIR / f"{stem}.metrics.json"


def _is_better_than_saved(new_metrics: dict, old_metrics: dict) -> bool:
    """True si le nouveau modèle est strictement meilleur sur le jeu test."""
    if new_metrics["r2_test"] > old_metrics["r2_test"]:
        return True
    if new_metrics["r2_test"] < old_metrics["r2_test"]:
        return False
    if new_metrics["rmse_test"] < old_metrics["rmse_test"]:
        return True
    if new_metrics["rmse_test"] > old_metrics["rmse_test"]:
        return False
    return new_metrics["mape_test"] < old_metrics["mape_test"]


def save_model_if_better(
    pipeline: Pipeline,
    name: str,
    test_metrics: dict,
    version: str = "latest",
) -> Path | None:
    """
    Sérialise le pipeline uniquement s'il n'existe pas encore
    ou s'il bat le modèle déjà enregistré (métriques test).
    """
    model_path, meta_path = _model_paths(name, version)

    if model_path.exists() and meta_path.exists():
        old_metrics = json.loads(meta_path.read_text(encoding="utf-8"))
        if not _is_better_than_saved(test_metrics, old_metrics):
            logger.info(
                "[Sauvegarde] Modèle %s conservé (R² test actuel %.4f >= nouveau %.4f)",
                name,
                old_metrics["r2_test"],
                test_metrics["r2_test"],
            )
            return None

    logger.info("[Sauvegarde] Enregistrement du modèle %s", name)
    joblib.dump(pipeline, model_path)
    meta_path.write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")
    logger.info("[Sauvegarde] Modèle enregistré : %s", model_path)
    return model_path


def save_model(pipeline: Pipeline, name: str, version: str = "latest") -> Path:
    """Sérialise le pipeline (écrase l'existant)."""
    model_path, _ = _model_paths(name, version)
    joblib.dump(pipeline, model_path)
    logger.info("[Sauvegarde] Modèle enregistré : %s", model_path)
    return model_path


def load_model(name: str, version: str = "latest") -> Pipeline:
    """Charge un pipeline sérialisé."""
    path, _ = _model_paths(name, version)
    if not path.exists():
        raise FileNotFoundError(f"Modèle introuvable : {path}")
    logger.info("[Chargement] Lecture du modèle %s", name)
    pipeline = joblib.load(path)
    logger.info("[Chargement] Modèle chargé depuis %s", path)
    return pipeline


# --------------------------------------------------------------------------- #
# Script de lancement                                                          #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    from preprocessing import run_pipeline

    data_paths = sys.argv[1:] if len(sys.argv) > 1 else []
    if not data_paths:
        logger.error("Usage : python models.py <fichier_2023.xls> <fichier_2024.xls>")
        sys.exit(1)

    logger.info("[Pipeline] Étape 1/4 : préparation des données (ETL)")
    x_train, x_val, x_test, y_train, y_val, y_test, feats = run_pipeline(data_paths)
    logger.info("[Pipeline] Données prêtes : %d variable(s)", len(feats))

    logger.info("[Pipeline] Étape 2/4 : entraînement des modèles")
    trained = train_all(x_train, y_train)

    logger.info("[Pipeline] Étape 3/4 : évaluation et comparaison")
    results = evaluate_all(trained, x_val, y_val, x_test, y_test)
    logger.info("[Pipeline] Résultats :\n%s", results.to_string(index=False))

    best_row = results.iloc[0]
    best_name = str(best_row["model"])
    test_metrics = {
        "r2_test": float(best_row["r2_test"]),
        "rmse_test": float(best_row["rmse_test"]),
        "mape_test": float(best_row["mape_test"]),
    }
    logger.info("[Pipeline] Étape 4/4 : sauvegarde du meilleur modèle (%s)", best_name)
    save_model_if_better(trained[best_name]["pipeline"], best_name, test_metrics)
    logger.info("[Pipeline] Exécution terminée")
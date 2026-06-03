"""
API REST FastAPI — Prédiction de la consommation électrique EDF.
MSPR Bloc 3 | Arthur Méry, Martin Dié, Imed Eddine Zeroual

Endpoints :
  POST /predict      → prédiction de consommation journalière
  GET  /health       → état de santé du service
  GET  /metrics      → métriques Prometheus (scraping)
  GET  /models       → liste des modèles disponibles
  GET  /models/info  → détails du modèle actif (métriques, hyperparamètres, date)
"""

from __future__ import annotations

import os
import time
import logging
from datetime import UTC, date as DateType, datetime
from pathlib import Path
from typing import Optional

import json
import joblib
import numpy as np
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field, field_validator

# --------------------------------------------------------------------------- #
# Config                                                                       #
# --------------------------------------------------------------------------- #

logging.basicConfig(
    level=logging.INFO,
    format='{"time": "%(asctime)s", "level": "%(levelname)s", "msg": "%(message)s"}',
)
logger = logging.getLogger(__name__)

MODELS_DIR    = Path(os.getenv("MODELS_DIR", "models"))
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "random_forest_latest")
APP_VERSION   = "1.0.0"
RATE_LIMIT    = int(os.getenv("RATE_LIMIT_PER_MIN", "100"))
# Si API_KEY est défini dans l'environnement, l'authentification est activée.
_API_KEY      = os.getenv("API_KEY", "")
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

from preprocessing.constants import FEATURE_COLS, JOURS_FERIES_FR

# --------------------------------------------------------------------------- #
# Cache modèles                                                                #
# --------------------------------------------------------------------------- #

_model_cache: dict = {}
_request_counts: dict = {}   # IP → (count, window_start)


def get_model(name: str):
    """Charge et met en cache un modèle sérialisé."""
    if name not in _model_cache:
        path = MODELS_DIR / f"{name}.joblib"
        if not path.exists():
            raise FileNotFoundError(f"Modèle {name} introuvable dans {MODELS_DIR}")
        logger.info("[API] Chargement du modèle : %s", name)
        _model_cache[name] = {
            "pipeline": joblib.load(path),
            "loaded_at": datetime.now(UTC).isoformat(),
            "path": str(path),
        }
        logger.info("[API] Modèle %s chargé et mis en cache", name)
    return _model_cache[name]["pipeline"]


def _load_metrics(name: str) -> dict:
    """Charge le fichier .metrics.json associé à un modèle, si présent."""
    path = MODELS_DIR / f"{name}.metrics.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _check_api_key(key: str | None) -> None:
    """Lève 401 si l'authentification par clé API est activée et que la clé est invalide."""
    if _API_KEY and key != _API_KEY:
        logger.warning("[API] Tentative d'accès avec une clé invalide")
        raise HTTPException(status_code=401, detail="Clé API manquante ou invalide.")


# --------------------------------------------------------------------------- #
# App                                                                          #
# --------------------------------------------------------------------------- #

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("[API] Application démarrée (version %s)", APP_VERSION)
    logger.info("[API] Modèle par défaut : %s", DEFAULT_MODEL)
    logger.info("[API] Répertoire des modèles : %s", MODELS_DIR.resolve())
    yield
    logger.info("[API] Arrêt de l'application")


app = FastAPI(
    title="EDF — API de prédiction de la consommation électrique",
    description=(
        "Solution IA de prédiction journalière basée sur les données RTE éCO2mix. "
        "MSPR Bloc 3 — EPSI RNCP36582."
    ),
    version=APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------------------------------- #
# Compteurs Prometheus (simples, sans dépendance externe)                      #
# --------------------------------------------------------------------------- #

class _Counters:
    requests_total    = 0
    predict_ok        = 0
    predict_errors    = 0
    latencies: list[float] = []

_cnt = _Counters()


# --------------------------------------------------------------------------- #
# Schémas Pydantic                                                             #
# --------------------------------------------------------------------------- #

class PredictRequest(BaseModel):
    """Corps de la requête POST /predict (aligné sur le modèle entraîné par Airflow)."""
    date: DateType = Field(..., description="Date cible (YYYY-MM-DD)")
    prevision_j1: Optional[float] = Field(None, description="Prévision RTE J-1 en MW (optionnel)")
    lag_1: Optional[float] = Field(None, description="Consommation J-1 réelle en MW")
    lag_7: Optional[float] = Field(None, description="Consommation J-7 réelle en MW")
    model_name: str = Field(DEFAULT_MODEL, description="Nom du modèle à utiliser")

    @field_validator("date")
    @classmethod
    def date_not_past_limit(cls, v):
        if v.year < 2020:
            raise ValueError("La date doit être postérieure à 2020.")
        return v


class PredictResponse(BaseModel):
    date:                str
    prediction_mw:       float
    model_name:          str
    model_version:       str
    r2_score:            Optional[float] = None
    rmse_mw:             Optional[float] = None
    mape_percent:        Optional[float] = None
    prevision_rte_j1_mw: Optional[float] = None
    latency_ms:          float
    timestamp:           str


class HealthResponse(BaseModel):
    status:        str
    version:       str
    model_loaded:  bool
    model_name:    str
    loaded_at:     Optional[str] = None
    uptime_s:      float


_start_time = time.time()

# --------------------------------------------------------------------------- #
# Rate limiting middleware                                                     #
# --------------------------------------------------------------------------- #

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    ip = request.client.host if request.client else "unknown"
    now = time.time()
    window = _request_counts.get(ip, (0, now))
    count, window_start = window

    if now - window_start > 60:
        _request_counts[ip] = (1, now)
    else:
        if count >= RATE_LIMIT:
            logger.warning(
                "[API] Limite de débit atteinte pour %s (%d requêtes/min)",
                ip,
                RATE_LIMIT,
            )
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=429,
                content={"detail": f"Trop de requêtes. Limite : {RATE_LIMIT}/min."},
            )
        _request_counts[ip] = (count + 1, window_start)

    return await call_next(request)


# --------------------------------------------------------------------------- #
# Helpers feature engineering                                                  #
# --------------------------------------------------------------------------- #

def _saison(month: int) -> int:
    if month in (12, 1, 2): return 0
    if month in (3, 4, 5):  return 1
    if month in (6, 7, 8):  return 2
    return 3


def _num(value: Optional[float], default: float) -> float:
    """None -> défaut ; 0 est une valeur valide (ne pas confondre avec absent)."""
    return default if value is None else float(value)


def _build_feature_vector(req: PredictRequest) -> np.ndarray:
    """Construit le vecteur de features (même ordre que l'entraînement Airflow)."""
    d = req.date
    is_holiday = int(d.strftime("%Y-%m-%d") in JOURS_FERIES_FR)
    is_weekend = int(d.weekday() >= 5)

    prevision = _num(req.prevision_j1, 50_000)
    vals = {
        "prevision_j1": prevision,
        "day_of_week": d.weekday(),
        "month": d.month,
        "day_of_year": d.timetuple().tm_yday,
        "is_weekend": is_weekend,
        "is_holiday": is_holiday,
        "saison": _saison(d.month),
        "month_sin": np.sin(2 * np.pi * d.month / 12),
        "month_cos": np.cos(2 * np.pi * d.month / 12),
        "dow_sin": np.sin(2 * np.pi * d.weekday() / 7),
        "dow_cos": np.cos(2 * np.pi * d.weekday() / 7),
        "lag_1": _num(req.lag_1, 50_000),
        "lag_7": _num(req.lag_7, 50_000),
        "prevision_j1_lag1": prevision,
    }

    return np.array([[vals[c] for c in FEATURE_COLS]])


# --------------------------------------------------------------------------- #
# Endpoints                                                                    #
# --------------------------------------------------------------------------- #

@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health():
    """Vérification de l'état du service (healthcheck)."""
    logger.info("[API] Requête GET /health")
    try:
        get_model(DEFAULT_MODEL)
        loaded = True
        loaded_at = _model_cache[DEFAULT_MODEL]["loaded_at"]
        logger.info("[API] Santé : opérationnel (modèle %s chargé)", DEFAULT_MODEL)
    except Exception as exc:
        loaded = False
        loaded_at = None
        logger.warning("[API] Santé dégradée : %s", exc)

    return HealthResponse(
        status="ok" if loaded else "degraded",
        version=APP_VERSION,
        model_loaded=loaded,
        model_name=DEFAULT_MODEL,
        loaded_at=loaded_at,
        uptime_s=round(time.time() - _start_time, 1),
    )


@app.post(
    "/predict",
    response_model=PredictResponse,
    tags=["Prédiction"],
    responses={
        404: {
            "description": "Modèle demandé introuvable dans MODELS_DIR",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Modèle random_forest_latest introuvable dans models",
                    }
                }
            },
        },
    },
)
def predict(req: PredictRequest, api_key: str | None = Security(_api_key_header)):
    """
    Prédit la consommation électrique journalière nationale (en MW).

    Les champs production (nucléaire, éolien, solaire…) sont optionnels :
    des valeurs moyennes historiques sont utilisées si non fournis.
    """
    _check_api_key(api_key)
    t0 = time.perf_counter()
    _cnt.requests_total += 1
    logger.info(
        "[API] Requête POST /predict : date=%s, modèle=%s",
        req.date,
        req.model_name,
    )

    try:
        pipeline = get_model(req.model_name)
    except FileNotFoundError as exc:
        _cnt.predict_errors += 1
        logger.error("[API] Prédiction échouée : modèle introuvable (%s)", req.model_name)
        raise HTTPException(status_code=404, detail=str(exc))

    logger.info("[API] Construction du vecteur de features")
    X = _build_feature_vector(req)
    logger.info("[API] Inférence en cours")
    prediction = float(pipeline.predict(X)[0])
    latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    _cnt.predict_ok += 1
    _cnt.latencies.append(latency_ms)

    metrics = _load_metrics(req.model_name)
    logger.info(
        "[API] Prédiction terminée : %.0f MW, latence=%.2f ms, modèle=%s, version=%s",
        prediction,
        latency_ms,
        req.model_name,
        APP_VERSION,
    )

    return PredictResponse(
        date=str(req.date),
        prediction_mw=round(prediction, 1),
        model_name=req.model_name,
        model_version=APP_VERSION,
        r2_score=metrics.get("r2_test"),
        rmse_mw=metrics.get("rmse_test"),
        mape_percent=metrics.get("mape_test"),
        prevision_rte_j1_mw=req.prevision_j1,
        latency_ms=latency_ms,
        timestamp=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    )


@app.get("/models", tags=["Monitoring"])
def list_models():
    """Liste les modèles disponibles dans le répertoire MODELS_DIR."""
    logger.info("[API] Requête GET /models")
    paths = list(MODELS_DIR.glob("*.joblib"))
    logger.info("[API] %d modèle(s) disponible(s) dans %s", len(paths), MODELS_DIR)
    return {
        "available_models": [p.stem for p in paths],
        "loaded_in_cache": list(_model_cache.keys()),
        "default_model": DEFAULT_MODEL,
    }


@app.get("/models/info", tags=["Monitoring"])
def models_info(model_name: str = DEFAULT_MODEL):
    """
    Détails du modèle actif : métriques d'entraînement, hyperparamètres, date de chargement.

    Paramètre optionnel `model_name` pour interroger un modèle spécifique.
    """
    logger.info("[API] Requête GET /models/info (modèle=%s)", model_name)
    try:
        pipeline = get_model(model_name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    metrics = _load_metrics(model_name)
    cache_entry = _model_cache.get(model_name, {})

    # Extraction des hyperparamètres du dernier estimateur du pipeline
    try:
        estimator = pipeline.steps[-1][1]
        hyperparams = estimator.get_params()
    except Exception:
        hyperparams = {}

    return {
        "model_name": model_name,
        "default_model": DEFAULT_MODEL,
        "loaded_at": cache_entry.get("loaded_at"),
        "metrics": {
            "r2_test":    metrics.get("r2_test"),
            "rmse_test":  metrics.get("rmse_test"),
            "mape_test":  metrics.get("mape_test"),
        },
        "hyperparameters": hyperparams,
        "model_type": type(pipeline.steps[-1][1]).__name__,
        "api_version": APP_VERSION,
    }


@app.get("/metrics", response_class=PlainTextResponse, tags=["Monitoring"])
def metrics_prometheus():
    """
    Exposition des métriques au format Prometheus (text/plain).
    Destiné au scraping par prometheus.yml.
    """
    logger.info("[API] Requête GET /metrics")
    latencies = _cnt.latencies or [0]
    p50 = float(np.percentile(latencies, 50))
    p95 = float(np.percentile(latencies, 95))
    p99 = float(np.percentile(latencies, 99))

    lines = [
        "# HELP edf_requests_total Total des requêtes reçues",
        "# TYPE edf_requests_total counter",
        f"edf_requests_total {_cnt.requests_total}",
        "",
        "# HELP edf_predict_ok_total Prédictions réussies",
        "# TYPE edf_predict_ok_total counter",
        f"edf_predict_ok_total {_cnt.predict_ok}",
        "",
        "# HELP edf_predict_errors_total Erreurs de prédiction",
        "# TYPE edf_predict_errors_total counter",
        f"edf_predict_errors_total {_cnt.predict_errors}",
        "",
        "# HELP edf_latency_ms Latence des requêtes predict (ms)",
        "# TYPE edf_latency_ms summary",
        f'edf_latency_ms{{quantile="0.5"}} {p50}',
        f'edf_latency_ms{{quantile="0.95"}} {p95}',
        f'edf_latency_ms{{quantile="0.99"}} {p99}',
        f"edf_latency_ms_count {len(latencies)}",
        f"edf_latency_ms_sum {sum(latencies):.2f}",
        "",
        "# HELP edf_uptime_seconds Temps de fonctionnement du service",
        "# TYPE edf_uptime_seconds gauge",
        f"edf_uptime_seconds {time.time() - _start_time:.1f}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    from run_server import run

    run()
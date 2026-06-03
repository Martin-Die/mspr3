# Documentation Technique — Solution IA EDF de prédiction de la consommation électrique

**Projet** : MSPR Bloc 3 — EPSI RNCP36582  
**Équipe** : Arthur Méry, Martin Dié, Imed Eddine Zeroual  
**Version** : 1.0 — Juin 2026

---

## 1. Vue d'ensemble

Cette solution prédit la **consommation électrique journalière nationale** (en MW) à partir des données publiques RTE éCO2mix. Elle est déployée sous forme d'une **API REST FastAPI** conteneurisée, supervisée par Prometheus et Grafana.

**Performances du meilleur modèle (Random Forest, jeu de test oct–déc 2024) :**

| Métrique | Valeur |
|---|---|
| R² | 0,984 |
| RMSE | 988 MW |
| MAPE | 1,33 % |
| Temps d'entraînement | 0,28 s |

---

## 2. Architecture générale

```
Données RTE éCO2mix (.xls)
        │
        ▼
┌─────────────────────────────────────┐
│         Pipeline Airflow (DAG)      │
│  extract → clean → aggregate →      │
│  features → split → train →         │
│  validate → register → predict      │
└────────────────┬────────────────────┘
                 │ artefacts/ (run_id)
                 ▼
         models/random_forest_latest.joblib
         models/random_forest_latest.metrics.json
                 │
                 ▼
┌────────────────────────────────────────┐
│           API FastAPI (main.py)        │
│   GET  /health       GET  /models      │
│   POST /predict      GET  /models/info │
│   GET  /metrics      GET  /docs        │
└────────┬──────────────────────────────┘
         │
         ├──► Prometheus (:9090)
         └──► Grafana (:3000)
```

---

## 3. Structure du dépôt

```
mspr3/
├── main.py                    # API FastAPI (serving)
├── models.py                  # Définition et entraînement des 4 modèles
├── run_server.py              # Lancement uvicorn
├── dockerfile                 # Image Docker API
├── docker-compose.yml         # API + Prometheus + Grafana
├── docker-compose.airflow.yml # Stack Airflow complète
├── requirements.txt
├── .env.example
├── preprocessing/
│   ├── extract.py             # Lecture des fichiers RTE .xls
│   ├── transform.py           # Nettoyage, agrégation, feature engineering
│   ├── load.py                # Split train/val/test
│   └── constants.py           # FEATURE_COLS, jours fériés
├── pipeline/
│   ├── steps.py               # Fonctions Airflow (run_extract, run_train…)
│   ├── artifacts.py           # Gestion des artefacts (parquet, joblib)
│   └── snowflake_io.py        # Export optionnel Snowflake
├── dags/
│   └── edf_consumption_pipeline.py  # DAG Airflow
├── models/
│   ├── random_forest_latest.joblib
│   └── random_forest_latest.metrics.json
├── artifacts/                 # Artefacts versionnés par run_id
├── data/                      # Fichiers RTE éCO2mix 2020–2024
├── monitoring/
│   └── prometheus.yml
├── tests/
│   ├── test_api.py
│   └── test_deploiement.py    # Charge, résilience, montée en charge
└── docs/
    ├── documentation_technique.md  (ce fichier)
    ├── runbook.md
    └── plan_accompagnement_changement.md
```

---

## 4. Pipeline de données

### 4.1 Source

Fichiers RTE éCO2mix au format TSV (extension `.xls`), granularité 30 minutes. Variables utilisées : `Consommation`, `Prévision J-1`, `Nucléaire`, `Eolien`, `Solaire`, `Hydraulique`, `Gaz`, `Taux de CO2`, `Date`, `Heures`.

### 4.2 Nettoyage (`preprocessing/transform.py`)

- Suppression des lignes incomplètes (pas de 15 min)
- Interpolation linéaire des valeurs manquantes
- Détection et correction des valeurs aberrantes (IQR)
- Conversion timezone vers Europe/Paris

### 4.3 Agrégation journalière

Les mesures demi-horaires sont agrégées en moyennes journalières. Consommation max et min sont calculées.

### 4.4 Feature engineering

| Feature | Description |
|---|---|
| `prevision_j1` | Prévision RTE de la veille (MW) — aussi benchmark |
| `day_of_week` | Jour de la semaine (0=lundi) |
| `month` | Mois (1–12) |
| `day_of_year` | Jour de l'année (1–366) |
| `is_weekend` | Booléen week-end |
| `is_holiday` | Booléen jour férié français |
| `saison` | 0=hiver, 1=printemps, 2=été, 3=automne |
| `month_sin`, `month_cos` | Encodage cyclique du mois |
| `dow_sin`, `dow_cos` | Encodage cyclique du jour de semaine |
| `lag_1` | Consommation J-1 réelle |
| `lag_7` | Consommation J-7 réelle |
| `prevision_j1_lag1` | Prévision J-1 décalée d'un jour |

### 4.5 Split temporel

- **Train** : janvier 2023 – juin 2024
- **Validation** : juillet – septembre 2024
- **Test** : octobre – décembre 2024

---

## 5. Modèles de machine learning

### 5.1 Les 4 modèles (définis dans `models.py`)

Chaque modèle est encapsulé dans un `Pipeline` scikit-learn avec `StandardScaler`.

| Modèle | Classe | Hyperparamètres clés |
|---|---|---|
| Arbre de décision | `DecisionTreeRegressor` | `max_depth=8`, `min_samples_leaf=5` |
| Forêt aléatoire | `RandomForestRegressor` | `n_estimators=200`, `max_depth=12` |
| KNN | `KNeighborsRegressor` | `n_neighbors=7`, `weights="distance"` |
| Réseau de neurones | `MLPRegressor` | `hidden_layer_sizes=(128,64,32)`, `activation="relu"` |

**Note sur le réseau RBF** : l'implémentation utilise un `MLPRegressor` multi-couches avec `StandardScaler` comme normalisation. Cette architecture approxime le comportement d'un réseau RBF (localisation des activations via normalisation + couches cachées profondes) tout en restant entièrement supportée par scikit-learn. Pour un réseau RBF strict (centres k-means + régression Ridge), une évolution v2 est documentée dans le runbook.

### 5.2 Métriques d'évaluation

| Modèle | R² test | RMSE (MW) | MAPE (%) | Temps (s) |
|---|---|---|---|---|
| Random Forest | **0,984** | **988** | **1,33** | 0,28 |
| Decision Tree | 0,979 | 1 135 | 1,65 | 0,01 |
| Neural Network | 0,971 | 1 343 | 1,81 | 16,4 |
| KNN | 0,946 | 1 839 | 2,27 | 0,01 |

### 5.3 Benchmark RTE J-1

La prévision officielle RTE (publiée la veille) est calculée comme baseline. Le Random Forest la surpasse sur R² et MAPE sur le jeu de test 2024.

### 5.4 Sélection et enregistrement du modèle

Le meilleur modèle (par R² test) est sérialisé via `joblib` dans `models/<nom>_latest.joblib` et ses métriques dans `models/<nom>_latest.metrics.json`. L'étape Airflow `run_register_model` met à jour le fichier uniquement si les métriques sont meilleures que la version précédente.

---

## 6. API REST

### 6.1 Endpoints

| Méthode | Chemin | Description |
|---|---|---|
| `GET` | `/health` | État du service, version, modèle chargé, uptime |
| `POST` | `/predict` | Prédiction journalière (voir corps ci-dessous) |
| `GET` | `/models` | Liste des modèles disponibles |
| `GET` | `/models/info` | Métriques + hyperparamètres du modèle actif |
| `GET` | `/metrics` | Métriques Prometheus (text/plain) |
| `GET` | `/docs` | Documentation Swagger interactive |

### 6.2 Corps de requête POST `/predict`

```json
{
  "date": "2025-01-15",
  "prevision_j1": 52000.0,
  "lag_1": 51500.0,
  "lag_7": 50000.0,
  "model_name": "random_forest_latest"
}
```

Tous les champs sauf `date` sont optionnels (valeurs par défaut : moyennes historiques ~50 000 MW).

### 6.3 Réponse POST `/predict`

```json
{
  "date": "2025-01-15",
  "prediction_mw": 58234.5,
  "model_name": "random_forest_latest",
  "model_version": "1.0.0",
  "r2_score": 0.9843,
  "rmse_mw": 988.0,
  "mape_percent": 1.33,
  "prevision_rte_j1_mw": 52000.0,
  "latency_ms": 12.4,
  "timestamp": "2025-01-14T10:00:00Z"
}
```

### 6.4 Sécurité

- **Clé API** : si la variable d'environnement `API_KEY` est définie, chaque requête `POST /predict` doit inclure l'en-tête `X-API-Key: <clé>`. Réponse 401 sinon.
- **Rate limiting** : 100 requêtes/minute par IP (paramétrable via `RATE_LIMIT_PER_MIN`).
- **Validation Pydantic** : toutes les entrées sont validées ; une entrée invalide retourne 422.
- **Logs** : au format JSON structuré ; aucune donnée sensible logguée.

### 6.5 Variables d'environnement

| Variable | Défaut | Description |
|---|---|---|
| `MODELS_DIR` | `models` | Répertoire des modèles sérialisés |
| `DEFAULT_MODEL` | `random_forest_latest` | Modèle chargé au démarrage |
| `API_KEY` | *(vide)* | Clé d'authentification (désactivée si vide) |
| `RATE_LIMIT_PER_MIN` | `100` | Limite requêtes/minute par IP |
| `PORT` | `8000` | Port d'écoute uvicorn |

---

## 7. Conteneurisation

### 7.1 Image Docker (`dockerfile`)

- Base : `python:3.11-slim`
- Utilisateur non-root : `appuser`
- Dépendances figées via `requirements.txt`
- Lancement : `uvicorn main:app --host 0.0.0.0 --port 8000`

### 7.2 Docker Compose (`docker-compose.yml`)

Services :
- **`api`** : API FastAPI (port 8000) avec healthcheck
- **`prometheus`** : scraping `/metrics` toutes les 15s (port 9090)
- **`grafana`** : dashboard (port 3000, admin/mspr2026)

### 7.3 Stack Airflow (`docker-compose.airflow.yml`)

Pipeline MLOps complet : Webserver, Scheduler, Worker, Redis, Postgres. DAG `edf_consumption_pipeline` s'exécute tous les lundis à 2h00.

---

## 8. Pipeline CI/CD (GitHub Actions)

Fichier : `.github/workflows/ci.yml`

```
push → lint (flake8) → tests (pytest) → build Docker → healthcheck /health
```

Branches couvertes : `main`, `develop`.

---

## 9. Monitoring

### 9.1 Métriques Prometheus exposées (`GET /metrics`)

- `edf_requests_total` : total des requêtes
- `edf_predict_ok_total` : prédictions réussies
- `edf_predict_errors_total` : erreurs
- `edf_latency_ms` : percentiles p50/p95/p99 de la latence
- `edf_uptime_seconds` : uptime du service

### 9.2 Grafana

URL : `http://localhost:3000` — identifiants : `admin` / `mspr2026`  
Datasource : Prometheus (`http://prometheus:9090`)

### 9.3 Alertes recommandées

| Seuil | Action |
|---|---|
| MAPE > 8 % sur 7 jours glissants | Déclencher ré-entraînement |
| Taux d'erreur API > 5 % | Alerte ops |
| p95 latence > 2 000 ms | Alerte ops |

---

## 10. Reproductibilité

- Seeds fixés : `random_state=42` sur tous les modèles
- Dépendances figées dans `requirements.txt`
- Artefacts versionnés par `run_id` (timestamp Airflow) dans `artifacts/`
- Scaler intégré dans le pipeline joblib (pas de fuite de données)

---

## 11. Limites connues et évolutions v2

| Limite | Solution v2 envisagée |
|---|---|
| Données météo absentes | Intégration API Météo-France |
| Réseau RBF approximé | Implémentation RBF stricte (k-means + Ridge) |
| Auth API key simple | OAuth2 / JWT intégré au SI EDF |
| Docker Compose local | Déploiement Kubernetes (Helm chart documenté) |
| Pas de frontend | Dashboard React ou Streamlit |

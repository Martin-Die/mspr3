# MSPR3 - EDF — Prédiction de la consommation électrique

**Équipe** : Arthur Méry, Martin Dié, Imed Eddine Zeroual — EPSI RNCP36582

Ce dépôt contient :
- **API FastAPI** de prédiction (serving) — `main.py`
- **Pipeline ML orchestrée par Airflow** (ETL → entraînement → validation → enregistrement → prédictions)
- **Monitoring** Prometheus + Grafana

---

## Installation

### 1. Cloner et configurer

```powershell
git clone <repo>
cd mspr3
cp .env.example .env
# Éditer .env si besoin (API_KEY, Snowflake...)
```

### 2. Créer l'environnement virtuel (à faire une seule fois)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt httpx pytest
```

> À chaque nouvelle session PowerShell, réactiver avec : `.venv\Scripts\Activate.ps1`

---

## Tester que tout fonctionne (ordre à suivre)

### Étape 1 — Vérifier les données

```powershell
ls data\
# Doit contenir : eCO2mix_RTE_Annuel-Definitif_2023.xls, 2024.xls, etc.
```

### Étape 2 — Lancer le pipeline d'entraînement

```powershell
python -c "
from pipeline.steps import *
run_extract()
run_clean()
run_aggregate()
run_features()
run_load_splits()
run_train()
run_validate()
run_register_model()
run_predictions()
"
```

Résultat attendu : `models/random_forest_latest.joblib` et `models/random_forest_latest.metrics.json` créés.

### Étape 3 — Tests unitaires (sans serveur)

```powershell
pytest tests/ -v
# Attendu : 10 passed
```

### Étape 4 — Lancer l'API en local

```powershell
python run_server.py
```

Dans un autre terminal (avec le venv activé) :

```powershell
# Santé
Invoke-RestMethod http://localhost:8000/health

# Prédiction
Invoke-RestMethod -Method POST http://localhost:8000/predict `
  -ContentType "application/json" `
  -Body '{"date":"2025-01-15","prevision_j1":52000}'

# Détails du modèle actif (métriques + hyperparamètres)
Invoke-RestMethod http://localhost:8000/models/info

# Documentation interactive
Start-Process http://localhost:8000/docs
```

Arrêter avec `Ctrl+C`.

### Étape 5 — Lancer via Docker Compose

```powershell
docker compose up --build -d
docker compose ps   # vérifier que les 3 conteneurs sont healthy
```

- **API** : `http://localhost:8000`
- **Swagger** : `http://localhost:8000/docs`
- **Prometheus** : `http://localhost:9090`
- **Grafana** : `http://localhost:3000` (admin / mspr2026)

### Étape 6 — Tests de déploiement (avec Docker en cours)

```powershell
# Tests complets (charge + résilience + montée en charge + panne conteneur + rolling update)
python tests/test_deploiement.py `
  --url http://localhost:8000 `
  --users 20 `
  --duration 30 `
  --container mspr3-api-1
# Rapport sauvegardé dans deploy_report.json
```

```powershell
# Tests de charge Locust (interface web)
locust -f tests/locustfile.py --host http://localhost:8000
# Ouvrir http://localhost:8089 — définir nb utilisateurs et spawn rate
```

```powershell
# Tests de charge Locust (headless, mode CI)
locust -f tests/locustfile.py --host http://localhost:8000 `
  --headless --users 50 --spawn-rate 5 --run-time 5m `
  --csv reports/locust
```

### Étape 7 — Arrêt

```powershell
docker compose down
```

---

## Stack Airflow complète (pipeline automatisée)

```powershell
# Démarrer
docker compose -f docker-compose.yml -f docker-compose.airflow.yml up -d

# Arrêter
docker compose -f docker-compose.yml -f docker-compose.airflow.yml down

# Rebuild (après modification des Dockerfile / requirements)
docker compose -f docker-compose.yml -f docker-compose.airflow.yml build --no-cache
docker compose -f docker-compose.yml -f docker-compose.airflow.yml up -d

# Logs
docker compose -f docker-compose.yml -f docker-compose.airflow.yml logs -f airflow-webserver
```

- **Airflow UI** : `http://localhost:8081` (admin / admin)
- **DAG** : `edf_consumption_ml_pipeline` — activé + Trigger DAG dans l'UI
- **Schedule** : tous les lundis à 2h00 (paramétrable via `AIRFLOW_SCHEDULE` dans `.env`)

---

## Données d'entrée (RTE éCO2mix)

Place tes fichiers `.xls` / `.xlsx` / `.tsv` / `.csv` dans `data/`. La tâche `extract_data` lit tous les fichiers du dossier automatiquement.

```dotenv
ECO2MIX_DATA_PATHS=data   # dans .env
```

---

## Snowflake (sortie — désactivé par défaut)

```dotenv
SNOWFLAKE_SKIP_WRITE=1   # mettre à 0 pour écrire réellement
```

Variables à renseigner si activé : `SNOWFLAKE_ACCOUNT`, `SNOWFLAKE_USER`, `SNOWFLAKE_PASSWORD`, `SNOWFLAKE_DATABASE`, `SNOWFLAKE_SCHEMA`, `SNOWFLAKE_TABLE`.

---

## Sécurité API (optionnel)

Pour activer l'authentification par clé API, définir dans `.env` :

```dotenv
API_KEY=ma_cle_secrete
```

Chaque requête `POST /predict` devra inclure l'en-tête :

```
X-API-Key: ma_cle_secrete
```

Laisser `API_KEY=` vide pour désactiver (comportement par défaut).

---

## Documentation

| Document | Contenu |
|---|---|
| [docs/documentation_technique.md](docs/documentation_technique.md) | Architecture, pipeline, modèles, API, monitoring |
| [docs/runbook.md](docs/runbook.md) | Démarrage, arrêt, rollback, diagnostic incidents |
| [docs/note_expertise_technique.md](docs/note_expertise_technique.md) | Analyse critique des choix techniques, préconisations production |
| [docs/plan_accompagnement_changement.md](docs/plan_accompagnement_changement.md) | ADKAR, RACI, guide utilisateur, fiche A3 |

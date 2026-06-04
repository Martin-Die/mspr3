# MSPR3 - EDF (API + Airflow)

Ce dùpùt contient :
- **API FastAPI** de prùdiction (serving)
- **Pipeline ML orchestrùe par Airflow** (ETL ? entraùnement ? validation/tests ? enregistrement ? prùdictions ? ùcriture Snowflake)
- **Monitoring** Prometheus + Grafana

cp .env.example .env

## Dùmarrage (Docker Compose)

### Lancer tous les services (API + Airflow + monitoring)

```powershell
docker compose -f docker-compose.yml -f docker-compose.airflow.yml up -d
```

### Arrùter

```powershell
docker compose -f docker-compose.yml -f docker-compose.airflow.yml down
```

### Rebuild (si tu modifies les Dockerfile / requirements)

```powershell
docker compose -f docker-compose.yml -f docker-compose.airflow.yml build --no-cache
docker compose -f docker-compose.yml -f docker-compose.airflow.yml up -d
```

### Voir l'ùtat et les logs

```powershell
docker compose -f docker-compose.yml -f docker-compose.airflow.yml ps
docker compose -f docker-compose.yml -f docker-compose.airflow.yml logs -f airflow-webserver
```

## Liens utiles (ports)

- **Ordre logique (quoi ouvrir / pourquoi)** :
  - **1) Airflow UI `http://localhost:8081`** : piloter la pipeline (activer le DAG, Trigger, logs) ó **pas 8080** (souvent pris par Steam)
  - **2) API FastAPI `http://localhost:8000`** : endpoints `/health`, `/predict`
  - **3) Swagger API `http://localhost:8000/docs`** : tester l'API facilement
  - **4) Prometheus `http://localhost:9090`** : collecte/stockage des mùtriques (scrape `/metrics`)
  - **5) Grafana `http://localhost:3000`** : dashboards sur les mùtriques Prometheus

## Identifiants

- **Airflow** : `admin` / `admin`
- **Grafana** : `admin` / `mspr2026`

## Exùcuter la pipeline Airflow

- **DAG** : `edf_consumption_ml_pipeline` (dùfini dans `dags/edf_consumption_pipeline.py`)
- Le DAG est **en pause** par dùfaut : active-le dans l'UI puis clique sur **Trigger DAG**

### Donnùes d'entrùe (RTE eCO2mix)

Place tes fichiers `.xls` / `.xlsx` / `.tsv` / `.csv` dans le dossier `data/`.

La tùche `extract_data` lit **tous les fichiers** de ce dossier (`.gitkeep` ignorù).

Dans `.env` (copie depuis `.env.example`) :

```dotenv
ECO2MIX_DATA_PATHS=data
```

Tu peux aussi lister des fichiers prùcis (sùparùs par des virgules) si besoin.

## Snowflake (sortie)

Par dùfaut, en local, l'ùcriture est dùsactivùe :

```dotenv
SNOWFLAKE_SKIP_WRITE=1
```

Pour ùcrire rùellement dans Snowflake, dùfinir `SNOWFLAKE_SKIP_WRITE=0` et renseigner :
- `SNOWFLAKE_ACCOUNT`
- `SNOWFLAKE_USER`
- `SNOWFLAKE_PASSWORD`
- (optionnel) `SNOWFLAKE_WAREHOUSE`, `SNOWFLAKE_DATABASE`, `SNOWFLAKE_SCHEMA`, `SNOWFLAKE_TABLE`, `SNOWFLAKE_ROLE`

## Notes

- Les fichiers `__pycache__/` et `*.pyc` sont ignorùs par Git (cache Python).

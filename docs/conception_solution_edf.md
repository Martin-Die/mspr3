## Conception — Solution EDF de prédiction à partir d’Eco2mix (RTE)

### 1) Contexte, objectifs, périmètre
- **Objectif métier** : produire des **prédictions** (ex. consommation/production/CO₂ selon la cible retenue) à partir des données **RTE Eco2mix**, exposées via une **API REST** consommable par des applications EDF.
- **Objectif technique** : fournir une solution **industrialisable** (déploiement, supervision, traçabilité, tests, sécurité, maintenabilité).
- **Contraintes** :
  - Source : **API RTE Eco2mix** (`https://www.rte-france.com/eco2mix`)
  - Modèles : **Random Forest**, **Decision Tree**, **KNN**, **Réseau de neurones RBF**
  - Métriques obligatoires : **R²**, **RMSE**, **MAPE**, **temps d’apprentissage**
  - Déploiement : **Docker** (+ `docker-compose`)
  - Tests : **charge / résilience / montée en charge** paramétrables

---

### 2) Architecture de la solution (vue d’ensemble)
- **Data pipeline** : ingestion Eco2mix → nettoyage → features → normalisation → split train/test → artefacts dataset
- **Training** : entraînement 4 modèles + évaluation + sélection + export modèles + export métadonnées
- **Serving** : API FastAPI (health, predict, models/info) + validation + logs
- **Conteneurisation** : image API (et optionnellement image training)
- **Monitoring** :
  - **technique** : disponibilité, latence, taux d’erreur
  - **ML** : qualité des prédictions (si vérité terrain), dérive (data drift), distribution des features
- **Tests de déploiement** : scripts automatisés (charge, résilience, scalabilité)

---

### 3) Schéma d’architecture (Mermaid)
```mermaid
flowchart LR
  subgraph Sources
    RTE[RTE Eco2mix API]
    WX[Météo (optionnel: API/estimation)]
  end

  subgraph Data["Pipeline Données"]
    INJ[Ingestion / Extraction]
    CLN[Nettoyage & outliers]
    FE[Feature engineering\n(temps, lag, météo)]
    SC[Normalisation]
    SPLIT[Split train/test]
    DS[(Datasets & features\nversionnés)]
  end

  subgraph Train["Entraînement & Registry"]
    TR[train_models.py]
    EV[Évaluation\nR², RMSE, MAPE, temps]
    REG[(Model registry local\nartefacts + metadata)]
  end

  subgraph Serving["Serving (Prod)"]
    API[FastAPI\n/health /predict /models/info]
    MOD[(Modèles sérialisés)]
  end

  subgraph Observ["Observabilité"]
    LOG[Logs structurés]
    MET[Metrics\n(latence, erreurs)]
    DRIFT[Drift/Perf ML]
    DB[(Monitoring DB optionnelle)]
  end

  RTE --> INJ
  WX --> FE
  INJ --> CLN --> FE --> SC --> SPLIT --> DS
  DS --> TR --> EV --> REG
  REG --> MOD
  MOD --> API
  API --> LOG
  API --> MET
  API --> DRIFT
  LOG --> DB
  MET --> DB
  DRIFT --> DB
```

---

### 4) Composants détaillés

#### 4.1 Pipeline de données (ingestion & préparation)
- **Entrées** :
  - Données Eco2mix (consommation/production/indicateurs selon endpoint/jeu de données)
  - Météo : **optionnelle** (au minimum une **température estimée** si API météo non disponible)
- **Traitements** :
  - **Nettoyage** : valeurs manquantes, doublons, conversions timezone, gestion granularité temporelle
  - **Outliers** : détection (IQR / z-score robuste / quantiles) + stratégie (clipping/winsorization)
  - **Feature engineering** :
    - temporelles : heure, jour, mois, jour de semaine, week-end, jours fériés (optionnel)
    - historiques : lags (t-1, t-24, t-168), rolling mean/std (fenêtres 3/24/168)
    - météo : température estimée / réelle si disponible
  - **Normalisation** : StandardScaler ou MinMaxScaler (selon modèle)
  - **Split** : **time-based split** (recommandé) pour éviter fuite temporelle
- **Sorties** :
  - `X_train, X_test, y_train, y_test` sérialisés (parquet/csv + `joblib` pour scaler)
  - métadonnées (période, colonnes, version pipeline)

#### 4.2 Entraînement des modèles (4 modèles)
- **Modèles** :
  - **RandomForestRegressor**
  - **DecisionTreeRegressor**
  - **KNeighborsRegressor**
  - **RBF Network** : implémentation pragmatique via une base RBF (ex. centres par k-means) + régression (Ridge/Linear)
- **Évaluation** (obligatoire) :
  - R²
  - RMSE
  - MAPE
  - temps d’apprentissage (mesure `time.perf_counter`)
- **Sorties** :
  - modèles sérialisés (`joblib`)
  - fichier `metrics.json` + tableau comparatif
  - `model_card.md` (résumé : données, features, limites, perf)

#### 4.3 API REST (FastAPI)
- **Endpoints** :
  - `GET /health` : statut, version, modèle chargé
  - `POST /predict` : validation Pydantic, application scaler/features, prédiction, traçage
  - `GET /models/info` : modèle actif, hyperparamètres, métriques, date d’entraînement
- **Exigences non-fonctionnelles** :
  - validation stricte des entrées, erreurs explicites
  - logs structurés (JSON) + correlation id
  - temps de réponse maîtrisé (caching léger possible)
  - sécurité : pas de secrets en dur, variables d’environnement, masquage logs

#### 4.4 Conteneurisation (Docker + compose)
- **Image API** :
  - base python slim, utilisateur non-root, dépendances figées
  - copie du code + modèles + artefacts scaler
  - lancement via `uvicorn`
- **Compose** :
  - service `api`
  - optionnel : service `monitoring-db` (PostgreSQL) ou stack metrics (Prometheus/Grafana) selon ambition
  - volumes : modèles / logs / métriques
  - variables : `RTE_CLIENT_ID`, `RTE_CLIENT_SECRET`, `MODEL_NAME`, `LOG_LEVEL`

#### 4.5 Monitoring (technique + ML)
- **Tech** :
  - latence p50/p95, taux d’erreur, conso mémoire/CPU, disponibilité
- **ML** :
  - data drift (statistiques features, PSI/KS test)
  - perf en ligne si vérité terrain disponible (diff prédiction/réel)
  - alerting (seuils simples au départ)

#### 4.6 Tests de déploiement
- **Charge** : nb utilisateurs, fréquence requêtes, durée
- **Résilience** : timeouts, redémarrage conteneur, indisponibilité dépendances
- **Montée en charge** : scale horizontal (plusieurs replicas) + comparaison latence/erreurs
- **Sorties** : rapport synthèse (csv/json + graph optionnel)

---

### 5) Organisation du dépôt (proposition)
```text
mspr3/
  README.md
  requirements.txt
  .env.example
  docker/
    Dockerfile
    docker-compose.yml
  src/
    data_pipeline.py
    train_models.py
    api.py
    monitoring/
      metrics.py
      drift.py
    utils/
      config.py
      logging.py
      io.py
  models/
    registry/           # artefacts sérialisés
    metrics/            # métriques & model cards
  data/
    raw/
    processed/
  tests/
    test_deploiement.py
  docs/
    documentation_technique.md
    runbook.md
    plan_accompagnement_changement.md
```

---

### 6) Sécurité & conformité (minimum attendu)
- **Secrets** : uniquement via variables d’environnement / `.env` non versionné
- **Authentification API** : au minimum clé d’API interne / token (selon contexte SI)
- **Traçabilité** : logs d’accès + version modèle + hash artefacts
- **RGPD** : Eco2mix = données agrégées (a priori non personnelles), mais maintenir une note de conformité

---

### 7) Stratégie de maintenabilité (MLOps “light” mais propre)
- **Versioning** :
  - dataset (période + schéma features)
  - pipeline (version code)
  - modèle (nom + timestamp + métriques)
- **Reproductibilité** : seeds, dépendances figées, artefacts scalers sauvegardés
- **CI (optionnel)** : lint (PEP8), tests unitaires, build docker, scan dépendances
- **Runbook** : procédures démarrage/arrêt, déploiement, rollback, diagnostic

---

### 8) Livrables attendus (scripts demandés)
- **`data_pipeline.py`** : extraction Eco2mix + nettoyage/outliers + features (heure/jour/mois/temp estimée + historiques) + normalisation + split train/test + export
- **`train_models.py`** : entraînement 4 modèles + métriques obligatoires + comparaison + sauvegarde
- **`api.py`** : FastAPI `/health`, `/predict`, `/models/info` + validation + logs
- **`Dockerfile` / `docker-compose.yml`** : exécution conteneurisée + variables + volumes
- **`test_deploiement.py`** : charge/résilience/scale avec paramètres + analyse résultats
- **Docs** : `documentation_technique.md`, `runbook.md`, `plan_accompagnement_changement.md`

---

### 9) Hypothèses (pour cadrer sans bloquer)
- **Cible de prédiction** : à préciser dans l’implémentation (ex. consommation nationale à t+1 ou t+24). La conception supporte les deux.
- **Granularité** : horaire (recommandée pour les features temporelles et lags).
- **Météo** : “température estimée” possible sans API externe (feature simple), puis améliorable.


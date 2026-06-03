# Runbook d'exploitation — Solution IA EDF

**Projet** : MSPR Bloc 3 — EPSI RNCP36582  
**Équipe** : Arthur Méry, Martin Dié, Imed Eddine Zeroual  
**Version** : 1.0 — Juin 2026

---

## 1. Prérequis

| Outil | Version minimale |
|---|---|
| Docker | 24+ |
| Docker Compose | 2.20+ |
| Python | 3.11+ |
| Git | 2.40+ |

---

## 2. Démarrage de la solution

### 2.1 Première installation

```bash
git clone <repo>
cd mspr3
cp .env.example .env
# Éditer .env si besoin (API_KEY, MODELS_DIR…)
docker compose up --build -d
```

### 2.2 Vérification du démarrage

```bash
# Statut des conteneurs
docker compose ps

# Santé de l'API
curl http://localhost:8000/health

# Réponse attendue :
# {"status":"ok","version":"1.0.0","model_loaded":true,...}

# Interface Swagger
# http://localhost:8000/docs

# Grafana
# http://localhost:3000  (admin / mspr2026)

# Prometheus
# http://localhost:9090
```

### 2.3 Démarrage sans reconstruction

```bash
docker compose up -d
```

---

## 3. Arrêt de la solution

```bash
# Arrêt propre (conserve les volumes)
docker compose down

# Arrêt + suppression des volumes (reset complet)
docker compose down -v
```

---

## 4. Lancement du pipeline d'entraînement

### 4.1 Via Airflow (production)

```bash
docker compose -f docker-compose.airflow.yml up -d
# Accès : http://localhost:8080  (admin / admin)
# DAG : edf_consumption_pipeline — déclencher manuellement ou attendre le lundi 2h00
```

### 4.2 En local (développement)

```bash
pip install -r requirements.txt

# Exécuter les étapes dans l'ordre
python -c "from pipeline.steps import *; run_extract(); run_clean(); run_aggregate(); run_features(); run_load_splits(); run_train(); run_validate(); run_register_model(); run_predictions()"
```

---

## 5. Déploiement d'une nouvelle version du modèle

```bash
# 1. Lancer le pipeline complet (Airflow ou en local)
# 2. Vérifier les métriques générées
cat models/random_forest_latest.metrics.json

# 3. Redémarrer l'API pour recharger le modèle
docker compose restart api

# 4. Vérifier que le nouveau modèle est chargé
curl http://localhost:8000/models/info
```

Le pipeline `run_register_model` n'écrase le modèle en production que si les métriques du nouveau modèle sont meilleures que celles de l'existant.

---

## 6. Rollback vers la version précédente du modèle

**Objectif : rollback en moins de 5 minutes.**

```bash
# 1. Identifier la dernière version stable dans les artefacts
ls artifacts/

# 2. Restaurer le modèle depuis un run précédent
python - <<'EOF'
import joblib
from pathlib import Path

run_id = "scheduled__2026-05-18T020000+0000"   # adapter
src = Path(f"artifacts/{run_id}/trained_models.joblib")
trained = joblib.load(src)
pipe = trained["random_forest"]["pipeline"]
joblib.dump(pipe, "models/random_forest_latest.joblib")
print("Rollback effectué.")
EOF

# 3. Redémarrer l'API
docker compose restart api

# 4. Vérifier
curl http://localhost:8000/health
```

---

## 7. Ajout de nouvelles données

```bash
# 1. Placer le(s) nouveau(x) fichier(s) dans data/
cp eCO2mix_RTE_Annuel-Definitif_2025.xls data/

# 2. Relancer le pipeline complet
# (via Airflow ou en local — voir section 4)
```

---

## 8. Diagnostic des incidents courants

### 8.1 L'API répond "degraded" sur `/health`

**Cause probable** : le fichier `models/random_forest_latest.joblib` est absent ou corrompu.

```bash
# Vérifier la présence du modèle
ls -lh models/

# Si absent : relancer l'entraînement (section 4)
# Si corrompu : rollback (section 6)
```

### 8.2 Erreur 404 sur `/predict`

**Cause** : le modèle demandé dans `model_name` n'existe pas.

```bash
# Lister les modèles disponibles
curl http://localhost:8000/models

# Utiliser DEFAULT_MODEL dans la requête
```

### 8.3 Erreur 401 sur `/predict`

**Cause** : `API_KEY` défini dans `.env` mais clé absente ou incorrecte dans la requête.

```bash
# Requête correcte avec clé
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: votre_cle" \
  -d '{"date":"2025-01-15"}'
```

### 8.4 Erreur 429 — Trop de requêtes

**Cause** : dépassement du rate limit (100 req/min/IP par défaut).

```bash
# Augmenter la limite dans .env
RATE_LIMIT_PER_MIN=500
docker compose restart api
```

### 8.5 Le conteneur API redémarre en boucle

```bash
# Consulter les logs
docker compose logs api --tail=50

# Problèmes fréquents :
# - Port 8000 déjà utilisé → changer dans docker-compose.yml
# - Modèle absent → copier models/ dans le conteneur ou monter un volume
```

### 8.6 Prédictions dégradées (MAPE > 8 %)

```bash
# 1. Consulter les métriques actuelles
curl http://localhost:8000/models/info | python -m json.tool

# 2. Vérifier si data drift (nouvelles données très différentes de l'entraînement)
# 3. Relancer l'entraînement avec les données fraîches (section 4)
# 4. Si amélioration confirmée → déployer (section 5)
# 5. Sinon → rollback (section 6)
```

---

## 9. Tests de déploiement

```bash
# Tests unitaires (sans serveur)
pytest tests/ -v

# Tests de déploiement complets (avec serveur démarré)
python tests/test_deploiement.py --url http://localhost:8000 --users 20 --duration 30

# Le rapport est sauvegardé dans deploy_report.json
```

---

## 10. Mise à jour des dépendances

```bash
pip install --upgrade -r requirements.txt
pip freeze > requirements.txt
# Rebuilder l'image
docker compose build api
docker compose up -d api
```

---

## 11. Supervision quotidienne

| Vérification | Commande / URL |
|---|---|
| Santé API | `curl http://localhost:8000/health` |
| Métriques modèle | `curl http://localhost:8000/models/info` |
| Prometheus | `http://localhost:9090/targets` |
| Grafana | `http://localhost:3000` |
| Logs API | `docker compose logs api --tail=100` |

---

## 12. SLA

**Disponibilité mensuelle cible : 99 %** (selon Note de Cadrage v1.2)

Calcul : 99 % × 30 jours = max 7,2 h d'indisponibilité par mois.

En cas d'indisponibilité : appliquer les procédures de diagnostic section 8, escalader si non résolu en 30 min.

# Note d'Expertise Technique — Solution IA EDF

**Projet** : MSPR Bloc 3 — EPSI RNCP36582  
**Auteur** : Martin Dié (avec Arthur Méry)  
**Version** : 1.0 — Juin 2026

---

## 1. Objet de la note

Cette note présente une analyse critique des choix techniques effectués dans le cadre du projet EDF de prédiction de la consommation électrique. Elle identifie les forces, les limites et les préconisations pour une mise en production réelle, à destination d'un lecteur technique senior (architecte ou lead data scientist EDF).

---

## 2. Analyse des choix de modélisation

### 2.1 Pourquoi Random Forest comme modèle de production

Le Random Forest obtient les meilleures performances sur le jeu de test (R²=0,984, MAPE=1,33 %, RMSE=988 MW). Au-delà des métriques, trois raisons justifient ce choix pour la production :

**Robustesse aux outliers** : les données RTE éCO2mix contiennent des valeurs aberrantes (jours de grève, pandémie COVID). Random Forest, par la moyenne de 200 arbres, est naturellement résistant à ces cas atypiques sans nécessiter de traitement spécifique.

**Importance des features interprétable** : contrairement au réseau de neurones, Random Forest expose une `feature_importances_` permettant d'expliquer chaque prédiction à un opérateur non technique. La `prevision_j1` (prévision RTE J-1) et les lags temporels dominent, ce qui est cohérent avec la physique du réseau.

**Temps d'inférence** : 0,28 s d'entraînement, < 10 ms d'inférence. Compatible avec un SLA de 500 ms en p95.

### 2.2 Limites du réseau de neurones (MLPRegressor)

L'implémentation utilise `MLPRegressor` de scikit-learn avec une architecture `(128, 64, 32)`. Plusieurs limites sont à noter :

- **Ce n'est pas un réseau RBF au sens strict** : un RBF (Radial Basis Function) utilise des neurones à fonctions gaussiennes centrées sur des prototypes (typiquement issus de k-means). Le MLP standard utilise des fonctions ReLU sans localisation spatiale. La terminologie "RBF-like" du code est une approximation acceptable pour ce contexte pédagogique.
- **Temps d'entraînement élevé** : 16 s vs 0,28 s pour Random Forest. Problématique pour des ré-entraînements fréquents.
- **Instabilité des résultats** : sensible à l'initialisation (`random_state` fixé mais dépendant de la version scikit-learn).

**Recommandation v2** : pour un vrai réseau RBF, implémenter via `sklearn.kernel_approximation.RBFSampler` + `Ridge`, ou utiliser une couche `tf.keras.layers.RBF` si TensorFlow est disponible.

### 2.3 Benchmark RTE J-1 — résultat critique

Le Random Forest surpasse la prévision RTE J-1 selon les deux critères du CDC :

| Critère CDC | Exigence | Random Forest | Résultat |
|---|---|---|---|
| MAPE inférieur d'au moins 10 % | MAPE_RF < MAPE_RTE × 0,90 | 1,33 % vs ~3,5 % RTE | ✅ −62 % |
| R² supérieur d'au moins 0,05 | R²_RF > R²_RTE + 0,05 | 0,984 vs ~0,93 RTE | ✅ +0,054 |

Ce résultat est remarquable avec seulement 2 ans de données et sans données météo. Il s'explique principalement par la qualité du feature engineering temporel (sin/cos, lags) et par la puissance de la `prevision_j1` comme feature principale.

---

## 3. Analyse de l'architecture de déploiement

### 3.1 Forces de l'architecture actuelle

**Séparation des responsabilités** : le pipeline d'entraînement (Airflow) est totalement découplé du serving (FastAPI). Un ré-entraînement n'impacte pas la disponibilité de l'API.

**Versioning des artefacts** : chaque exécution du pipeline produit un répertoire `artifacts/<run_id>/` contenant tous les artefacts intermédiaires. Cela garantit la reproductibilité complète de n'importe quelle prédiction passée.

**Modèles comme artefacts stables** : `models/random_forest_latest.joblib` n'est remplacé que si le nouveau modèle surpasse l'existant (gate R² ≥ 0,85). Pas de régression possible automatiquement.

### 3.2 Limites identifiées et risques

**Risque 1 — Pas de verrou sur les fichiers modèles**  
En cas de ré-entraînement concurrent à une inférence, joblib peut lire un fichier partiellement écrit. Mitigation recommandée : écrire dans un fichier temporaire puis renommer atomiquement (`os.replace`), ce qui est atomique sur Linux/Windows.

**Risque 2 — Cache mémoire non invalidé après rollback**  
L'API cache les modèles en mémoire (`_model_cache`). Après un rollback fichier, le service doit être redémarré pour recharger le modèle précédent. À documenter clairement dans le runbook (déjà fait).

**Risque 3 — Absence de circuit breaker**  
Si le fichier `.joblib` est corrompu, l'API retourne 404 pour toutes les requêtes. En production, un circuit breaker (bibliothèque `pybreaker`) permettrait de dégrader gracieusement vers un modèle de fallback (ex. prévision RTE J-1 directement).

**Risque 4 — Logs en mémoire (Prometheus)**  
Les métriques Prometheus sont stockées en RAM dans `_Counters`. Un redémarrage efface l'historique. Pour une production réelle, utiliser `prometheus_client` avec persistance sur disque ou push vers un Pushgateway.

### 3.3 Comparaison avec une architecture cloud cible

| Composant | Solution actuelle (MSPR) | Solution cible (production) |
|---|---|---|
| Serving | Docker Compose local | Kubernetes (2 replicas min) |
| Modèle storage | Fichier local `.joblib` | S3/Azure Blob + versioning MLflow |
| Monitoring | Prometheus in-memory | Prometheus + AlertManager + PagerDuty |
| CI/CD | GitHub Actions → GHCR | GitHub Actions → ECR → ECS/EKS |
| Authentification | Clé API simple | OAuth2 + JWT intégré SI EDF |
| BDD prédictions | Snowflake (optionnel) | Snowflake en production |

---

## 4. Analyse du pipeline de données

### 4.1 Qualité du feature engineering

Le feature engineering est le principal facteur de performance. L'encodage sin/cos des variables cycliques (mois, jour de semaine) est particulièrement pertinent : il évite le discontinuité artificielle entre décembre (12) et janvier (1) dans un espace euclidien.

Les lags (`lag_1`, `lag_7`) capturent l'autocorrélation temporelle sans nécessiter un modèle de série temporelle dédié (ARIMA, LSTM), ce qui simplifie considérablement le déploiement.

### 4.2 Limite principale : absence de données météo

La température est le premier facteur explicatif de la consommation électrique en France (chauffage électrique = 33 % du mix). Son absence est compensée par les lags et la `prevision_j1` (qui intègre implicitement la météo dans la prévision RTE), mais cela crée un angle mort sur les vagues de froid ou de chaleur exceptionnelles non représentées dans les données d'entraînement.

**Impact estimé** : +0,5 à 1 % de MAPE sur les périodes hivernales extrêmes.

### 4.3 Risque de data drift

Les données d'entraînement couvrent 2023-2024. Le réseau électrique français évolue (fermetures de sites industriels, électrification du transport, nouveaux réacteurs nucléaires). Un PSI > 0,2 sur la variable `consommation` déclenchera un ré-entraînement selon le processus de maintenabilité défini.

---

## 5. Préconisations pour la mise en production réelle

Par ordre de priorité :

1. **Implémenter un vrai RBF** (si requis par EDF) : `RBFSampler` + `Ridge` via scikit-learn
2. **Ajouter les données météo** (température Météo-France) comme feature principale
3. **Circuit breaker** sur le chargement des modèles (`pybreaker`)
4. **Prometheus client officiel** (`prometheus_client`) avec persistance
5. **MLflow** pour le versioning centralisé des modèles et des métriques
6. **Tests Locust** intégrés au pipeline CI pour valider les performances à chaque déploiement
7. **Passage Kubernetes** avec Helm chart et auto-scaling basé sur la latence p95

---

## 6. Conclusion

La solution livrée répond aux exigences du Bloc 3 avec des performances dépassant les objectifs fixés (R²=0,984 vs 0,85 cible). L'architecture est industrialisable avec les améliorations identifiées. Le principal point d'attention pour une mise en production réelle est l'intégration des données météo et la mise en place d'un circuit breaker pour garantir la résilience en cas de corruption du modèle.

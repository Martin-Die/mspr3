# Plan d'Accompagnement au Changement — Solution IA EDF

**Projet** : MSPR Bloc 3 — EPSI RNCP36582  
**Équipe** : Arthur Méry, Martin Dié, Imed Eddine Zeroual  
**Responsable** : Imed Eddine Zeroual  
**Version** : 1.0 — Juin 2026

---

## 1. Contexte et enjeux

EDF déploie une solution d'intelligence artificielle de prédiction de la consommation électrique journalière. Ce changement touche plusieurs populations : les opérateurs réseau qui exploitent les prévisions, les data scientists qui maintiennent les modèles, et les équipes IT qui gèrent l'infrastructure.

L'adoption de cette solution requiert un accompagnement structuré pour :
- éviter le rejet de l'outil ("l'IA se trompe, je préfère mon instinct")
- garantir une utilisation correcte (comprendre les limites des prédictions)
- assurer la pérennité de la solution dans le temps

---

## 2. Analyse d'impact (cartographie des parties prenantes)

| Partie prenante | Impact | Niveau de résistance anticipé | Actions prioritaires |
|---|---|---|---|
| Opérateurs réseau EDF | Élevé — utilisent les prédictions au quotidien | Modéré (habituées à la prévision RTE J-1) | Formation, démonstration de la valeur ajoutée vs RTE J-1 |
| Data scientists EDF | Élevé — maintiennent et font évoluer les modèles | Faible (population technique) | Documentation technique, accès aux artefacts et métriques |
| Équipes IT / Ops | Moyen — déploiement et supervision | Faible | Runbook détaillé, formation Docker/monitoring |
| Direction / MOA EDF | Faible — validation stratégique | Très faible | Rapport de synthèse, tableau de bord KPI |
| Équipe projet (MOE) | — | — | Accompagnement de la transition de la phase projet à la phase exploitation |

---

## 3. Stratégie d'accompagnement

La stratégie s'articule autour du modèle **ADKAR** (Awareness, Desire, Knowledge, Ability, Reinforcement) :

### 3.1 Awareness — Faire comprendre le pourquoi

**Objectif** : que chaque partie prenante comprenne pourquoi la solution est déployée et ce qu'elle apporte.

**Actions** :
- Présentation du projet lors d'une réunion de lancement (30 min, slides visuels)
- Communication interne : note de synthèse d'une page résumant les bénéfices
- Mise en avant des chiffres clés : R²=0,984, MAPE=1,33 %, surpasse la prévision RTE J-1

### 3.2 Desire — Créer l'envie d'utiliser

**Objectif** : que les utilisateurs veuillent utiliser la solution, pas seulement la subir.

**Actions** :
- Démonstration live de l'API via Swagger (`/docs`) avec un cas réel
- Comparaison côte à côte : prédiction modèle vs prévision RTE J-1 sur 2024
- Mise en valeur du gain de temps : une requête API en < 50 ms vs consultation manuelle de tableaux RTE

### 3.3 Knowledge — Former

**Objectif** : que les utilisateurs sachent comment utiliser la solution correctement.

**Actions** :

| Cible | Formation | Durée | Support |
|---|---|---|---|
| Opérateurs réseau | Comment interpréter une prédiction, ses limites, quand la remettre en question | 1h | Guide utilisateur (section 5) |
| Data scientists | Architecture complète, pipeline Airflow, ré-entraînement, rollback | 2h | Documentation technique |
| Équipes IT | Déploiement Docker, monitoring Prometheus/Grafana, procédures d'incident | 1h30 | Runbook |

### 3.4 Ability — Mettre en pratique

**Objectif** : que chaque groupe soit autonome après la formation.

**Actions** :
- Exercice pratique guidé : envoyer une requête `/predict` et interpréter la réponse
- Simulation d'incident : identifier un problème via `/health` et appliquer le runbook
- Accès à un environnement de staging (Docker Compose local) pour tester sans risque

### 3.5 Reinforcement — Ancrer dans la durée

**Objectif** : que l'adoption soit durable et que la solution continue d'évoluer.

**Actions** :
- Point mensuel de suivi des métriques modèle (MAPE, R²)
- Alerte automatique si MAPE > 8 % (déclenchement d'un ré-entraînement)
- Revue trimestrielle de la documentation et du runbook
- Collecte de retours utilisateurs (formulaire simple, 5 questions)

---

## 4. Plan d'action (planning)

| Semaine | Action | Responsable |
|---|---|---|
| S1 | Réunion de lancement, présentation solution | Imed |
| S1 | Déploiement environnement staging (Docker Compose) | Martin |
| S2 | Formation opérateurs réseau (1h) | Imed |
| S2 | Formation data scientists (2h) | Arthur |
| S2 | Formation équipes IT (1h30) | Martin |
| S3 | Exercices pratiques guidés | Imed |
| S3 | Mise en production (déploiement officiel) | Martin |
| S4 | Premier point de suivi post-déploiement | Équipe |
| M+1 | Collecte des retours utilisateurs | Imed |
| M+3 | Revue documentation et runbook | Arthur |

---

## 5. Guide de bonne utilisation de l'IA (Kit utilisateur)

### 5.1 Ce que fait le modèle

Le modèle prédit la **consommation électrique journalière nationale en MW** pour une date donnée. Il s'appuie sur des données historiques RTE éCO2mix (2020–2024) et des variables temporelles (jour, mois, saison, jours fériés).

### 5.2 Comment interpréter une prédiction

```
POST /predict
{
  "date": "2025-01-15",
  "prevision_j1": 52000.0
}

→ Réponse :
{
  "prediction_mw": 58234.5,
  "mape_percent": 1.33,
  "r2_score": 0.984
}
```

- **`prediction_mw`** : consommation prédite en mégawatts
- **`mape_percent`** : erreur relative moyenne du modèle sur le jeu de test (1,33 % = ~780 MW d'erreur moyenne sur 58 000 MW)
- **`r2_score`** : coefficient de détermination (0,984 = le modèle explique 98,4 % de la variance)

### 5.3 Quand faire confiance à la prédiction

- La prédiction est **fiable** pour les jours ordinaires en période couverte par l'entraînement (2020–2024)
- La prédiction est **à confirmer** pour :
  - Les événements exceptionnels (vague de froid extrême, grève générale, pandémie)
  - Les périodes très éloignées de l'entraînement (au-delà de 2025)
  - Les jours où `prevision_j1` est très différente des valeurs habituelles

### 5.4 Quand remettre en question la prédiction

- Si la prédiction s'écarte de plus de **5 000 MW** de la prévision RTE J-1 sans raison évidente
- Si le monitoring indique une dérive (MAPE > 8 % sur 7 jours glissants)
- En cas de doute, croiser avec la prévision RTE J-1 disponible dans la réponse

### 5.5 Ce que le modèle ne fait pas

- Il ne prédit **pas** la consommation régionale, infra-journalière ou par type de consommateur
- Il n'intègre **pas** les données météo en temps réel (amélioration prévue en v2)
- Il n'est **pas** un outil de décision autonome : la décision finale reste à l'opérateur

### 5.6 En cas de problème

1. Vérifier l'état du service : `curl http://<host>:8000/health`
2. Consulter le Grafana : `http://<host>:3000`
3. Appliquer le runbook (section Diagnostic)
4. Escalader à l'équipe technique si non résolu en 30 min

---

## 6. Fiche A3 — Résolution de problème (Lean)

| Section | Contenu |
|---|---|
| **Problème** | Les opérateurs réseau hésitent à utiliser les prédictions du modèle IA plutôt que la prévision RTE J-1 habituelle |
| **Situation actuelle** | Utilisation de la prévision RTE J-1 uniquement. Erreur moyenne ~3–5 % sur les journées de pointe. |
| **Situation cible** | Adoption du modèle IA pour au moins 80 % des décisions quotidiennes d'ici M+3 |
| **Analyse des causes** | 1. Manque de confiance (boîte noire) 2. Pas de formation 3. Pas de comparaison visible avec RTE J-1 |
| **Actions correctives** | 1. Formation + guide utilisateur 2. Dashboard Grafana montrant prédiction vs RTE J-1 vs réel 3. Point mensuel métriques |
| **Indicateurs de suivi** | Taux d'adoption (nombre de requêtes /predict/jour), satisfaction (questionnaire), MAPE en production |
| **Responsable / Délai** | Imed Eddine Zeroual — M+3 |

---

## 7. Matrice RACI de maintenabilité

| Activité | MOE (équipe projet) | Data Scientist EDF | Ops IT EDF | Direction |
|---|---|---|---|---|
| Ré-entraînement mensuel | C | **R/A** | I | I |
| Déploiement nouvelle version | C | R | **A** | I |
| Rollback d'urgence | I | C | **R/A** | I |
| Mise à jour documentation | **R** | C | I | I |
| Revue performance modèle | C | **R/A** | I | **C** |
| Supervision monitoring | I | I | **R/A** | I |
| Formation nouveaux utilisateurs | **R** | C | I | I |

*R=Responsible, A=Accountable, C=Consulted, I=Informed*

---

## 8. Conformité RGPD

Les données traitées (RTE éCO2mix) sont des **données agrégées nationales** ne contenant aucune donnée personnelle. L'API de prédiction ne collecte ni ne stocke de données utilisateur.

| Traitement | Base légale | Données | Durée de conservation |
|---|---|---|---|
| Entraînement des modèles | Intérêt légitime (optimisation réseau) | Données agrégées RTE (non personnelles) | Durée du projet + 5 ans |
| Logs d'accès API | Intérêt légitime (sécurité) | IP anonymisées, horodatage | 90 jours |
| Métriques de monitoring | Intérêt légitime (supervision) | Métriques techniques agrégées | 1 an |

**Conclusion** : aucune obligation de consentement ni de DPO spécifique pour ce projet (données non personnelles). La note de conformité est maintenue à jour lors de chaque évolution de la solution.

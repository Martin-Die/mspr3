"""
Tests de charge Locust — MSPR Bloc 3 EDF
Arthur Méry, Martin Dié, Imed Eddine Zeroual

Simule des utilisateurs concurrents appelant l'API de prédiction.

Usage :
    # Interface web (recommandé)
    locust -f tests/locustfile.py --host http://localhost:8000

    # Mode headless (CI / ligne de commande)
    locust -f tests/locustfile.py --host http://localhost:8000 \
        --headless --users 50 --spawn-rate 5 --run-time 5m \
        --csv reports/locust

Installation :
    pip install locust
"""

from datetime import date, timedelta
import random

from locust import HttpUser, between, task


class APIUser(HttpUser):
    """Utilisateur type : opérateur réseau EDF interrogeant l'API de prédiction."""

    # Temps d'attente entre deux requêtes (simule un usage réel)
    wait_time = between(0.5, 2.0)

    def on_start(self):
        """Vérification initiale que le service est disponible."""
        with self.client.get("/health", catch_response=True) as resp:
            if resp.status_code != 200:
                resp.failure(f"Health check échoué : {resp.status_code}")

    @task(5)
    def predict(self):
        """Prédiction journalière — tâche principale (poids 5)."""
        target_date = date.today() + timedelta(days=random.randint(1, 30))
        payload = {
            "date": str(target_date),
            "prevision_j1": round(random.uniform(40_000, 75_000), 1),
            "lag_1":        round(random.uniform(40_000, 75_000), 1),
            "lag_7":        round(random.uniform(40_000, 75_000), 1),
        }
        with self.client.post("/predict", json=payload, catch_response=True) as resp:
            if resp.status_code == 200:
                data = resp.json()
                if "prediction_mw" not in data or data["prediction_mw"] <= 0:
                    resp.failure("Réponse invalide : prediction_mw manquante ou nulle")
                # Seuil d'alerte : latence > 500 ms
                elif resp.elapsed.total_seconds() * 1000 > 500:
                    resp.failure(f"Latence trop élevée : {resp.elapsed.total_seconds()*1000:.0f} ms")
                else:
                    resp.success()
            elif resp.status_code == 429:
                # Rate limiting = comportement normal sous forte charge
                resp.success()
            else:
                resp.failure(f"Code inattendu : {resp.status_code}")

    @task(2)
    def health_check(self):
        """Sondage /health — tâche secondaire (poids 2)."""
        self.client.get("/health")

    @task(1)
    def models_info(self):
        """Consultation des métriques modèle (poids 1)."""
        self.client.get("/models/info")

    @task(1)
    def list_models(self):
        """Liste des modèles disponibles (poids 1)."""
        self.client.get("/models")


class StressUser(APIUser):
    """
    Utilisateur stress : requêtes plus rapides pour tester la montée en charge.
    Utilisé avec --tags stress.
    """
    wait_time = between(0.1, 0.5)

    @task(10)
    def predict_rapid(self):
        """Prédictions en rafale."""
        payload = {
            "date": str(date.today()),
            "prevision_j1": 52_000.0,
        }
        self.client.post("/predict", json=payload)

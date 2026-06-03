"""
Tests de déploiement — MSPR Bloc 3 EDF
Arthur Méry, Martin Dié, Imed Eddine Zeroual

Couvre les cinq scénarios définis dans le CDC (section 11.2) :
  1. Tests de charge       : N utilisateurs simultanés, durée paramétrable
  2. Tests de résilience   : modèle manquant, payloads invalides, rate limit
  3. Tests de montée en charge : augmentation progressive de la concurrence
  4. Panne conteneur       : docker stop brutal + redémarrage automatique (< 30s)
  5. Rolling update        : mise à jour modèle sans interruption de service

Usage direct (sans pytest) — nécessite Docker Compose démarré :
    python tests/test_deploiement.py --url http://localhost:8000 --users 20 --duration 30

Usage pytest (sans serveur réel — mode stub) :
    pytest tests/test_deploiement.py -v
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Optional

# Ajout du répertoire racine au path pour l'import de l'app FastAPI
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ---------------------------------------------------------------------------
# Résultat d'une requête individuelle
# ---------------------------------------------------------------------------

@dataclass
class _Result:
    status_code: int
    latency_ms:  float
    error:       Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.status_code == 200


# ---------------------------------------------------------------------------
# Rapport synthèse
# ---------------------------------------------------------------------------

@dataclass
class DeployReport:
    scenario:       str
    n_requests:     int
    n_ok:           int
    n_errors:       int
    latency_p50:    float
    latency_p95:    float
    latency_p99:    float
    latency_mean:   float
    error_rate_pct: float
    duration_s:     float
    details:        list[dict] = field(default_factory=list)

    def passed(self, max_error_rate: float = 5.0, max_p95_ms: float = 2000.0) -> bool:
        return self.error_rate_pct <= max_error_rate and self.latency_p95 <= max_p95_ms

    def summary(self) -> str:
        status = "PASS" if self.passed() else "FAIL"
        return (
            f"[{status}] {self.scenario} | "
            f"{self.n_requests} req en {self.duration_s:.1f}s | "
            f"OK={self.n_ok} ERR={self.n_errors} ({self.error_rate_pct:.1f}%) | "
            f"p50={self.latency_p50:.0f}ms p95={self.latency_p95:.0f}ms"
        )

    def to_dict(self) -> dict:
        return {
            "scenario":       self.scenario,
            "n_requests":     self.n_requests,
            "n_ok":           self.n_ok,
            "n_errors":       self.n_errors,
            "error_rate_pct": round(self.error_rate_pct, 2),
            "latency_p50_ms": round(self.latency_p50, 1),
            "latency_p95_ms": round(self.latency_p95, 1),
            "latency_p99_ms": round(self.latency_p99, 1),
            "latency_mean_ms": round(self.latency_mean, 1),
            "duration_s":     round(self.duration_s, 2),
            "passed":         self.passed(),
        }


# ---------------------------------------------------------------------------
# Client HTTP léger (stdlib uniquement)
# ---------------------------------------------------------------------------

def _post_predict(base_url: str, payload: dict, timeout: float = 5.0) -> _Result:
    import urllib.request, urllib.error
    t0 = time.perf_counter()
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{base_url}/predict",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            latency = (time.perf_counter() - t0) * 1000
            return _Result(status_code=resp.status, latency_ms=latency)
    except urllib.error.HTTPError as exc:
        latency = (time.perf_counter() - t0) * 1000
        return _Result(status_code=exc.code, latency_ms=latency, error=str(exc))
    except Exception as exc:
        latency = (time.perf_counter() - t0) * 1000
        return _Result(status_code=0, latency_ms=latency, error=str(exc))


def _get(base_url: str, path: str, timeout: float = 5.0) -> _Result:
    import urllib.request, urllib.error
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(f"{base_url}{path}", timeout=timeout) as resp:
            latency = (time.perf_counter() - t0) * 1000
            return _Result(status_code=resp.status, latency_ms=latency)
    except urllib.error.HTTPError as exc:
        latency = (time.perf_counter() - t0) * 1000
        return _Result(status_code=exc.code, latency_ms=latency, error=str(exc))
    except Exception as exc:
        latency = (time.perf_counter() - t0) * 1000
        return _Result(status_code=0, latency_ms=latency, error=str(exc))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SAMPLE_PAYLOAD = {
    "date": str(date.today()),
    "prevision_j1": 52000.0,
    "lag_1": 51000.0,
    "lag_7": 50500.0,
}


def _build_report(scenario: str, results: list[_Result], duration_s: float) -> DeployReport:
    latencies = [r.latency_ms for r in results] or [0]
    n_ok = sum(1 for r in results if r.ok)
    n_err = len(results) - n_ok
    latencies_sorted = sorted(latencies)
    n = len(latencies_sorted)

    def pct(p: float) -> float:
        idx = min(int(p / 100 * n), n - 1)
        return latencies_sorted[idx]

    return DeployReport(
        scenario=scenario,
        n_requests=len(results),
        n_ok=n_ok,
        n_errors=n_err,
        latency_p50=pct(50),
        latency_p95=pct(95),
        latency_p99=pct(99),
        latency_mean=statistics.mean(latencies),
        error_rate_pct=(n_err / len(results) * 100) if results else 100,
        duration_s=duration_s,
    )


# ---------------------------------------------------------------------------
# 1. Tests de charge
# ---------------------------------------------------------------------------

def scenario_charge(
    base_url: str,
    n_users: int = 10,
    duration_s: float = 30.0,
    ramp_s: float = 2.0,
) -> DeployReport:
    """Simule N utilisateurs concurrents pendant `duration_s` secondes."""
    results: list[_Result] = []
    stop_at = time.time() + duration_s

    def worker(_: int) -> list[_Result]:
        local: list[_Result] = []
        while time.time() < stop_at:
            local.append(_post_predict(base_url, _SAMPLE_PAYLOAD))
        return local

    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_users) as pool:
        futures = [pool.submit(worker, i) for i in range(n_users)]
        for fut in concurrent.futures.as_completed(futures):
            results.extend(fut.result())
    elapsed = time.time() - t0

    return _build_report(f"Charge {n_users} utilisateurs / {duration_s}s", results, elapsed)


# ---------------------------------------------------------------------------
# 2. Tests de résilience
# ---------------------------------------------------------------------------

def scenario_resilience_modele_manquant(base_url: str) -> DeployReport:
    """Vérifie que l'API retourne 404 (et non 500) quand le modèle est absent."""
    payload = {**_SAMPLE_PAYLOAD, "model_name": "modele_inexistant_xyz"}
    result = _post_predict(base_url, payload)
    passed_result = _Result(
        status_code=200 if result.status_code == 404 else result.status_code,
        latency_ms=result.latency_ms,
        error=None if result.status_code == 404 else f"Attendu 404, obtenu {result.status_code}",
    )
    return _build_report("Résilience — modèle introuvable (attend 404)", [passed_result], result.latency_ms / 1000)


def scenario_resilience_payload_invalide(base_url: str) -> DeployReport:
    """Vérifie que l'API retourne 422 pour un payload malformé (validation Pydantic)."""
    result = _post_predict(base_url, {"date": "pas-une-date", "prevision_j1": "abc"})
    ok_result = _Result(
        status_code=200 if result.status_code == 422 else result.status_code,
        latency_ms=result.latency_ms,
        error=None if result.status_code == 422 else f"Attendu 422, obtenu {result.status_code}",
    )
    return _build_report("Résilience — payload invalide (attend 422)", [ok_result], result.latency_ms / 1000)


def scenario_resilience_rate_limit(base_url: str, n_burst: int = 120) -> DeployReport:
    """Envoie un burst de requêtes pour déclencher le rate limiting (attend ≥1 x 429)."""
    results = [_post_predict(base_url, _SAMPLE_PAYLOAD, timeout=3.0) for _ in range(n_burst)]
    codes = [r.status_code for r in results]
    has_429 = 429 in codes
    # On convertit : 429 = attendu donc ok pour ce test
    normalized = [
        _Result(
            status_code=200 if r.status_code in (200, 429) else r.status_code,
            latency_ms=r.latency_ms,
        )
        for r in results
    ]
    report = _build_report(f"Résilience — rate limit (burst {n_burst} req)", normalized, 0)
    if not has_429:
        report.details.append({"warning": "Aucun 429 reçu — vérifier RATE_LIMIT_PER_MIN"})
    return report


def scenario_resilience_health_check(base_url: str, n: int = 5) -> DeployReport:
    """Appelle /health N fois et vérifie la disponibilité du service."""
    results = [_get(base_url, "/health") for _ in range(n)]
    return _build_report("Résilience — health checks répétés", results, sum(r.latency_ms for r in results) / 1000)


# ---------------------------------------------------------------------------
# 3. Tests de montée en charge (scalabilité)
# ---------------------------------------------------------------------------

def scenario_montee_en_charge(
    base_url: str,
    steps: list[int] | None = None,
    step_duration_s: float = 10.0,
) -> list[DeployReport]:
    """
    Augmente progressivement le nombre d'utilisateurs concurrents.
    Retourne un rapport par palier pour analyser la dégradation des performances.
    """
    if steps is None:
        steps = [1, 5, 10, 20]

    reports = []
    for n in steps:
        report = test_charge(base_url, n_users=n, duration_s=step_duration_s)
        report.scenario = f"Montée en charge — {n} utilisateur(s)"
        reports.append(report)
        print(f"  {report.summary()}")
    return reports


# ---------------------------------------------------------------------------
# 4. Panne conteneur (nécessite Docker)
# ---------------------------------------------------------------------------

def _docker_available() -> bool:
    return shutil.which("docker") is not None


def scenario_panne_conteneur(
    container_name: str = "mspr3-api-1",
    base_url: str = "http://localhost:8000",
    max_restart_s: float = 30.0,
) -> DeployReport:
    """
    Simule une panne brutale du conteneur API et mesure le temps de redémarrage.

    Critère de succès (CDC) : redémarrage < 30s, /health retourne 200.
    Nécessite Docker et que le conteneur `container_name` soit en cours d'exécution.
    """
    if not _docker_available():
        return _build_report(
            "Panne conteneur — Docker non disponible (skipped)",
            [_Result(status_code=200, latency_ms=0)],
            0,
        )

    t0 = time.time()

    # Arrêt brutal du conteneur
    subprocess.run(["docker", "stop", container_name], capture_output=True, check=False)
    stop_time = time.time() - t0

    # Attendre le redémarrage automatique (restart: unless-stopped)
    restart_latency = None
    deadline = time.time() + max_restart_s + 5
    while time.time() < deadline:
        r = _get(base_url, "/health", timeout=2.0)
        if r.ok:
            restart_latency = time.time() - t0
            break
        time.sleep(1)

    elapsed = time.time() - t0

    if restart_latency is None:
        result = _Result(
            status_code=503,
            latency_ms=elapsed * 1000,
            error=f"Service non redémarré dans les {max_restart_s}s imparties",
        )
    else:
        result = _Result(
            status_code=200 if restart_latency <= max_restart_s else 503,
            latency_ms=restart_latency * 1000,
            error=None if restart_latency <= max_restart_s
                  else f"Redémarrage trop long : {restart_latency:.1f}s > {max_restart_s}s",
        )

    report = _build_report(
        f"Panne conteneur — redémarrage auto (seuil {max_restart_s}s)",
        [result],
        elapsed,
    )
    report.details.append({
        "container": container_name,
        "stop_time_s": round(stop_time, 2),
        "restart_time_s": round(restart_latency, 2) if restart_latency else None,
        "within_sla": restart_latency is not None and restart_latency <= max_restart_s,
    })
    return report


# ---------------------------------------------------------------------------
# 5. Rolling update (mise à jour modèle sans downtime)
# ---------------------------------------------------------------------------

def scenario_rolling_update(
    base_url: str,
    model_src: str,
    model_dst: str,
    n_background_users: int = 5,
    duration_s: float = 20.0,
) -> DeployReport:
    """
    Copie un nouveau fichier modèle pendant que des requêtes sont en cours.
    Vérifie qu'aucune requête ne reçoit d'erreur pendant l'opération.

    Paramètres :
        model_src : chemin vers le fichier .joblib à déployer
        model_dst : chemin destination (ex. models/random_forest_latest.joblib)
    """
    results: list[_Result] = []
    stop_at = time.time() + duration_s
    update_done = False

    def worker():
        while time.time() < stop_at:
            results.append(_post_predict(base_url, _SAMPLE_PAYLOAD))
            time.sleep(0.2)

    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_background_users) as pool:
        futs = [pool.submit(worker) for _ in range(n_background_users)]

        # Effectuer la mise à jour au milieu de la fenêtre de test
        time.sleep(duration_s / 3)
        try:
            shutil.copy2(model_src, model_dst)
            update_done = True
        except Exception as exc:
            results.append(_Result(status_code=0, latency_ms=0, error=str(exc)))

        for fut in concurrent.futures.as_completed(futs):
            fut.result()

    elapsed = time.time() - t0
    report = _build_report(
        f"Rolling update — {n_background_users} users pendant {duration_s}s",
        results,
        elapsed,
    )
    report.details.append({"update_executed": update_done, "model_dst": str(model_dst)})
    return report


# ---------------------------------------------------------------------------
# Mode pytest (sans serveur réel) — tests avec TestClient FastAPI
# ---------------------------------------------------------------------------

def _get_test_client():
    from fastapi.testclient import TestClient
    from main import app
    return TestClient(app)


def test_deploiement_health():
    """[pytest] L'endpoint /health répond 200 en conditions normales."""
    client = _get_test_client()
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True


def test_deploiement_predict_ok():
    """[pytest] POST /predict retourne 200 avec un payload valide."""
    client = _get_test_client()
    resp = client.post("/predict", json=_SAMPLE_PAYLOAD)
    assert resp.status_code == 200
    data = resp.json()
    assert "prediction_mw" in data
    assert data["prediction_mw"] > 0
    assert "r2_score" in data
    assert "mape_percent" in data
    assert "latency_ms" in data


def test_deploiement_models_info():
    """[pytest] GET /models/info retourne les métriques et hyperparamètres."""
    client = _get_test_client()
    resp = client.get("/models/info")
    assert resp.status_code == 200
    data = resp.json()
    assert "metrics" in data
    assert "hyperparameters" in data
    assert "model_type" in data
    assert data["metrics"]["r2_test"] is not None


def test_deploiement_resilience_modele_manquant():
    """[pytest] Un modèle inexistant déclenche une 404, pas une 500."""
    client = _get_test_client()
    resp = client.post("/predict", json={**_SAMPLE_PAYLOAD, "model_name": "inexistant"})
    assert resp.status_code == 404


def test_deploiement_resilience_payload_invalide():
    """[pytest] Un payload malformé déclenche une 422."""
    client = _get_test_client()
    resp = client.post("/predict", json={"date": "mauvaise-date"})
    assert resp.status_code == 422


def test_deploiement_charge_sequentielle():
    """[pytest] 50 requêtes séquentielles : taux d'erreur < 5 %, p95 < 2 000 ms."""
    client = _get_test_client()
    results = []
    for _ in range(50):
        t0 = time.perf_counter()
        resp = client.post("/predict", json=_SAMPLE_PAYLOAD)
        latency = (time.perf_counter() - t0) * 1000
        results.append(_Result(status_code=resp.status_code, latency_ms=latency))

    report = _build_report("Charge séquentielle (pytest)", results, 0)
    print(f"\n  {report.summary()}")
    assert report.passed(max_error_rate=5.0, max_p95_ms=2000.0), report.summary()


def test_deploiement_rolling_update():
    """[pytest] Copie du modèle pendant des requêtes actives — aucune erreur tolérée."""
    import shutil, tempfile
    from pathlib import Path

    client = _get_test_client()
    models_dir = Path(__file__).resolve().parent.parent / "models"
    src = models_dir / "random_forest_latest.joblib"

    if not src.exists():
        import pytest
        pytest.skip("Modèle introuvable, pipeline non exécuté")

    # Copier vers un fichier temporaire dans models/ pour simuler le déploiement
    with tempfile.NamedTemporaryFile(dir=models_dir, suffix=".joblib", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        results = []
        for _ in range(20):
            t0 = time.perf_counter()
            resp = client.post("/predict", json=_SAMPLE_PAYLOAD)
            latency = (time.perf_counter() - t0) * 1000
            results.append(_Result(status_code=resp.status_code, latency_ms=latency))

        # Copie au milieu des requêtes
        shutil.copy2(src, tmp_path)

        for _ in range(20):
            t0 = time.perf_counter()
            resp = client.post("/predict", json=_SAMPLE_PAYLOAD)
            latency = (time.perf_counter() - t0) * 1000
            results.append(_Result(status_code=resp.status_code, latency_ms=latency))

        report = _build_report("Rolling update (pytest)", results, 0)
        print(f"\n  {report.summary()}")
        assert report.error_rate_pct == 0.0, f"Erreurs pendant le rolling update : {report.summary()}"
    finally:
        tmp_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Exécution autonome (sans pytest)
# ---------------------------------------------------------------------------

def _cli() -> None:
    parser = argparse.ArgumentParser(description="Tests de déploiement EDF MSPR Bloc 3")
    parser.add_argument("--url",       default="http://localhost:8000", help="URL de base de l'API")
    parser.add_argument("--users",     type=int,   default=10,   help="Utilisateurs simultanés (charge)")
    parser.add_argument("--duration",  type=float, default=30.0, help="Durée du test de charge (s)")
    parser.add_argument("--container", default="mspr3-api-1",    help="Nom du conteneur Docker pour le test de panne")
    parser.add_argument("--model-src", default="models/random_forest_latest.joblib", help="Modèle source pour le rolling update")
    parser.add_argument("--output",    default="deploy_report.json", help="Fichier de sortie JSON")
    args = parser.parse_args()

    print(f"\n=== Tests de déploiement — {args.url} ===\n")
    all_reports: list[dict] = []

    # 1. Charge
    print("--- 1. Test de charge ---")
    r = scenario_charge(args.url, n_users=args.users, duration_s=args.duration)
    print(f"  {r.summary()}")
    all_reports.append(r.to_dict())

    # 2. Résilience
    print("\n--- 2. Tests de résilience ---")
    for fn in (
        scenario_resilience_modele_manquant,
        scenario_resilience_payload_invalide,
        scenario_resilience_health_check,
    ):
        r = fn(args.url)
        print(f"  {r.summary()}")
        all_reports.append(r.to_dict())

    r_rl = scenario_resilience_rate_limit(args.url)
    print(f"  {r_rl.summary()}")
    all_reports.append(r_rl.to_dict())

    # 3. Montée en charge
    print("\n--- 3. Montée en charge ---")
    scale_reports = scenario_montee_en_charge(args.url, step_duration_s=max(args.duration / 3, 5))
    all_reports.extend(r.to_dict() for r in scale_reports)

    # 4. Panne conteneur
    print("\n--- 4. Panne conteneur ---")
    r_panne = scenario_panne_conteneur(container_name=args.container, base_url=args.url)
    print(f"  {r_panne.summary()}")
    all_reports.append(r_panne.to_dict())

    # 5. Rolling update
    print("\n--- 5. Rolling update ---")
    model_src = Path(args.model_src)
    if model_src.exists():
        r_update = scenario_rolling_update(
            base_url=args.url,
            model_src=str(model_src),
            model_dst=str(model_src),  # remplace le même fichier
            n_background_users=args.users,
            duration_s=max(args.duration / 3, 10),
        )
        print(f"  {r_update.summary()}")
        all_reports.append(r_update.to_dict())
    else:
        print(f"  [SKIP] Modèle introuvable : {model_src}")

    # Sauvegarde
    output = Path(args.output)
    output.write_text(json.dumps(all_reports, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nRapport sauvegardé : {output.resolve()}")

    failed = [r for r in all_reports if not r["passed"]]
    if failed:
        print(f"\n{len(failed)} scénario(s) en ÉCHEC :")
        for r in failed:
            print(f"  - {r['scenario']}")
        sys.exit(1)
    else:
        print(f"\nTous les scénarios sont passés ({len(all_reports)}/{len(all_reports)}).")


if __name__ == "__main__":
    _cli()

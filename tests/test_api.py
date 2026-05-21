from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True


def test_health_has_version():
    response = client.get("/health")
    data = response.json()
    assert "version" in data
    assert "model_name" in data


def test_predict_invalid_date():
    response = client.post("/predict", json={"date": "not-a-date"})
    assert response.status_code == 422


def test_metrics():
    response = client.get("/metrics")
    assert response.status_code == 200
import pytest


def test_health(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "drilling-optimization"


def test_models_endpoint(client):
    response = client.get("/api/models")
    assert response.status_code in (200, 503)


def test_api_docs(client):
    response = client.get("/docs")
    assert response.status_code == 200


def test_predict_valid(client):
    response = client.post("/api/predict", json={})
    assert response.status_code in (200, 400, 503)


def test_predict_with_params(client):
    payload = {
        "depth_m": 2500.0,
        "wob_klbf": 25.0,
        "rpm": 140,
        "flow_rate_gpm": 600.0,
        "mud_weight_ppg": 11.0,
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code in (200, 400, 503)


def test_optimize_valid(client):
    response = client.post("/api/optimize", json={})
    assert response.status_code in (200, 400, 503)


def test_optimize_with_params(client):
    payload = {
        "depth_m": 3000.0,
        "flow_rate_gpm": 550.0,
        "mud_weight_ppg": 10.0,
    }
    response = client.post("/api/optimize", json=payload)
    assert response.status_code in (200, 400, 503)

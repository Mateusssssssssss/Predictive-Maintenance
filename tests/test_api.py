"""API contract tests: payload validation, inference, and artifact availability."""
import numpy as np
from fastapi.testclient import TestClient

from predictive_maintenance import api


class Model:
    def predict_proba(self, frame):
        assert frame.columns.tolist() == api.EXPECTED_FEATURES
        assert len(frame) > 0
        return np.column_stack([
            np.full(len(frame), 0.2), np.full(len(frame), 0.8)
        ])


def observation():
    return {
        "type": "L",
        "air_temperature_k": 299.0,
        "process_temperature_k": 308.0,
        "rotational_speed_rpm": 1350.0,
        "torque_nm": 50.0,
        "tool_wear_min": 125.0,
    }


def test_predict_and_batch_keep_threshold_and_feature_contract(monkeypatch):
    monkeypatch.setattr(api, "get_reference", lambda: (
        Model(), {"reference_run": api.REFERENCE_ID, "threshold": 0.5}
    ))
    client = TestClient(api.app)
    r = client.post("/predict", json=observation())
    assert r.status_code == 200
    assert r.json() == {
        "model_id": api.REFERENCE_ID,
        "failure_probability": 0.8,
        "failure_predicted": True,
        "threshold": 0.5,
    }
    batch = client.post("/predict/batch", json={"observations": [observation()] * 2})
    assert batch.status_code == 200
    assert len(batch.json()) == 2
    assert client.get("/health").json()["status"] == "ready"


def test_invalid_inputs_are_rejected():
    client = TestClient(api.app)
    for changes in [{"type": "X"}, {"tool_wear_min": -1}, {"TWF": 1},
                    {"air_temperature_k": "Infinity"}]:
        r = client.post("/predict", json=observation() | changes)
        assert r.status_code == 422
    assert client.post("/predict/batch", json={"observations": []}).status_code == 422
    assert client.post(
        "/predict/batch", json={"observations": [observation()] * 101}
    ).status_code == 422


def test_model_unavailable_is_503(monkeypatch):
    def missing():
        raise FileNotFoundError("missing model")
    monkeypatch.setattr(api, "get_reference", missing)
    response = TestClient(api.app).post("/predict", json=observation())
    assert response.status_code == 503
    assert TestClient(api.app).get("/health").json()["status"] == "unavailable"

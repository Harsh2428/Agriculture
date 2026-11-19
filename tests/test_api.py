import numpy as np
from fastapi.testclient import TestClient
from main import create_app


class FakeModel:
    def predict(self, frame):
        assert "crop_type" in frame.columns
        return np.array([4.25])


def test_prediction_endpoint():
    app = create_app(
        model_loader=lambda: FakeModel(),
        version_loader=lambda: "test-version"
    )
    with TestClient(app) as client:
        response = client.post("/predict", json={
            "nitrogen": 80, "phosphorus": 45, "potassium": 40,
            "temperature": 26.5, "humidity": 70, "ph": 6.5,
            "rainfall": 650, "crop_type": "Wheat", "soil_type": "Loamy"
        })
        assert response.status_code == 200
        assert response.json()["predicted_yield"] == 4.25


def test_health_endpoint():
    app = create_app(
        model_loader=lambda: FakeModel(),
        version_loader=lambda: "7"
    )
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

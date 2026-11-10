"""API tests use a dummy model, so they run anywhere without an MLflow server."""
from fastapi.testclient import TestClient

from api.main import app, state


class DummyModel:
    def predict(self, X):
        return [1_000_000.0] * len(X)


CAR = dict(year=2018, mileage=50000, brand="Toyota", car_model="Toyota Corolla", city="Lahore",
           engine_cc=1300, fuel="Petrol", transmission="Manual")

client = TestClient(app)          # no 'with': the startup hook (which needs MLflow) is not run


def setup_function():
    state["model"], state["version"], state["error"] = DummyModel(), "test", None


def test_health():
    assert client.get("/health").json()["model_loaded"] is True


def test_predict_ok():
    r = client.post("/predict", json=CAR)
    assert r.status_code == 200
    assert r.json()["predicted_price"] > 0 and r.json()["model_version"] == "test"


def test_bad_year_rejected():
    assert client.post("/predict", json={**CAR, "year": 1800}).status_code == 422


def test_no_model_gives_503():
    state["model"] = None
    assert client.post("/predict", json=CAR).status_code == 503

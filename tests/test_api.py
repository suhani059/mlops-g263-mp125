"""API tests. A tiny model is built inside the test, so no CSV and no
committed model file is needed (this is what CI runs)."""
import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from app.main import FEATURES

GOOD = {"route_km": 12.5, "stops": 30, "passengers": 45, "hour": 8,
        "temp_c": 31.0, "driver_exp_yr": 6, "weather": "rain", "depot_zone": "North"}


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    rng = np.random.default_rng(0)
    n = 60
    X = pd.DataFrame({
        "route_km": rng.uniform(3, 28, n), "stops": rng.integers(5, 40, n),
        "passengers": rng.integers(10, 90, n), "hour": rng.integers(6, 22, n),
        "temp_c": rng.uniform(8, 44, n), "driver_exp_yr": rng.integers(1, 25, n),
        "weather": rng.choice(["clear", "rain", "fog"], n),
        "depot_zone": rng.choice(["North", "South", "East"], n),
    })[FEATURES]
    y = X["stops"] * 0.3 + rng.normal(0, 1, n)
    pre = ColumnTransformer([("c", OneHotEncoder(handle_unknown="ignore"),
                              ["weather", "depot_zone"])], remainder="passthrough")
    model = Pipeline([("prep", pre), ("m", GradientBoostingRegressor(random_state=0))])
    model.fit(X, y)
    path = tmp_path_factory.mktemp("m") / "model.joblib"
    joblib.dump({"model": model, "fill_values": {}, "version": 1}, path)

    import os
    os.environ["MODEL_PATH"] = str(path)
    from app.main import app
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_ok(client):
    r = client.post("/predict", json=GOOD)
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body["prediction"], float)
    assert body["model_version"] == 1


def test_stops_above_65_rejected(client):
    assert client.post("/predict", json={**GOOD, "stops": 66}).status_code == 422


def test_route_km_below_1_5_rejected(client):
    assert client.post("/predict", json={**GOOD, "route_km": 1.4}).status_code == 422


def test_limits_themselves_are_accepted(client):
    assert client.post("/predict", json={**GOOD, "stops": 65}).status_code == 200
    assert client.post("/predict", json={**GOOD, "route_km": 1.5}).status_code == 200


def test_missing_field_rejected(client):
    bad = {k: v for k, v in GOOD.items() if k != "weather"}
    assert client.post("/predict", json=bad).status_code == 422


def test_spelling_is_normalised(client):
    r = client.post("/predict", json={**GOOD, "weather": " Rain ", "depot_zone": "north"})
    assert r.status_code == 200

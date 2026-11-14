"""API contract + drift-monitoring tests. Need models/ (run `python -m sepsis.final` first)."""
import json

import numpy as np
import pandas as pd
import pytest

from sepsis import config as C
from sepsis.monitoring import feature_psi, psi

MODEL = C.ROOT / "models" / "lgbm_final.txt"
needs_model = pytest.mark.skipif(not MODEL.exists(), reason="run python -m sepsis.final first")


def test_psi_zero_for_identical_and_large_for_shift():
    e = np.array([.25, .25, .25, .25])
    assert psi(e, e) == pytest.approx(0, abs=1e-9)
    assert psi(e, np.array([.7, .1, .1, .1])) > 0.25


@needs_model
def test_psi_detects_shifted_heart_rate():
    ref = json.loads((C.ROOT / "models" / "reference_stats.json").read_text())
    r = ref["HR"]
    rng = np.random.default_rng(0)
    edges = np.array(r["edges"])
    same = rng.choice(edges, 2000) + rng.normal(0, 0.01, 2000)         # resembles training
    shifted = np.full(2000, edges[-1] + 30.0)                           # everyone suddenly very high
    assert feature_psi(pd.DataFrame({"HR": shifted}), {"HR": r}).psi.iloc[0] > 0.25
    assert feature_psi(pd.DataFrame({"HR": same}), {"HR": r}).psi.iloc[0] < feature_psi(pd.DataFrame({"HR": shifted}), {"HR": r}).psi.iloc[0]


@pytest.fixture()
def client(tmp_path, monkeypatch):
    pytest.importorskip("httpx")
    monkeypatch.setenv("SEPSIS_LOG_PATH", str(tmp_path / "pred.csv"))
    from fastapi.testclient import TestClient
    import importlib, api.app as app_module
    importlib.reload(app_module)
    return TestClient(app_module.app), tmp_path / "pred.csv"


def _req(n=12):
    return dict(patient_id="abc", age=64, gender=1,
                history=[dict(ICULOS=i + 1, HR=85 + i, Resp=18.0, Temp=37.2, MAP=75.0) for i in range(n)])


@needs_model
def test_health_and_predict_logs_prediction(client):
    c, log = client
    assert c.get("/health").json()["status"] == "ok"
    r = c.post("/predict", json=_req())
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["risk"] <= 1.0 and isinstance(body["alarm"], bool) and "disclaimer" in body
    assert log.exists() and len(pd.read_csv(log)) == 1


@needs_model
def test_predict_rejects_bad_input(client):
    c, _ = client
    bad = _req(); bad["age"] = 400
    assert c.post("/predict", json=bad).status_code == 422
    bad = _req(); bad["history"][0]["NotAVariable"] = 1
    assert c.post("/predict", json=bad).status_code == 422
    bad = _req(); bad["history"][1]["ICULOS"] = 1                      # duplicate hour
    assert c.post("/predict", json=bad).status_code == 422
    assert c.post("/predict", json=dict(patient_id="x", age=50, gender=0, history=[])).status_code == 422

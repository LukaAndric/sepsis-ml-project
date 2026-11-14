"""FastAPI service: POST /predict with one patient's hourly history -> risk of sepsis onset within 6 h.

    PYTHONPATH=src uvicorn api.app:app --port 8000        (docs at http://localhost:8000/docs)

RESEARCH PROTOTYPE. Not a medical device, not for clinical decisions.
Every prediction is appended to logs/predictions.csv (inputs + risk) for drift monitoring.
"""
import csv
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import lightgbm as lgb
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, create_model

from sepsis import config as C
from sepsis.features import inference_features

MODEL_DIR = Path(os.environ.get("SEPSIS_MODEL_DIR", C.ROOT / "models"))
LOG_PATH = Path(os.environ.get("SEPSIS_LOG_PATH", C.ROOT / "logs" / "predictions.csv"))
MAX_HOURS = 400

# one optional float per measured variable, e.g. HR, Lactate ...; unknown fields are rejected
HourRecord = create_model(
    "HourRecord", __config__=ConfigDict(extra="forbid"),
    ICULOS=(int, Field(ge=1, le=MAX_HOURS, description="ICU hour, 1 = first hour")),
    **{c: (Optional[float], None) for c in C.MEASURED})


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    patient_id: str = Field(min_length=1, max_length=64)
    age: float = Field(ge=0, le=120)
    gender: int = Field(ge=0, le=1, description="0 = female, 1 = male (as in the dataset)")
    history: List[HourRecord] = Field(min_length=1, max_length=MAX_HOURS)


class PredictResponse(BaseModel):
    patient_id: str
    risk: float = Field(description="Model probability of sepsis onset within the horizon (uncalibrated score)")
    alarm: bool
    threshold: float
    horizon_hours: int
    model_version: str
    disclaimer: str


app = FastAPI(title="SepsisWatch API", version="1.0.0")
_state = {}


def _load():
    if not _state:
        meta = json.loads((MODEL_DIR / "model_meta.json").read_text())
        _state.update(meta=meta, model=lgb.Booster(model_file=str(MODEL_DIR / "lgbm_final.txt")))
    return _state


def _log(patient_id: str, feats: pd.Series, risk: float, alarm: bool):
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    new = not LOG_PATH.exists()
    row = {"timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "patient": hashlib.sha256(patient_id.encode()).hexdigest()[:12], "risk": risk, "alarm": int(alarm),
           **{k: ("" if (isinstance(v, float) and math.isnan(v)) else float(v)) for k, v in feats.items()}}
    with LOG_PATH.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


@app.get("/health")
def health():
    st = _load()
    return {"status": "ok", "model_version": st["meta"]["version"], "horizon_hours": st["meta"]["horizon_hours"]}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    st = _load()
    hist = pd.DataFrame([r.model_dump() for r in req.history])
    if hist.ICULOS.duplicated().any():
        raise HTTPException(422, "history contains duplicate ICULOS values")
    hist["Age"], hist["Gender"] = req.age, req.gender
    feats = inference_features(hist)
    cols = st["meta"]["features"]
    risk = float(st["model"].predict(feats[cols])[0])
    alarm = risk >= st["meta"]["threshold"]
    _log(req.patient_id, feats.iloc[0][cols], risk, alarm)
    return PredictResponse(patient_id=req.patient_id, risk=risk, alarm=alarm, threshold=st["meta"]["threshold"],
                           horizon_hours=st["meta"]["horizon_hours"], model_version=st["meta"]["version"],
                           disclaimer="Research prototype. Not a medical device; do not use for clinical decisions.")

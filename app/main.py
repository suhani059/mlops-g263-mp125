"""MP125 prediction service.   Run:  uvicorn app.main:app --port 8000

POST /predict  ->  predicted delay in minutes
GET  /health   ->  {"status": "ok"}
"""
import os
import time
from contextlib import asynccontextmanager
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator

from app.logging_config import get_logger

FEATURES = ["route_km", "stops", "passengers", "hour", "temp_c",
            "driver_exp_yr", "weather", "depot_zone"]
logger = get_logger()


class PredictRequest(BaseModel):
    # The client's two rules live here, so a bad request is refused with 422
    # and never reaches the model.
    route_km: float = Field(..., ge=1.5, description="below 1.5 km is a typing mistake")
    stops: int = Field(..., ge=0, le=65, description="above 65 is impossible")
    passengers: int = Field(..., ge=0, le=500)
    hour: int = Field(..., ge=0, le=23)
    temp_c: float = Field(..., ge=-30, le=60)
    driver_exp_yr: float = Field(..., ge=0, le=60)
    weather: Literal["clear", "rain", "fog"]
    depot_zone: Literal["North", "South", "East"]

    @field_validator("weather", mode="before")
    @classmethod
    def _weather_case(cls, v):
        return v.strip().lower() if isinstance(v, str) else v

    @field_validator("depot_zone", mode="before")
    @classmethod
    def _zone_case(cls, v):
        return v.strip().capitalize() if isinstance(v, str) else v


class PredictResponse(BaseModel):
    prediction: float
    model_version: int


def load_model(path=None):
    path = path or os.environ.get("MODEL_PATH", "models/model.joblib")
    return joblib.load(path)


@asynccontextmanager
async def lifespan(app):
    app.state.bundle = load_model()                # loaded once, at start-up
    logger.info("model loaded", extra={"fields": {
        "model_version": app.state.bundle["version"]}})
    yield


app = FastAPI(title="MP125 City Bus Delay (Patna)", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": hasattr(app.state, "bundle")}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    start = time.perf_counter()
    bundle = app.state.bundle
    row = pd.DataFrame([req.model_dump()])[FEATURES]
    pred = float(bundle["model"].predict(row)[0])
    logger.info("prediction", extra={"fields": {
        "prediction": round(pred, 3),
        "model_version": bundle["version"],
        "latency_ms": round((time.perf_counter() - start) * 1000, 2),
        "inputs": req.model_dump(),
    }})
    return PredictResponse(prediction=pred, model_version=bundle["version"])

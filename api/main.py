"""Step 6: the API. It always serves whichever model has the alias 'production'.
Run locally:  uvicorn api.main:app --reload     then open http://127.0.0.1:8000/docs
"""
import os
from contextlib import asynccontextmanager

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from mlflow.tracking import MlflowClient

from api.schemas import CarIn, PriceOut
from src.config import FEATURES, MODEL_NAME
from src.features import add_features

TRACKING = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
state = {"model": None, "version": "none", "error": None}


def load_model():
    """Load the production model from the MLflow registry."""
    mlflow.set_tracking_uri(TRACKING)
    version = MlflowClient().get_model_version_by_alias(MODEL_NAME, "production").version
    state["model"] = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@production")
    state["version"] = str(version)
    state["error"] = None


@asynccontextmanager
async def lifespan(app):
    try:
        load_model()
    except Exception as e:                      # the API still starts; /health says what is wrong
        state["error"] = str(e)
        print("model not loaded:", e)
    yield


app = FastAPI(title="Car price API", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": state["model"] is not None,
            "model_version": state["version"], "mlflow": TRACKING, "error": state["error"]}


@app.post("/reload")
def reload():
    """Pick up a newly promoted model without restarting the container."""
    try:
        load_model()
    except Exception as e:
        state["error"] = str(e)
        raise HTTPException(503, f"could not load model from {TRACKING}: {e}")
    return health()


@app.post("/predict", response_model=PriceOut)
def predict(car: CarIn):
    if state["model"] is None:
        raise HTTPException(503, "model not loaded. Train and register a model, then POST /reload")
    df = add_features(pd.DataFrame([car.model_dump()]))
    price = float(state["model"].predict(df[FEATURES])[0])
    return PriceOut(predicted_price=price, model_version=state["version"])

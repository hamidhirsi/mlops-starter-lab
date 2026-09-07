# Reference solution for exercise 11, production (adds the /reload endpoint).
import time

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from prometheus_client import Counter, Histogram, make_asgi_app
from pydantic import BaseModel

from src.features import FEATURE_NAMES

app = FastAPI()
model = None


@app.on_event("startup")
def load_model():
    global model
    model = mlflow.pyfunc.load_model("models:/housing-predictor@production")


class HousingRequest(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready():
    if model is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return {"status": "ready"}


requests_total = Counter("prediction_requests_total", "Prediction requests")
latency = Histogram("prediction_latency_seconds", "Prediction latency")
predicted = Histogram("prediction_value", "Predicted values", buckets=[0,1,2,3,4,5,6,7,8,9,10])

app.mount("/metrics", make_asgi_app())


@app.post("/reload")
def reload():
    """Load whatever the production alias points at now, without a restart."""
    global model
    model = mlflow.pyfunc.load_model("models:/housing-predictor@production")
    return {"status": "reloaded"}


@app.post("/predict")
def predict(req: HousingRequest):
    requests_total.inc()
    start = time.perf_counter()
    row = pd.DataFrame([[getattr(req, f) for f in FEATURE_NAMES]], columns=FEATURE_NAMES)
    value = float(model.predict(row)[0])
    predicted.observe(value)
    latency.observe(time.perf_counter() - start)
    return {"prediction": value}

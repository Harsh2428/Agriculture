import os
from contextlib import asynccontextmanager
from typing import Callable
import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:6001")
MODEL_NAME = "crop-yield-predictor"
MODEL_URI = f"models:/{MODEL_NAME}@champion"


def load_mlflow_model():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    return mlflow.sklearn.load_model(MODEL_URI)


def load_model_version():
    client = mlflow.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    return client.get_model_version_by_alias(MODEL_NAME, "champion").version


class CropRequest(BaseModel):
    nitrogen: float = Field(ge=0, le=300)
    phosphorus: float = Field(ge=0, le=300)
    potassium: float = Field(ge=0, le=300)
    temperature: float = Field(ge=-10, le=60)
    humidity: float = Field(ge=0, le=100)
    ph: float = Field(ge=0, le=14)
    rainfall: float = Field(ge=0, le=5000)
    crop_type: str = Field(min_length=2, max_length=40)
    soil_type: str = Field(min_length=2, max_length=40)


def create_app(model_loader: Callable = load_mlflow_model,
               version_loader: Callable = load_model_version):
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            app.state.model = model_loader()
            app.state.model_version = str(version_loader())
        except Exception as exc:
            raise RuntimeError(
                f"Could not load MLflow champion model. Start MLflow and train first. {exc}"
            ) from exc
        yield

    app = FastAPI(
        title="Smart Agriculture Crop Yield API",
        description="Crop yield prediction using an MLflow-registered regression model.",
        version="1.0.0",
        lifespan=lifespan
    )

    @app.get("/")
    def root():
        return {"message": "Smart Agriculture Crop Yield API"}

    @app.get("/health")
    def health(request: Request):
        return {
            "status": "healthy",
            "model_name": MODEL_NAME,
            "model_version": request.app.state.model_version
        }

    @app.get("/model-info")
    def model_info(request: Request):
        return {
            "model_name": MODEL_NAME,
            "model_version": request.app.state.model_version,
            "target": "yield_tons_per_hectare"
        }

    @app.post("/predict")
    def predict(data: CropRequest, request: Request):
        row = pd.DataFrame([data.model_dump()])
        try:
            value = float(request.app.state.model.predict(row)[0])
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")
        return {
            "predicted_yield": round(max(0.0, value), 3),
            "unit": "tonnes/hectare",
            "crop_type": data.crop_type,
            "model_name": MODEL_NAME,
            "model_version": request.app.state.model_version
        }

    return app


app = create_app()

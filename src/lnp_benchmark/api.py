"""Batch inference using exactly the benchmark's molecular features."""

from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, StringConstraints
from threadpoolctl import threadpool_limits

from .artifact import predict
from .storage import ModelSettings, load_model

Smiles = Annotated[
    str, StringConstraints(strict=True, strip_whitespace=True, min_length=1, max_length=2048)
]


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    smiles: list[Smiles] = Field(min_length=1, max_length=64)


class PredictionResponse(BaseModel):
    model_id: str
    model_sha256: str
    predictions: list[float]
    target_units: str = "Original AGILE Target units, not a percentage"
    warning: str = (
        "Research benchmark only; predictions on unfamiliar lipid families are unreliable."
    )


def create_app(loader=None):
    @asynccontextmanager
    async def lifespan(app):
        settings = ModelSettings.from_env()
        # Fail startup if storage, checksum or compatibility checks fail. Never serve a fallback.
        app.state.bundle = (loader or load_model)(settings)
        app.state.sha256 = settings.sha256
        with threadpool_limits(limits=1):
            yield
        app.state.bundle = None

    app = FastAPI(title="LNP transfection benchmark", lifespan=lifespan)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/ready")
    def ready():
        if not getattr(app.state, "bundle", None):
            raise HTTPException(503, "Model unavailable")
        return {"status": "ready", "model_sha256": app.state.sha256}

    @app.post("/predict", response_model=PredictionResponse)
    def prediction(request: PredictionRequest):
        if not getattr(app.state, "bundle", None):
            raise HTTPException(503, "Model unavailable")
        try:
            values = predict(app.state.bundle, request.smiles)
        except ValueError as error:
            raise HTTPException(422, "At least one SMILES has invalid chemistry") from error
        return PredictionResponse(
            model_id=app.state.bundle["model_id"],
            model_sha256=app.state.sha256,
            predictions=values,
        )

    return app


app = create_app()

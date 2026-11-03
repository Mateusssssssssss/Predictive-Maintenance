"""HTTP inference for the frozen Random Forest reference; no training on requests."""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

REFERENCE_ID = "train_random_forest_20261103T170044Z_3b8896f7"
ROOT = Path(__file__).resolve().parents[2]
FEATURE_MAPPING = {
    "type": "Type",
    "air_temperature_k": "Air temperature [K]",
    "process_temperature_k": "Process temperature [K]",
    "rotational_speed_rpm": "Rotational speed [rpm]",
    "torque_nm": "Torque [Nm]",
    "tool_wear_min": "Tool wear [min]",
}
EXPECTED_FEATURES = [
    "Air temperature [K]", "Process temperature [K]",
    "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]", "Type",
]


class Observation(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    type: Literal["L", "M", "H"]
    air_temperature_k: float = Field(gt=0)
    process_temperature_k: float = Field(gt=0)
    rotational_speed_rpm: float = Field(gt=0)
    torque_nm: float = Field(ge=0)
    tool_wear_min: float = Field(ge=0)


class BatchRequest(BaseModel):
    observations: list[Observation] = Field(min_length=1, max_length=100)


class Prediction(BaseModel):
    model_id: str
    failure_probability: float
    failure_predicted: bool
    threshold: float


def artifact_paths() -> tuple[Path, Path]:
    model = Path(os.environ.get("PM_MODEL_PATH", str(ROOT / "models" / "api_reference.joblib")))
    metadata = Path(os.environ.get("PM_MODEL_METADATA_PATH", str(model.with_suffix(".json"))))
    return model, metadata


@lru_cache(maxsize=1)
def get_reference() -> tuple[object, dict]:
    model_path, metadata_path = artifact_paths()
    if not model_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(
            "Artefato indisponível: execute scripts/export_reference_model.py "
            "ou configure PM_MODEL_PATH e PM_MODEL_METADATA_PATH."
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("reference_run") != REFERENCE_ID:
        raise ValueError("Proveniência de modelo incompatível.")
    if metadata.get("feature_names") != EXPECTED_FEATURES:
        raise ValueError("Esquema do modelo incompatível.")
    from predictive_maintenance.data import sha256

    if sha256(model_path) != metadata.get("artifact_sha256"):
        raise ValueError("Hash do artefato não confere com manifesto.")
    # joblib/pickle executa código ao carregar: só aceitar artefatos próprios confiáveis.
    return joblib.load(model_path), metadata


def prediction_frame(items: list[Observation]) -> pd.DataFrame:
    records = [
        {target: getattr(item, field) for field, target in FEATURE_MAPPING.items()}
        for item in items
    ]
    return pd.DataFrame(records, columns=EXPECTED_FEATURES)


def evaluate(items: list[Observation]) -> list[Prediction]:
    try:
        model, manifest = get_reference()
    except (FileNotFoundError, ValueError, OSError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    scores = np.asarray(model.predict_proba(prediction_frame(items)), dtype=float)[:, 1]
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise HTTPException(status_code=503, detail="Modelo retornou probabilidades inválidas.")
    threshold = float(manifest["threshold"])
    return [
        Prediction(
            model_id=REFERENCE_ID,
            failure_probability=float(probability),
            failure_predicted=bool(probability >= threshold),
            threshold=threshold,
        )
        for probability in scores
    ]


app = FastAPI(
    title="Predictive Maintenance — Reference Random Forest",
    version="0.1.0",
    description=(
        "Classificação exploratória de falha com sensores estáticos. "
        "Não comprova previsão antecipada nem segurança operacional."
    ),
)


@app.get("/health")
def health():
    try:
        _, manifest = get_reference()
    except (FileNotFoundError, ValueError, OSError) as exc:
        return {"status": "unavailable", "reason": str(exc)}
    return {"status": "ready", "model_id": manifest["reference_run"]}


@app.post("/predict", response_model=Prediction)
def predict(observation: Observation):
    return evaluate([observation])[0]


@app.post("/predict/batch", response_model=list[Prediction])
def predict_batch(batch: BatchRequest):
    return evaluate(batch.observations)

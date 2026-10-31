"""Leitura validada e separação determinística, sem aprender padrões do TEST."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def find_root(start: Path | None = None) -> Path:
    """Resolve a raiz ao executar da pasta do projeto ou de notebooks."""
    origin = (start or Path.cwd()).resolve()
    for candidate in (origin, *origin.parents):
        if (candidate / "configs/data.json").is_file() and (candidate / "pyproject.toml").is_file():
            return candidate
    raise FileNotFoundError("Raiz não encontrada; execute dentro do projeto.")


def load_config(root: Path) -> dict:
    return json.loads((root / "configs/data.json").read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_data(root: Path, config: dict) -> pd.DataFrame:
    predictors = set(config["numeric_features"] + config["categorical_features"])
    forbidden = set(config["identifiers"] + config["failure_modes"] + [config["target"]])
    if predictors & forbidden:
        raise ValueError("Preditores contêm identificadores ou rótulos com vazamento.")
    path = root / config["source"]
    if not path.is_file():
        raise FileNotFoundError(f"Coloque o CSV original em {path}.")
    frame = pd.read_csv(path)
    required = (
        config["numeric_features"]
        + config["categorical_features"]
        + config["identifiers"]
        + config["failure_modes"]
        + [config["target"]]
    )
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"Colunas obrigatórias ausentes: {missing}")
    for column in ["UDI", *config["numeric_features"], *config["failure_modes"], config["target"]]:
        if not pd.api.types.is_numeric_dtype(frame[column]):
            raise ValueError(f"Coluna numérica inválida: {column}")
        if not np.isfinite(frame[column].dropna()).all():
            raise ValueError(f"Valores infinitos em {column}")
    for column in [config["target"], *config["failure_modes"]]:
        if frame[column].isna().any() or not frame[column].isin([0, 1]).all():
            raise ValueError(f"Rótulo binário ausente ou inválido: {column}")
    if frame["UDI"].isna().any() or frame["UDI"].duplicated().any():
        raise ValueError("UDI deve ser preenchido e único para separar os blocos.")
    invalid_types = set(frame["Type"].dropna()) - {"L", "M", "H"}
    if invalid_types:
        raise ValueError(f"Tipos de produto inesperados: {sorted(invalid_types)}")
    return frame


def development_split(frame: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reserva o bloco final em UDI; proporção é uma decisão inicial documentada."""
    if config["split_strategy"] != "ordered_prefix":
        raise ValueError("Estratégia de split não implementada.")
    fraction = config["development_fraction"]
    if not 0 < fraction < 1:
        raise ValueError("development_fraction deve estar entre 0 e 1.")
    ordered = frame.sort_values("UDI", kind="stable")
    boundary = int(len(ordered) * fraction)
    if boundary == 0 or boundary == len(ordered):
        raise ValueError("Dados insuficientes para dois conjuntos não vazios.")
    train, test = ordered.iloc[:boundary].copy(), ordered.iloc[boundary:].copy()
    features = config["numeric_features"] + config["categorical_features"]
    # Igualdade exata entre conjuntos tornaria a avaliação futura otimista.
    train_hashes = pd.util.hash_pandas_object(train[features], index=False)
    test_hashes = pd.util.hash_pandas_object(test[features], index=False)
    if set(train_hashes) & set(test_hashes):
        raise ValueError("Features duplicadas entre TRAIN e TEST; revisar separação por grupos.")
    return train, test

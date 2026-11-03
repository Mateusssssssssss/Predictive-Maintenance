"""Reconstruct the selected reference recipe without tuning or touching TEST."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator

from predictive_maintenance.advanced import build_pipeline
from predictive_maintenance.api import EXPECTED_FEATURES, REFERENCE_ID
from predictive_maintenance.data import load_config, load_data, sha256
from predictive_maintenance.evaluation import binary_metrics

ROOT = Path(__file__).resolve().parents[1]
PARAMS = {
    "balancing_strategy": "none",
    "n_estimators": 150,
    "max_depth": 6,
    "min_samples_leaf": 1,
    "max_features": 0.7,
    "criterion": "gini",
    "max_samples": 1.0,
}
EXPECTED = {"tp": 26, "fp": 0, "fn": 9, "recall": 26 / 35, "precision": 1.0}


def main():
    config = load_config(ROOT)
    raw = load_data(ROOT, config).sort_values("UDI", kind="stable")
    # Explicitly forbid using reserved TEST rows: all slices below are < 8001.
    fit = raw.iloc[:4800].copy()
    calibration = raw.iloc[4800:5600].copy()
    validation = raw.iloc[6400:8000].copy()
    del raw
    features = config["numeric_features"] + config["categorical_features"]
    if features != EXPECTED_FEATURES[:-1] + ["Type"]:
        raise ValueError("Data schema changed; refusing reconstruction.")
    base = build_pipeline(config, "random_forest", PARAMS, seed=42)
    base.fit(fit[features], fit[config["target"]])
    calibrated = CalibratedClassifierCV(FrozenEstimator(base), method="sigmoid")
    calibrated.fit(calibration[features], calibration[config["target"]])
    scores = calibrated.predict_proba(validation[features])[:, 1]
    observed = binary_metrics(validation[config["target"]], scores, threshold=0.5)
    for name, value in EXPECTED.items():
        if not np.isclose(observed[name], value, rtol=0, atol=1e-10):
            raise RuntimeError(
                f"Não foi possível reproduzir {name}: {observed[name]} != {value}; "
                "não publicar modelo sem investigar dependências e dados."
            )
    # Does NOT alter historical runs or their stored artifacts.
    out = ROOT / "models" / "api_reference.joblib"
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(calibrated, out)
    meta = {
        "reference_run": REFERENCE_ID,
        "artifact_type": "reconstructed-calibrated-classifier",
        "source_experiment": "GitHub Actions 41679733671",
        "note": "Reconstrução determinística da receita histórica; binário não é o arquivo original.",
        "threshold": 0.5,
        "feature_names": EXPECTED_FEATURES,
        "fit_rows": 4800,
        "calibration_rows": 800,
        "validation_rows": 1600,
        "test_evaluated_by_this_script": False,
        "parameters": PARAMS,
        "source_dataset_sha256": sha256(ROOT / config["source"]),
        "validation_metrics": {
            "tp": observed["tp"], "fp": observed["fp"], "fn": observed["fn"],
            "precision": observed["precision"], "recall": observed["recall"],
            "f1": observed["f1"], "ap": observed["ap"],
        },
        "artifact_sha256": sha256(out),
    }
    (out.with_suffix(".json")).write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(meta["validation_metrics"], indent=2))
    print("Artefato pronto:", out)


if __name__ == "__main__":
    main()

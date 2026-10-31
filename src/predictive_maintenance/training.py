"""GridSearchCV no prefixo de TRAIN; validação posterior, sem avaliar TEST."""

import hashlib
import importlib.metadata
import json
import platform
import time
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, ParameterGrid, TimeSeriesSplit, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from predictive_maintenance.data import development_split, load_config, load_data, sha256
from predictive_maintenance.evaluation import binary_metrics, segment_metrics

SCORE_NAMES = ("ap", "roc_auc", "f1", "recall", "precision", "balanced_accuracy")


def score_at_threshold(y, scores, metric: str, threshold: float) -> float:
    functions = {
        "f1": f1_score,
        "recall": recall_score,
        "precision": precision_score,
        "balanced_accuracy": balanced_accuracy_score,
    }
    kwargs = {} if metric == "balanced_accuracy" else {"zero_division": 0}
    return functions[metric](y, np.asarray(scores) >= threshold, **kwargs)


def make_scoring(threshold: float) -> dict:
    return {
        "ap": "average_precision",
        "roc_auc": "roc_auc",
        **{
            name: make_scorer(
                score_at_threshold,
                response_method="predict_proba",
                metric=name,
                threshold=threshold,
            )
            for name in SCORE_NAMES[2:]
        },
    }


def read_training_config(root: Path) -> dict:
    return json.loads((root / "configs/training.json").read_text(encoding="utf-8"))


def ordered_folds(X, y, n_splits: int, gap: int = 0) -> list:
    """Índices posicionais crescentes; não interpreta UDI como data real."""
    folds = list(TimeSeriesSplit(n_splits=n_splits, gap=gap).split(X))
    for fit, valid in folds:
        if len(np.unique(np.asarray(y)[fit])) != 2 or len(np.unique(np.asarray(y)[valid])) != 2:
            raise ValueError("Fold sem as duas classes; revisar protocolo antes da busca.")
        if fit.max() >= valid.min():
            raise ValueError("Ordem dos folds inválida.")
    return folds


def make_pipeline(data_config: dict, estimator) -> Pipeline:
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    category = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    # Árvores não exigem scaling. Imputação e encoder são ajustados em cada fold.
    preprocessor = ColumnTransformer(
        [
            ("numeric", numeric, data_config["numeric_features"]),
            ("category", category, data_config["categorical_features"]),
        ],
        remainder="drop",
    )
    return Pipeline([("preprocess", preprocessor), ("model", estimator)])


def latest_run(root: Path, model_name: str) -> Path:
    paths = sorted((root / "reports/experiments").glob(f"train_{model_name}_*/manifest.json"))
    if not paths:
        raise FileNotFoundError(f"Sem resultado de {model_name}. Execute scripts/train_models.py.")
    return paths[-1].parent


def run_experiment(root: Path, model_name: str) -> Path:
    if model_name not in ("random_forest", "xgboost"):
        raise ValueError("Modelo deve ser random_forest ou xgboost.")
    config, settings = load_config(root), read_training_config(root)
    development, reserved_test = development_split(load_data(root, config), config)
    boundary = int(len(development) * settings["fit_fraction_of_train"])
    if not 0 < boundary < len(development):
        raise ValueError("Fração de ajuste deve produzir ajuste e validação não vazios.")
    fit_frame, val_frame = development.iloc[:boundary].copy(), development.iloc[boundary:].copy()
    features = config["numeric_features"] + config["categorical_features"]
    X, y = fit_frame[features], fit_frame[config["target"]]
    Xv, yv = val_frame[features], val_frame[config["target"]]
    if yv.nunique() != 2:
        raise ValueError("Validação sem duas classes; revisar protocolo.")
    # Impedir entradas exatamente repetidas entre ajuste e validação.
    if set(pd.util.hash_pandas_object(X, index=False)) & set(
        pd.util.hash_pandas_object(Xv, index=False)
    ):
        raise ValueError("Duplicação entre ajuste e validação; revisar grupos.")
    folds = ordered_folds(X, y, settings["cv_splits"], settings["cv_gap"])
    specification = settings[model_name]
    factory = RandomForestClassifier if model_name == "random_forest" else XGBClassifier
    estimator = factory(random_state=settings["seed"], **specification["fixed"])
    pipeline = make_pipeline(config, estimator)
    scoring = make_scoring(settings["threshold"])
    candidates = len(ParameterGrid(specification["grid"]))
    print(
        f"{model_name}: {candidates} candidatos × {len(folds)} folds = "
        f"{candidates * len(folds)} fits de busca + 1 refit; baseline separado.",
        flush=True,
    )
    # Não há eval_set de TEST nem early stopping que reutilize a validação externa.
    search = GridSearchCV(
        pipeline,
        specification["grid"],
        scoring=scoring,
        refit=settings["refit_metric"],
        cv=folds,
        n_jobs=settings["search_n_jobs"],
        pre_dispatch=settings["search_n_jobs"],
        return_train_score=True,
        error_score="raise",
        verbose=1,
    )
    started = time.perf_counter()
    search.fit(X, y)
    search_seconds = time.perf_counter() - started
    baseline = make_pipeline(config, DummyClassifier(strategy="prior"))
    baseline_cv = cross_validate(
        baseline,
        X,
        y,
        scoring=scoring,
        cv=folds,
        return_train_score=True,
        error_score="raise",
        n_jobs=1,
    )
    baseline.fit(X, y)
    best = search.best_estimator_
    rid = (
        f"train_{model_name}_"
        + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        + "_"
        + uuid4().hex[:8]
    )
    experiment = root / "reports/experiments" / rid
    tables = root / "reports/metrics" / rid
    model_dir = root / "models" / rid
    for directory in (experiment, tables, model_dir):
        directory.mkdir(parents=True, exist_ok=False)
    grid_results = pd.DataFrame(search.cv_results_)
    grid_results["params"] = grid_results["params"].map(lambda v: json.dumps(v, sort_keys=True))
    grid_results.to_csv(tables / "grid_results.csv", index=False)
    fold_rows, membership = [], []
    for number, (train_indices, valid_indices) in enumerate(folds):
        for name, source in ((model_name, search.cv_results_), ("dummy_prior", baseline_cv)):
            row = {
                "model": name,
                "fold": number + 1,
                "origin": "CV_SELECTION" if name == model_name else "CV_BASELINE",
                "fit_n": len(train_indices),
                "validation_n": len(valid_indices),
                "fit_positives": int(y.iloc[train_indices].sum()),
                "validation_positives": int(y.iloc[valid_indices].sum()),
                "threshold": settings["threshold"],
            }
            for metric in SCORE_NAMES:
                for partition in ("train", "test"):
                    value = (
                        source[f"split{number}_{partition}_{metric}"][search.best_index_]
                        if name == model_name
                        else source[f"{partition}_{metric}"][number]
                    )
                    row[f"{'fit' if partition == 'train' else 'validation'}_{metric}"] = value
            fold_rows.append(row)
        for role, indices in (("FIT", train_indices), ("VALIDATION", valid_indices)):
            membership.append(
                pd.DataFrame(
                    {
                        "fold": number + 1,
                        "role": role,
                        "UDI": fit_frame.iloc[indices]["UDI"].to_numpy(),
                    }
                )
            )
    pd.DataFrame(fold_rows).to_csv(tables / "fold_metrics.csv", index=False)
    pd.concat(membership).to_csv(experiment / "cv_membership.csv", index=False)
    pd.concat(
        [
            fit_frame[["UDI"]].assign(partition="FIT"),
            val_frame[["UDI"]].assign(partition="VALIDATION"),
            reserved_test[["UDI"]].assign(partition="TEST_RESERVED"),
        ]
    ).to_csv(experiment / "split_membership.csv", index=False)
    del reserved_test
    rows = []
    for name, fitted in ((model_name, best), ("dummy_prior", baseline)):
        for origin, inputs, targets in (
            ("FIT_RESUBSTITUTION", X, y),
            ("VALIDATION_TRAIN_BLOCK", Xv, yv),
        ):
            scores = fitted.predict_proba(inputs)[:, 1]
            rows.append(
                {
                    "model": name,
                    "origin": origin,
                    **binary_metrics(targets, scores, settings["threshold"]),
                }
            )
    pd.DataFrame(rows).to_csv(tables / "model_metrics.csv", index=False)
    predictions = Xv.copy()
    predictions.insert(0, "UDI", val_frame["UDI"])
    predictions["y_true"] = yv.to_numpy()
    predictions["score"] = best.predict_proba(Xv)[:, 1]
    predictions["dummy_score"] = baseline.predict_proba(Xv)[:, 1]
    predictions["prediction"] = (predictions["score"] >= settings["threshold"]).astype(int)
    predictions.to_csv(tables / "validation_predictions.csv", index=False)
    report = classification_report(
        yv, predictions["prediction"], labels=[0, 1], output_dict=True, zero_division=0
    )
    pd.DataFrame(report).T.to_csv(tables / "classification_report.csv")
    thresholds = [
        {
            "origin": "VALIDATION_DIAGNOSTIC_ONLY",
            **binary_metrics(yv, predictions["score"], float(t)),
        }
        for t in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    ]
    pd.DataFrame(thresholds).to_csv(tables / "threshold_diagnostics.csv", index=False)
    segment_metrics(predictions, "Type", settings["threshold"]).to_csv(
        tables / "segment_type.csv", index=False
    )
    bin_tables = []
    for feature in config["numeric_features"]:
        edges = np.unique(np.nanquantile(X[feature], [0, 0.2, 0.4, 0.6, 0.8, 1]))
        if len(edges) >= 2:
            edges[0], edges[-1] = -np.inf, np.inf
            segmented = predictions.assign(segment=pd.cut(predictions[feature], edges))
        else:
            segmented = predictions.assign(segment="constant")
        table = segment_metrics(segmented, "segment", settings["threshold"])
        table["group"] = feature
        bin_tables.append(table)
    pd.concat(bin_tables).to_csv(tables / "segment_numeric.csv", index=False)
    importance = permutation_importance(
        best,
        Xv,
        yv,
        scoring="average_precision",
        n_repeats=settings["permutation_repeats"],
        random_state=settings["seed"],
        n_jobs=1,
    )
    associations = []
    for index, feature in enumerate(features):
        if feature in config["numeric_features"]:
            pearson = X[feature].corr(y, method="pearson")
            spearman = X[feature].corr(y, method="spearman")
            strength, method = abs(spearman), "abs_spearman"
        else:
            contingency = pd.crosstab(X[feature], y)
            chi2 = chi2_contingency(contingency, correction=False)[0]
            strength = np.sqrt(
                chi2 / (contingency.to_numpy().sum() * min(np.array(contingency.shape) - 1))
            )
            pearson, spearman, method = np.nan, np.nan, "cramers_v"
        associations.append(
            {
                "feature": feature,
                "pearson_fit": pearson,
                "spearman_fit": spearman,
                "association_strength_fit": strength,
                "association_method": method,
                "permutation_ap_drop_validation": importance.importances_mean[index],
                "permutation_std": importance.importances_std[index],
            }
        )
    pd.DataFrame(associations).to_csv(tables / "feature_association_importance.csv", index=False)
    # Curva de tamanho com parâmetros congelados, sem nova escolha de hiperparâmetros.
    curve_rows = []
    for fraction in (0.25, 0.5, 0.75, 1.0):
        end = int(len(X) * fraction)
        fitted = best if fraction == 1 else clone(best).fit(X.iloc[:end], y.iloc[:end])
        for origin, inputs, targets in (
            ("FIT_PREFIX", X.iloc[:end], y.iloc[:end]),
            ("VALIDATION_TRAIN_BLOCK", Xv, yv),
        ):
            curve_rows.append(
                {
                    "fit_n": end,
                    "origin": origin,
                    **binary_metrics(
                        targets, fitted.predict_proba(inputs)[:, 1], settings["threshold"]
                    ),
                }
            )
    pd.DataFrame(curve_rows).to_csv(tables / "learning_diagnostic.csv", index=False)
    model_path = model_dir / "pipeline.joblib"
    joblib.dump(best, model_path)
    restored = joblib.load(model_path)
    np.testing.assert_allclose(
        restored.predict_proba(Xv), best.predict_proba(Xv), rtol=0, atol=1e-12
    )
    manifest = {
        "run_id": rid,
        "model": model_name,
        "dataset_sha256": sha256(root / config["source"]),
        "data_config": config,
        "training_config": settings,
        "features": features,
        "best_params": search.best_params_,
        "best_candidate_index": int(search.best_index_),
        "best_cv_ap": float(search.best_score_),
        "selection": "maximum mean CV AP; first candidate breaks exact ties",
        "fixed_params": specification["fixed"],
        "candidates": candidates,
        "cv_search_fits": candidates * len(folds),
        "search_seconds": search_seconds,
        "fit_rows": len(X),
        "validation_rows": len(Xv),
        "test_evaluated": False,
        "threshold": settings["threshold"],
        "threshold_policy": "fixed reference; not tuned",
        "python": platform.python_version(),
        "versions": {
            p: importlib.metadata.version(p)
            for p in ["scikit-learn", "xgboost", "pandas", "numpy", "scipy", "joblib"]
        },
        "model_path": model_path.relative_to(root).as_posix(),
        "model_sha256": sha256(model_path),
        "serialization_verified": True,
        "split_sha256": sha256(experiment / "split_membership.csv"),
        "cv_sha256": sha256(experiment / "cv_membership.csv"),
        "code_sha256": {
            p.relative_to(root).as_posix(): sha256(p)
            for p in sorted((root / "src/predictive_maintenance").glob("*.py"))
        },
        "config_sha256": hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest(),
        "limitations": [
            "UDI is record order, not verified time",
            "no gap or physical machine ID",
            "CV winner scores are selection-optimistic",
            "validation belongs to EDA-explored TRAIN",
            "probability calibration not demonstrated",
            "no final TEST evaluation",
        ],
    }
    (experiment / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"Concluído: {rid}; CV AP={search.best_score_:.4f}; busca={search_seconds:.1f}s", flush=True
    )
    return experiment

"""Optuna, engenharia de variáveis e pós-processamento com blocos disjuntos de TRAIN."""

import copy
import importlib.metadata
import json
import platform
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from uuid import uuid4

import joblib
import numpy as np
import optuna
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report, make_scorer
from sklearn.model_selection import TunedThresholdClassifierCV, cross_validate
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from predictive_maintenance.data import development_split, load_config, load_data, sha256
from predictive_maintenance.evaluation import auprc_score, binary_metrics, segment_metrics
from predictive_maintenance.features import DERIVED_FEATURES, OperationalFeatures
from predictive_maintenance.training import (
    make_pipeline,
    make_sampling_pipeline,
    make_scoring,
    ordered_folds,
)


def split_training_blocks(frame, settings):
    a, b, c = settings["search_end"], settings["calibration_end"], settings["threshold_end"]
    if not 0 < a < b < c < len(frame):
        raise ValueError("Fronteiras devem criar quatro blocos ordenados não vazios.")
    blocks = [
        frame.iloc[:a].copy(),
        frame.iloc[a:b].copy(),
        frame.iloc[b:c].copy(),
        frame.iloc[c:].copy(),
    ]
    if any(left.UDI.max() >= right.UDI.min() for left, right in pairwise(blocks)):
        raise ValueError("Blocos fora da ordem UDI.")
    return blocks


def build_pipeline(config, name, parameters, seed, engineered=True):
    params = dict(parameters)
    strategy = params.pop("balancing_strategy", "smotenc")
    if strategy not in {"none", "weights_only", "smotenc", "smotenc_weights"}:
        raise ValueError(f"Estratégia de balanceamento inválida: {strategy}")
    ratio = params.pop("sampling_strategy", None)
    neighbors = params.pop("k_neighbors", 3)
    weight = params.pop("positive_weight", 1)
    uses_smotenc = strategy in {"smotenc", "smotenc_weights"}
    uses_weights = strategy in {"weights_only", "smotenc_weights"}
    if uses_smotenc and ratio is None:
        raise ValueError("SMOTENC requer sampling_strategy.")
    if not uses_weights:
        weight = 1
    local = copy.deepcopy(config)
    if engineered:
        local["numeric_features"] += DERIVED_FEATURES
    if name == "xgboost":
        model = XGBClassifier(
            **params,
            scale_pos_weight=weight,
            random_state=seed,
            n_jobs=1,
            tree_method="hist",
            objective="binary:logistic",
            eval_metric="logloss",
        )
    else:
        model = RandomForestClassifier(
            **params,
            class_weight=None if weight == 1 else {0: 1, 1: weight},
            random_state=seed,
            n_jobs=1,
            bootstrap=True,
        )
    sampler_pipeline = make_sampling_pipeline(local, model, seed, neighbors)
    if uses_smotenc:
        sampler_pipeline.set_params(sampler__sampling_strategy=ratio)
    else:
        sampler_pipeline.set_params(sampler="passthrough")
    return Pipeline(
        [("features", OperationalFeatures(enabled=engineered)), ("classifier", sampler_pipeline)]
    )


def sample_parameters(trial, name, space):
    common = space["common"]
    strategy = trial.suggest_categorical("balancing_strategy", common["balancing_strategy"])
    parameters = {"balancing_strategy": strategy}
    if strategy in {"smotenc", "smotenc_weights"}:
        parameters["sampling_strategy"] = trial.suggest_categorical(
            "sampling_strategy", common["sampling_strategy"]
        )
        parameters["k_neighbors"] = trial.suggest_categorical(
            "k_neighbors", common["k_neighbors"]
        )
    if strategy in {"weights_only", "smotenc_weights"}:
        parameters["positive_weight"] = trial.suggest_categorical(
            "positive_weight", common["positive_weight"]
        )
    for key, values in space[name].items():
        parameters[key] = (
            trial.suggest_float(key, *values, log=True)
            if key == "learning_rate"
            else trial.suggest_categorical(key, values)
        )
    return parameters


def fit_postprocessors(base, X_calibration, y_calibration, X_threshold, y_threshold, settings):
    # FrozenEstimator impede reajustar o modelo nos blocos de calibração/limiar.
    calibrated = CalibratedClassifierCV(
        FrozenEstimator(base), method=settings["calibration_method"]
    )
    calibrated.fit(X_calibration, y_calibration)
    threshold_scores = calibrated.predict_proba(X_threshold)[:, 1]
    thresholds = np.unique(np.r_[0.0, threshold_scores, 0.5, 1.0])
    tuned = TunedThresholdClassifierCV(
        FrozenEstimator(calibrated),
        scoring=settings["threshold_objective"],
        response_method="predict_proba",
        thresholds=thresholds,
        cv="prefit",
        refit=False,
        store_cv_results=True,
    )
    tuned.fit(X_threshold, y_threshold)
    return calibrated, tuned


def cv_rows(result, name, folds, y):
    records = []
    for number, (fit, valid) in enumerate(folds):
        row = {
            "model": name,
            "fold": number + 1,
            "origin": "CV_SELECTION" if name != "dummy_prior" else "CV_BASELINE",
            "fit_n": len(fit),
            "validation_n": len(valid),
            "fit_positives": int(y.iloc[fit].sum()),
            "validation_positives": int(y.iloc[valid].sum()),
            "threshold": 0.5,
        }
        for key, values in result.items():
            if key.startswith(("train_", "test_")):
                prefix, metric = key.split("_", 1)
                row[f"{'fit' if prefix == 'train' else 'validation'}_{metric}"] = values[number]
        records.append(row)
    return records


def run_advanced(root: Path, name: str):
    config = load_config(root)
    settings = json.loads((root / "configs/advanced.json").read_text(encoding="utf-8"))
    previous_paths = sorted((root / "reports/experiments").glob(f"train_{name}_*/manifest.json"))
    previous = (
        json.loads(previous_paths[-1].read_text(encoding="utf-8")) if previous_paths else None
    )
    development, reserved = development_split(load_data(root, config), config)
    fit, cal, threshold_block, val = split_training_blocks(development, settings)
    features = config["numeric_features"] + config["categorical_features"]
    target = config["target"]
    for frame in [fit, cal, threshold_block, val]:
        if frame[target].nunique() != 2:
            raise ValueError("Bloco sem as duas classes; revisar fronteiras.")
    X, y = fit[features], fit[target]
    Xv, yv = val[features], val[target]
    folds = ordered_folds(X, y, settings["cv_splits"], settings["cv_gap"])
    metric = settings["selection_metrics"][name]
    scoring = make_scoring(0.5)
    rid = f"train_{name}_" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
    experiment, tables, model_dir = [
        root / path / rid for path in ["reports/experiments", "reports/metrics", "models"]
    ]
    for path in [experiment, tables, model_dir]:
        path.mkdir(parents=True, exist_ok=False)
    membership = pd.concat(
        [
            frame[["UDI"]].assign(partition=role)
            for frame, role in zip(
                [fit, cal, threshold_block, val, reserved],
                ["FIT_SEARCH", "CALIBRATION", "THRESHOLD", "VALIDATION", "TEST_RESERVED"],
            )
        ]
    )
    membership.to_csv(experiment / "split_membership.csv", index=False)
    del reserved, development
    cv_membership = pd.concat(
        [
            pd.DataFrame(
                {"fold": number + 1, "role": role, "UDI": fit.iloc[indices].UDI.to_numpy()}
            )
            for number, (train, valid) in enumerate(folds)
            for role, indices in [("FIT", train), ("VALIDATION", valid)]
        ]
    )
    cv_membership.to_csv(experiment / "cv_membership.csv", index=False)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=settings["seed"]),
        pruner=optuna.pruners.NopPruner(),
        storage=f"sqlite:///{(experiment / 'optuna.sqlite3').as_posix()}",
        study_name=rid,
    )
    # Cobertura mínima dos quatro braços mesmo com orçamento pequeno do Optuna.
    if settings["trials_per_model"] >= 4:
        for strategy in settings["space"]["common"]["balancing_strategy"]:
            study.enqueue_trial({"balancing_strategy": strategy})
    results = {}

    def objective(trial):
        parameters = sample_parameters(trial, name, settings["space"])
        pipeline = build_pipeline(config, name, parameters, settings["seed"])
        result = cross_validate(
            pipeline,
            X,
            y,
            scoring=scoring,
            cv=folds,
            return_train_score=True,
            n_jobs=settings["cv_n_jobs"],
            error_score="raise",
        )
        results[trial.number] = result
        trial.set_user_attr("fold_scores", {k: np.asarray(v).tolist() for k, v in result.items()})
        value = float(np.mean(result[f"test_{metric}"]))
        print(
            f"{name}: trial {trial.number + 1}/{settings['trials_per_model']}, {metric}={value:.4f}",
            flush=True,
        )
        return value

    print(
        f"{name}: {settings['trials_per_model']} trials × {len(folds)} folds; busca apenas em {len(X)} registros.",
        flush=True,
    )
    study.optimize(objective, n_trials=settings["trials_per_model"], n_jobs=1)
    study.trials_dataframe().to_csv(tables / "optuna_trials.csv", index=False)
    best_trial = study.best_trial
    parameters = best_trial.params
    best = build_pipeline(config, name, parameters, settings["seed"]).fit(X, y)
    # Ablação pareada de features com receita congelada, somente em CV.
    no_features = build_pipeline(config, name, parameters, settings["seed"], engineered=False)
    ablation_result = cross_validate(
        no_features,
        X,
        y,
        scoring=scoring,
        cv=folds,
        return_train_score=True,
        n_jobs=settings["cv_n_jobs"],
        error_score="raise",
    )
    baseline = make_pipeline(config, DummyClassifier(strategy="prior"))
    baseline_result = cross_validate(
        baseline,
        X,
        y,
        scoring=scoring,
        cv=folds,
        return_train_score=True,
        n_jobs=1,
        error_score="raise",
    )
    baseline.fit(X, y)
    pd.DataFrame(
        cv_rows(results[best_trial.number], name, folds, y)
        + cv_rows(baseline_result, "dummy_prior", folds, y)
    ).to_csv(tables / "fold_metrics.csv", index=False)
    ablation = []
    for label, result in [
        ("SEM_DERIVADAS", ablation_result),
        ("COM_DERIVADAS", results[best_trial.number]),
    ]:
        for number in range(len(folds)):
            ablation.append(
                {
                    "version": label,
                    "fold": number + 1,
                    **{m: result[f"test_{m}"][number] for m in scoring},
                }
            )
    pd.DataFrame(ablation).to_csv(tables / "feature_ablation.csv", index=False)
    grid_rows = []
    for trial in study.trials:
        result = results[trial.number]
        row = {
            "candidate_index": trial.number,
            "params": json.dumps(trial.params),
            "mean_fit_time": np.mean(result["fit_time"]),
        }
        for m in scoring:
            for prefix in ["test", "train"]:
                row[f"mean_{prefix}_{m}"] = np.mean(result[f"{prefix}_{m}"])
                row[f"std_{prefix}_{m}"] = np.std(result[f"{prefix}_{m}"])
        for key, value in trial.params.items():
            row[f"param_model__{key}"] = value
        grid_rows.append(row)
    grid = pd.DataFrame(grid_rows)
    for m in scoring:
        grid[f"rank_test_{m}"] = (
            grid[f"mean_test_{m}"].rank(method="min", ascending=False).astype(int)
        )
    grid.to_csv(tables / "grid_results.csv", index=False)
    calibrated, tuned = fit_postprocessors(
        best,
        cal[features],
        cal[target],
        threshold_block[features],
        threshold_block[target],
        settings,
    )
    chosen_threshold = float(tuned.best_threshold_)
    threshold_scores = tuned.predict_proba(threshold_block[features])[:, 1]
    np.testing.assert_array_equal(
        tuned.predict(threshold_block[features]), (threshold_scores >= chosen_threshold).astype(int)
    )
    pd.DataFrame(tuned.cv_results_).to_csv(tables / "threshold_selection.csv", index=False)
    stage_rows = []
    for label, model, decision in [
        ("OPTUNA_FEATURES", best, 0.5),
        ("CALIBRATED_REFERENCE", calibrated, 0.5),
        ("CALIBRATED_TUNED", tuned, chosen_threshold),
    ]:
        stage_rows.append(
            {
                "model": name,
                "version": label,
                "origin": "VALIDATION_TRAIN_BLOCK",
                **binary_metrics(yv, model.predict_proba(Xv)[:, 1], decision),
            }
        )
    pd.DataFrame(stage_rows).to_csv(tables / "stage_metrics.csv", index=False)
    metrics = []
    for label, model, decision in [(name, tuned, chosen_threshold), ("dummy_prior", baseline, 0.5)]:
        for origin, inputs, targets in [
            ("FIT_RESUBSTITUTION", X, y),
            ("VALIDATION_TRAIN_BLOCK", Xv, yv),
        ]:
            metrics.append(
                {
                    "model": label,
                    "origin": origin,
                    **binary_metrics(targets, model.predict_proba(inputs)[:, 1], decision),
                }
            )
    pd.DataFrame(metrics).to_csv(tables / "model_metrics.csv", index=False)
    predictions = Xv.copy()
    predictions.insert(0, "UDI", val.UDI)
    predictions["y_true"] = yv.to_numpy()
    predictions["score"] = tuned.predict_proba(Xv)[:, 1]
    predictions["dummy_score"] = baseline.predict_proba(Xv)[:, 1]
    predictions["prediction"] = (predictions.score >= chosen_threshold).astype(int)
    predictions.to_csv(tables / "validation_predictions.csv", index=False)
    pd.DataFrame(
        classification_report(
            yv, predictions.prediction, labels=[0, 1], output_dict=True, zero_division=0
        )
    ).T.to_csv(tables / "classification_report.csv")
    pd.DataFrame(
        [
            {
                "origin": "VALIDATION_DIAGNOSTIC_ONLY",
                **binary_metrics(yv, predictions.score, float(t)),
            }
            for t in sorted({0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9, chosen_threshold})
        ]
    ).to_csv(tables / "threshold_diagnostics.csv", index=False)
    segment_metrics(predictions, "Type", chosen_threshold).to_csv(
        tables / "segment_type.csv", index=False
    )
    segments = []
    for feature in config["numeric_features"]:
        edges = np.unique(np.nanquantile(X[feature], [0, 0.2, 0.4, 0.6, 0.8, 1]))
        edges[0], edges[-1] = -np.inf, np.inf
        table = segment_metrics(
            predictions.assign(segment=pd.cut(predictions[feature], edges)),
            "segment",
            chosen_threshold,
        )
        table["group"] = feature
        segments.append(table)
    pd.concat(segments).to_csv(tables / "segment_numeric.csv", index=False)
    importance = permutation_importance(
        tuned,
        Xv,
        yv,
        scoring={
            "ap": "average_precision",
            "auprc": make_scorer(auprc_score, response_method="predict_proba"),
        },
        n_repeats=settings["permutation_repeats"],
        random_state=settings["seed"],
        n_jobs=1,
    )
    association = []
    for index, feature in enumerate(features):
        if feature in config["numeric_features"]:
            pearson, spearman = X[feature].corr(y), X[feature].corr(y, method="spearman")
            strength, method = abs(spearman), "abs_spearman"
        else:
            contingency = pd.crosstab(X[feature], y)
            strength = np.sqrt(
                chi2_contingency(contingency, correction=False)[0]
                / (len(X) * min(np.array(contingency.shape) - 1))
            )
            pearson, spearman, method = np.nan, np.nan, "cramers_v"
        association.append(
            {
                "feature": feature,
                "pearson_fit": pearson,
                "spearman_fit": spearman,
                "association_strength_fit": strength,
                "association_method": method,
                "permutation_ap_drop_validation": importance["ap"].importances_mean[index],
                "permutation_std": importance["ap"].importances_std[index],
                "permutation_auprc_drop_validation": importance["auprc"].importances_mean[index],
                "permutation_auprc_std": importance["auprc"].importances_std[index],
            }
        )
    pd.DataFrame(association).to_csv(tables / "feature_association_importance.csv", index=False)
    sampler = best["classifier"]["sampler"]
    counts = pd.Series(y).value_counts().to_dict()
    before_counts = sampler.before_counts_ if sampler != "passthrough" else counts
    after_counts = sampler.after_counts_ if sampler != "passthrough" else counts
    pd.DataFrame(
        [
            {
                "origin": "REFIT_SEARCH",
                "strategy": parameters.get("balancing_strategy", "smotenc"),
                "majority_before": before_counts.get(0, 0),
                "minority_before": before_counts.get(1, 0),
                "majority_after": after_counts.get(0, 0),
                "minority_after": after_counts.get(1, 0),
                "synthetic_created": after_counts.get(1, 0) - before_counts.get(1, 0),
            }
        ]
    ).to_csv(tables / "sampling_counts.csv", index=False)
    model_path = model_dir / "pipeline.joblib"
    joblib.dump(tuned, model_path)
    restored = joblib.load(model_path)
    np.testing.assert_allclose(
        restored.predict_proba(Xv), tuned.predict_proba(Xv), rtol=0, atol=1e-12
    )
    np.testing.assert_array_equal(restored.predict(Xv), predictions.prediction)
    comparison = []
    # Um experimento histórico de outro CSV não é comparável: preservá-lo,
    # sem abortar o treinamento atual nem combinar métricas incompatíveis.
    comparable_previous = None
    if previous and previous.get("dataset_sha256") == sha256(root / config["source"]):
        old_path = root / "reports/metrics" / previous["run_id"] / "validation_predictions.csv"
        if old_path.is_file():
            old = pd.read_csv(old_path)
            if (
                np.array_equal(old.UDI.to_numpy(), predictions.UDI.to_numpy())
                and np.array_equal(old.y_true, predictions.y_true)
            ):
                comparable_previous = previous
                comparison.append(
                    {
                        "model": name,
                        "version": "Anterior",
                        "origin": "VALIDATION_TRAIN_BLOCK",
                        **binary_metrics(old.y_true, old.score, previous["threshold"]),
                    }
                )
    comparison.append(
        {
            "model": name,
            "version": "Atual",
            "origin": "VALIDATION_TRAIN_BLOCK",
            **binary_metrics(yv, predictions.score, chosen_threshold),
        }
    )
    pd.DataFrame(comparison).to_csv(tables / "previous_comparison.csv", index=False)
    manifest = {
        "run_id": rid,
        "model": name,
        "experiment_type": "OPTUNA_FEATURES_CALIBRATION_THRESHOLD",
        "previous_run": comparable_previous["run_id"] if comparable_previous else None,
        "previous_comparison_skipped": bool(previous and not comparable_previous),
        "settings": settings,
        "dataset_sha256": sha256(root / config["source"]),
        "data_config": config,
        "features": features,
        "derived_features": DERIVED_FEATURES,
        "best_params": parameters,
        "best_candidate_index": best_trial.number,
        "selection_metric": metric,
        "best_cv_score": best_trial.value,
        "best_cv_ap": float(np.mean(results[best_trial.number]["test_ap"])),
        "best_cv_auprc": float(np.mean(results[best_trial.number]["test_auprc"])),
        "threshold": chosen_threshold,
        "threshold_policy": "F1 on disjoint THRESHOLD block; frozen calibrated model",
        "threshold_selection_f1": float(tuned.best_score_),
        "calibration": "sigmoid on disjoint CALIBRATION block",
        "fit_rows": len(fit),
        "calibration_rows": len(cal),
        "threshold_rows": len(threshold_block),
        "validation_rows": len(val),
        "requires_smotenc": False,
        "balancing_strategy": parameters.get("balancing_strategy", "smotenc"),
        "test_evaluated": False,
        "candidates": len(study.trials),
        "cv_search_fits": len(study.trials) * len(folds),
        "model_path": model_path.relative_to(root).as_posix(),
        "model_sha256": sha256(model_path),
        "serialization_verified": True,
        "split_sha256": sha256(experiment / "split_membership.csv"),
        "cv_sha256": sha256(experiment / "cv_membership.csv"),
        "python": platform.python_version(),
        "versions": {
            p: importlib.metadata.version(p)
            for p in [
                "optuna",
                "scikit-learn",
                "xgboost",
                "imbalanced-learn",
                "pandas",
                "numpy",
                "joblib",
            ]
        },
        "code_sha256": {
            p.relative_to(root).as_posix(): sha256(p)
            for p in sorted((root / "src/predictive_maintenance").glob("*.py"))
        },
        "limitations": [
            "Reused development validation, not independent confirmation",
            "Base model fit on 4800 rather than previous 6400; CV not directly comparable",
            "Only 27 calibration positives and 17 threshold positives",
            "Sigmoid fit does not prove calibration on unseen data",
            "Derived synthetic features may not satisfy exact physical identities",
            "UDI not verified time; gap zero; no TEST evaluation",
        ],
    }
    (experiment / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Concluído {rid}; threshold={chosen_threshold:.5f}", flush=True)
    print(
        pd.DataFrame(comparison)[
            ["version", "ap", "auprc", "f1", "recall", "precision", "brier", "threshold"]
        ].to_string(index=False),
        flush=True,
    )
    return experiment

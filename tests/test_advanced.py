"""Contratos para derivadas e blocos independentes de pós-processamento."""

from itertools import pairwise

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from predictive_maintenance.features import DERIVED_FEATURES, OperationalFeatures


def test_derived_features_are_physical_row_operations_and_do_not_mutate_input():
    frame = pd.DataFrame(
        {
            "Air temperature [K]": [300.0, 301.0],
            "Process temperature [K]": [310.0, 315.0],
            "Rotational speed [rpm]": [60.0, 120.0],
            "Torque [Nm]": [10.0, 20.0],
            "Tool wear [min]": [2.0, 3.0],
            "Type": ["L", "H"],
        }
    )
    original = frame.copy(deep=True)
    result = OperationalFeatures().fit_transform(frame)
    pd.testing.assert_frame_equal(original, frame)
    np.testing.assert_allclose(result[DERIVED_FEATURES[0]], [10.0, 14.0])
    np.testing.assert_allclose(result[DERIVED_FEATURES[1]], [20 * np.pi, 80 * np.pi])
    np.testing.assert_allclose(result[DERIVED_FEATURES[2]], [20.0, 60.0])
    assert not any(c in result for c in ["Machine failure", "TWF", "UDI", "Product ID"])
    pd.testing.assert_frame_equal(OperationalFeatures(enabled=False).fit_transform(frame), original)


def test_training_auxiliary_blocks_are_disjoint_and_ordered():
    from predictive_maintenance.advanced import split_training_blocks

    frame = pd.DataFrame({"UDI": np.arange(1, 81)})
    blocks = split_training_blocks(
        frame, {"search_end": 48, "calibration_end": 56, "threshold_end": 64}
    )
    assert [len(b) for b in blocks] == [48, 8, 8, 16]
    assert len(set().union(*(set(b.UDI) for b in blocks))) == 80
    assert all(a.UDI.max() < b.UDI.min() for a, b in pairwise(blocks))
    with pytest.raises(ValueError):
        split_training_blocks(frame, {"search_end": 56, "calibration_end": 48, "threshold_end": 64})


def test_frozen_calibration_and_threshold_do_not_refit_base(monkeypatch):
    from predictive_maintenance.advanced import fit_postprocessors

    X = np.linspace(-3, 3, 40).reshape(-1, 1)
    y = (X[:, 0] > 0).astype(int)
    base = LogisticRegression().fit(X, y)
    before = base.coef_.copy()

    def reject_fit(*args, **kwargs):
        raise AssertionError("Modelo base não pode aprender com calibração ou limiar.")

    monkeypatch.setattr(base, "fit", reject_fit)
    X_cal = np.linspace(-2.9, 2.9, 20).reshape(-1, 1)
    X_thr = np.linspace(-2.8, 2.8, 18).reshape(-1, 1)
    calibrated, tuned = fit_postprocessors(
        base,
        X_cal,
        (X_cal[:, 0] > 0.2).astype(int),
        X_thr,
        (X_thr[:, 0] > 0.4).astype(int),
        {"calibration_method": "sigmoid", "threshold_objective": "f1"},
    )
    np.testing.assert_array_equal(base.coef_, before)
    scores = calibrated.predict_proba(X_thr)[:, 1]
    np.testing.assert_array_equal(
        tuned.predict(X_thr), (scores >= tuned.best_threshold_).astype(int)
    )
    assert 0 <= tuned.best_threshold_ <= 1


@pytest.mark.parametrize("name", ["xgboost", "random_forest"])
@pytest.mark.parametrize("strategy", ["none", "weights_only", "smotenc", "smotenc_weights"])
def test_balancing_strategies_use_only_requested_sampler_and_weights(name, strategy):
    from predictive_maintenance.advanced import build_pipeline

    config = {
        "numeric_features": [
            "Air temperature [K]", "Process temperature [K]",
            "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]",
        ],
        "categorical_features": ["Type"],
    }
    model_params = (
        {"n_estimators": 5, "max_depth": 2}
        if name == "xgboost"
        else {"n_estimators": 5, "max_depth": 2}
    )
    pipeline = build_pipeline(
        config, name,
        {
            **model_params,
            "balancing_strategy": strategy,
            "positive_weight": 3,
            "sampling_strategy": 0.5,
            "k_neighbors": 3,
        },
        seed=42,
    )
    sampler = pipeline["classifier"]["sampler"]
    assert (sampler == "passthrough") == (strategy in {"none", "weights_only"})
    model = pipeline["classifier"]["model"]
    if name == "xgboost":
        assert model.scale_pos_weight == (3 if "weights" in strategy else 1)
    else:
        assert model.class_weight == ({0: 1, 1: 3} if "weights" in strategy else None)


def test_optuna_samples_all_four_balancing_branches():
    from predictive_maintenance.advanced import sample_parameters

    class Trial:
        def __init__(self, strategy):
            self.strategy = strategy
            self.calls = []

        def suggest_categorical(self, key, values):
            self.calls.append(key)
            return self.strategy if key == "balancing_strategy" else values[0]

    space = {
        "common": {
            "balancing_strategy": ["none", "weights_only", "smotenc", "smotenc_weights"],
            "sampling_strategy": [0.25], "k_neighbors": [3],
            "positive_weight": [3],
        },
        "random_forest": {"n_estimators": [5]},
    }
    for strategy in space["common"]["balancing_strategy"]:
        trial = Trial(strategy)
        result = sample_parameters(trial, "random_forest", space)
        assert result["balancing_strategy"] == strategy
        assert ("sampling_strategy" in result) == strategy.startswith("smotenc")
        assert ("positive_weight" in result) == ("weights" in strategy)


@pytest.mark.parametrize(
    ("y_true", "y_pred", "expected_min", "expected_max"),
    [
        ([1] * 10 + [0] * 10, [1] * 9 + [0] + [1] + [0] * 9, 2, 4),
        ([1] * 10 + [0] * 10, [1] * 9 + [0] + [1] * 5 + [0] * 5, 1, 2),
        ([1] * 10 + [0] * 10, [1] * 8 + [0] * 12, 0, 1),
    ],
)
def test_recall_target_policy(y_true, y_pred, expected_min, expected_max):
    from predictive_maintenance.advanced import recall_priority_score

    score = recall_priority_score(y_true, y_pred, target_recall=0.9, target_precision=0.8)
    assert expected_min <= score < expected_max


def test_recall_target_policy_prefers_precision_among_feasible_candidates():
    from predictive_maintenance.advanced import recall_priority_score

    truth = np.array([1] * 10 + [0] * 30)
    tight = np.array([1] * 9 + [0] + [1] + [0] * 29)
    loose = np.array([1] * 10 + [1] * 5 + [0] * 25)
    assert recall_priority_score(truth, tight) > recall_priority_score(truth, loose)

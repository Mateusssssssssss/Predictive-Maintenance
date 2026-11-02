"""Contratos de split, limiar e pipeline; não avalia o conjunto TEST."""

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

from predictive_maintenance.evaluation import binary_metrics
from predictive_maintenance.training import make_pipeline, ordered_folds, score_at_threshold


@pytest.mark.parametrize("name", ["xgboost", "random_forest"])
def test_conditional_grid_separates_recipes_and_respects_smotenc_requirement(name):
    from sklearn.model_selection import ParameterGrid

    from predictive_maintenance.training import (
        balance_strategy,
        make_sampling_pipeline,
        make_search_grid,
        select_candidate,
    )

    sampler = make_sampling_pipeline(
        {"numeric_features": ["sensor"], "categorical_features": ["Type"]}, DummyClassifier(), 42, 3
    )["sampler"]
    spec = {
        "grid": {"model__random_state": [42]},
        "balance_search": {"positive_weights": [3, 10], "sampling_ratios": [0.25, 0.5]},
    }
    candidates = list(ParameterGrid(make_search_grid(spec, name, sampler)))
    assert len(candidates) == 9
    assert {balance_strategy(p) for p in candidates} == {
        "NO_BALANCING",
        "WEIGHTS_ONLY",
        "SMOTENC_ONLY",
        "SMOTENC_WEIGHTS",
    }
    scores = np.arange(9, dtype=float) / 10
    scores[0] = 1.0  # Controle acima dos elegíveis não pode vencer refit obrigatório.
    results = {"params": candidates, "mean_test_ap": scores}
    winner = select_candidate(results, "ap", require_smotenc=True)
    assert balance_strategy(candidates[winner]) == "SMOTENC_WEIGHTS"
    assert select_candidate(results, "ap", require_smotenc=False) == 0
    for p in candidates:
        if p["sampler"] == "passthrough":
            assert "sampler__sampling_strategy" not in p
        if name == "random_forest" and p["model__class_weight"]:
            assert p["model__class_weight"][1] > p["model__class_weight"][0]


def test_auprc_uses_scores_and_is_distinct_from_ap():
    metrics = binary_metrics([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8])
    assert metrics["auprc"] == pytest.approx(19 / 24)
    assert metrics["ap"] == pytest.approx(5 / 6)
    assert binary_metrics([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8], 0.2)["auprc"] == metrics["auprc"]


def test_smotenc_is_fit_only_and_preserves_categories(monkeypatch):
    from predictive_maintenance.training import make_sampling_pipeline

    X = pd.DataFrame({"sensor": np.arange(40, dtype=float), "Type": ["L", "H"] * 20})
    y = np.array([1] * 8 + [0] * 32)
    pipe = make_sampling_pipeline(
        {"numeric_features": ["sensor"], "categorical_features": ["Type"]},
        DummyClassifier(strategy="prior"),
        seed=42,
        neighbors=3,
    )
    pipe.set_params(sampler__sampling_strategy=0.5)
    pipe.fit(X, y)
    sampler = pipe["sampler"]
    assert sampler.before_counts_ == {0: 32, 1: 8}
    assert sampler.after_counts_ == {0: 32, 1: 16}

    def reject_resampling(*args, **kwargs):
        raise AssertionError("Resampling durante predict_proba")

    monkeypatch.setattr(sampler, "fit_resample", reject_resampling)
    probabilities = pipe.predict_proba(
        pd.DataFrame({"sensor": [1000.0, np.nan], "Type": ["NEW", "L"]})
    )
    assert probabilities.shape == (2, 2)


def test_threshold_tie_is_positive_and_metrics_use_scores():
    result = binary_metrics([0, 1, 1, 0], [0.1, 0.5, 0.3, 0.8], 0.5)
    assert (result["tn"], result["fp"], result["fn"], result["tp"]) == (1, 1, 1, 1)
    assert result["recall"] == 0.5
    assert score_at_threshold([1, 0], [0.5, 0.49], "recall", 0.5) == 1
    assert result["ap"] == binary_metrics([0, 1, 1, 0], [0.1, 0.5, 0.3, 0.8], 0.2)["ap"]


def test_no_alert_precision_is_undefined():
    assert np.isnan(binary_metrics([0, 1], [0.1, 0.2], 0.5)["precision"])


def test_ordered_folds_never_fit_on_future_rows():
    X = pd.DataFrame({"value": range(40)})
    y = np.tile([0, 1], 20)
    for fit, validation in ordered_folds(X, y, 3, gap=2):
        assert set(fit).isdisjoint(validation)
        assert fit.max() + 2 < validation.min()
    with pytest.raises(ValueError, match="duas classes"):
        ordered_folds(X, np.zeros(40), 3)


def test_imputer_and_encoder_learn_only_from_fit_and_accept_unknown_categories():
    config = {"numeric_features": ["sensor"], "categorical_features": ["Type"]}
    fit = pd.DataFrame({"sensor": [1.0, 3.0, np.nan, 5.0], "Type": ["L", "H", "L", "H"]})
    pipeline = make_pipeline(config, DummyClassifier(strategy="prior"))
    pipeline.fit(fit, [0, 1, 0, 1])
    before = pipeline["preprocess"].named_transformers_["numeric"]["imputer"].statistics_.copy()
    scores = pipeline.predict_proba(
        pd.DataFrame({"sensor": [1000.0, np.nan], "Type": ["NEW", "L"]})
    )
    np.testing.assert_array_equal(before, [3.0])
    np.testing.assert_array_equal(
        before, pipeline["preprocess"].named_transformers_["numeric"]["imputer"].statistics_
    )
    assert scores.shape == (2, 2)
    assert np.isfinite(scores).all()

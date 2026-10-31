"""Contratos de split, limiar e pipeline; não avalia o conjunto TEST."""

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

from predictive_maintenance.evaluation import binary_metrics
from predictive_maintenance.training import make_pipeline, ordered_folds, score_at_threshold


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

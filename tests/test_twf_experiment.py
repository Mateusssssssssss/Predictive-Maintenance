"""Contratos de features TWF sem vazamento e sem alterar o baseline."""
import numpy as np
import pandas as pd
import pytest

from predictive_maintenance.twf_experiment import (
    CANDIDATES, REFERENCE_PARAMS, WearHypothesisFeatures,
    make_candidate, summarize_twf,
)


def data():
    return pd.DataFrame({
        "Air temperature [K]": [300., 301.],
        "Process temperature [K]": [310., 314.],
        "Rotational speed [rpm]": [1200., 1400.],
        "Torque [Nm]": [40., 50.],
        "Tool wear [min]": [100., 200.],
        "Temperature gap [K]": [10., 13.],
        "Type": ["L", "M"],
    })


@pytest.mark.parametrize("mode", CANDIDATES)
def test_hypotheses_add_only_rowwise_sensor_features(mode):
    x = data()
    before = x.copy(deep=True)
    result = WearHypothesisFeatures(mode).fit_transform(x)
    pd.testing.assert_frame_equal(x, before)
    assert not {"TWF", "HDF", "PWF", "OSF", "RNF", "Machine failure"}.intersection(result)
    if mode == "wear_squared":
        np.testing.assert_allclose(result["Wear squared [min2]"], [10000., 40000.])
    elif mode == "wear_context":
        np.testing.assert_allclose(result["Wear rpm [min rpm]"], [120000., 280000.])
        np.testing.assert_allclose(result["Wear heat gap [min K]"], [1000., 2600.])


def test_reference_hyperparameters_are_unchanged_and_sampler_is_disabled():
    config = {
        "numeric_features": [
            "Air temperature [K]", "Process temperature [K]",
            "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]",
        ], "categorical_features": ["Type"],
    }
    pipe = make_candidate(config, "wear_squared")
    rf = pipe["classifier"]["model"]
    assert rf.n_estimators == REFERENCE_PARAMS["n_estimators"]
    assert rf.max_depth == REFERENCE_PARAMS["max_depth"]
    assert pipe["classifier"]["sampler"] == "passthrough"
    assert rf.class_weight is None
    assert rf.random_state == 42


def test_twf_audit_uses_labels_only_after_scores_and_respects_zero_positive():
    summary = summarize_twf([1, 1, 0], [0.8, 0.1, 0.9], [1, 1, 0])
    assert summary["twf_tp"] == 1
    assert summary["twf_fn"] == 1
    assert summary["twf_recall"] == 0.5
    assert np.isnan(summarize_twf([1, 0], [0.9, 0.2], [0, 0])["twf_recall"])

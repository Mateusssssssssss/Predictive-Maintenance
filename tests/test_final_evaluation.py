import numpy as np

from predictive_maintenance.test_evaluation import bootstrap_intervals


def test_bootstrap_reproducible_and_bounded():
    y = np.array([0, 0, 0, 1, 1, 1] * 8)
    scores = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 0.9] * 8)
    first = bootstrap_intervals(y, scores, 0.5, repeats=30, seed=12)
    second = bootstrap_intervals(y, scores, 0.5, repeats=30, seed=12)
    assert first.equals(second)
    assert first.lower_95.between(0, 1).all()
    assert first.upper_95.between(0, 1).all()
    assert (first.lower_95 <= first.upper_95).all()
    assert first.valid_repeats.eq(30).all()

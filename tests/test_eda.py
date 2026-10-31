import pandas as pd

from predictive_maintenance.eda import outcome_rates


def test_outcome_rates_keep_denominators_and_missing_categories():
    frame = pd.DataFrame({"group": ["L", "L", "H", None], "failure": [1, 0, 0, 1]})
    table = outcome_rates(frame, "group", "failure")
    assert table["n"].sum() == len(frame)
    assert table.loc["L", "failure_rate"] == 0.5
    assert table["failures"].sum() == 2

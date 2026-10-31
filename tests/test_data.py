"""Contratos que protegem a separação e evitam uso silencioso de dados inválidos."""

import pandas as pd
import pytest

from predictive_maintenance.data import development_split, find_root, load_config, load_data


def test_ordered_split_is_disjoint_and_has_no_feature_duplicates():
    root = find_root()
    config = load_config(root)
    data = load_data(root, config)
    train, test = development_split(data, config)
    assert len(train) + len(test) == len(data)
    assert train.UDI.max() < test.UDI.min()
    assert not set(train.UDI) & set(test.UDI)


def test_cross_split_duplicates_are_rejected():
    config = {
        "split_strategy": "ordered_prefix",
        "development_fraction": 0.5,
        "numeric_features": ["sensor"],
        "categorical_features": [],
    }
    frame = pd.DataFrame({"UDI": [1, 2, 3, 4], "sensor": [10, 20, 10, 30]})
    with pytest.raises(ValueError, match="duplicadas"):
        development_split(frame, config)


@pytest.mark.parametrize("fraction", [0, 1, -0.2])
def test_invalid_split_fraction_is_rejected(fraction):
    config = {"split_strategy": "ordered_prefix", "development_fraction": fraction}
    with pytest.raises(ValueError, match="entre 0 e 1"):
        development_split(pd.DataFrame({"UDI": [1, 2]}), config)


def test_config_rejects_target_leakage():
    root = find_root()
    config = load_config(root)
    config["numeric_features"].append(config["target"])
    with pytest.raises(ValueError, match="vazamento"):
        load_data(root, config)


@pytest.mark.parametrize("problem", ["missing_column", "invalid_label", "duplicate_id", "infinity"])
def test_schema_rejects_invalid_data(tmp_path, problem):
    root = find_root()
    config = load_config(root)
    frame = pd.read_csv(root / config["source"]).head(5)
    if problem == "missing_column":
        frame = frame.drop(columns=["Torque [Nm]"])
    elif problem == "invalid_label":
        frame.loc[0, config["target"]] = 2
    elif problem == "duplicate_id":
        frame.loc[0, "UDI"] = frame.loc[1, "UDI"]
    else:
        frame.loc[0, "Torque [Nm]"] = float("inf")
    frame.to_csv(tmp_path / config["source"], index=False)
    with pytest.raises(ValueError):
        load_data(tmp_path, config)

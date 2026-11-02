"""Variáveis determinísticas calculadas dos sensores, sem rótulos ou estatísticas globais."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

DERIVED_FEATURES = ["Temperature gap [K]", "Mechanical power [W]", "Wear torque [min Nm]"]


class OperationalFeatures(TransformerMixin, BaseEstimator):
    def __init__(self, enabled=True):
        self.enabled = enabled

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("OperationalFeatures exige DataFrame com nomes de sensores.")
        required = [
            "Air temperature [K]",
            "Process temperature [K]",
            "Rotational speed [rpm]",
            "Torque [Nm]",
            "Tool wear [min]",
        ]
        if not set(required).issubset(X.columns):
            raise ValueError("Sensores necessários às variáveis derivadas estão ausentes.")
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = len(X.columns)
        return self

    def transform(self, X):
        check_is_fitted(self)
        result = X.copy()
        if self.enabled:
            result[DERIVED_FEATURES[0]] = (
                result["Process temperature [K]"] - result["Air temperature [K]"]
            )
            result[DERIVED_FEATURES[1]] = (
                result["Torque [Nm]"] * result["Rotational speed [rpm]"] * (2 * np.pi / 60)
            )
            result[DERIVED_FEATURES[2]] = result["Tool wear [min]"] * result["Torque [Nm]"]
        return result

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self)
        return (
            np.r_[self.feature_names_in_, DERIVED_FEATURES]
            if self.enabled
            else self.feature_names_in_
        )

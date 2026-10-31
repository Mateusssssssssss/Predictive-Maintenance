"""Tabelas descritivas; todas recebem explicitamente o conjunto explorável."""

import pandas as pd


def quality_table(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "dtype": frame.dtypes.astype(str),
            "missing": frame.isna().sum(),
            "missing_fraction": frame.isna().mean(),
            "unique": frame.nunique(),
        }
    ).rename_axis("feature")


def outlier_table(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """IQR sinaliza extremos estatísticos, não erros nem regra de exclusão."""
    rows = []
    for column in columns:
        values = frame[column].dropna()
        q1, q3 = values.quantile([0.25, 0.75])
        lower, upper = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
        rows.append(
            {
                "feature": column,
                "lower_iqr": lower,
                "upper_iqr": upper,
                "flagged": int(((values < lower) | (values > upper)).sum()),
                "negative": int((values < 0).sum()),
            }
        )
    return pd.DataFrame(rows).set_index("feature")


def outcome_rates(frame: pd.DataFrame, group: str, target: str) -> pd.DataFrame:
    table = frame.groupby(group, observed=True, dropna=False)[target].agg(
        n="size", failures="sum", failure_rate="mean"
    )
    table["non_failures"] = table["n"] - table["failures"]
    return table

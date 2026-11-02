"""Métricas com classe positiva 1 e política explícita de limiar."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def auprc_score(y, scores) -> float:
    """Área trapezoidal PR com scores positivos; não equivale a Average Precision."""
    if not np.any(np.asarray(y) == 1):
        return np.nan
    precision, recall, _ = precision_recall_curve(y, scores)
    return float(auc(recall, precision))


def binary_metrics(y, scores, threshold: float = 0.5) -> dict:
    y, scores = np.asarray(y), np.asarray(scores)
    if len(y) == 0 or len(y) != len(scores):
        raise ValueError("Alvos e scores devem ter o mesmo comprimento não vazio.")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(scores).all():
        raise ValueError("Alvos binários e scores finitos são obrigatórios.")
    if (scores < 0).any() or (scores > 1).any() or not 0 <= threshold <= 1:
        raise ValueError("Scores e threshold devem estar em [0, 1].")
    pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    both_classes = len(np.unique(y)) == 2
    return {
        "n": len(y),
        "positives": int(y.sum()),
        "prevalence": float(y.mean()),
        "threshold": threshold,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "f1": f1_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0) if tp + fn else np.nan,
        "precision": precision_score(y, pred, zero_division=0) if tp + fp else np.nan,
        "specificity": tn / (tn + fp) if tn + fp else np.nan,
        "npv": tn / (tn + fn) if tn + fn else np.nan,
        "fpr": fp / (tn + fp) if tn + fp else np.nan,
        "fnr": fn / (tp + fn) if tp + fn else np.nan,
        "alert_rate": float(pred.mean()),
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred) if both_classes else np.nan,
        "ap": average_precision_score(y, scores) if y.sum() else np.nan,
        "auprc": auprc_score(y, scores),
        "roc_auc": roc_auc_score(y, scores) if both_classes else np.nan,
        "brier": brier_score_loss(y, scores),
        "log_loss": log_loss(y, scores, labels=[0, 1]),
        "mcc": matthews_corrcoef(y, pred) if both_classes else np.nan,
        "kappa": cohen_kappa_score(y, pred, labels=[0, 1]) if both_classes else np.nan,
    }


def segment_metrics(frame: pd.DataFrame, group: str, threshold: float) -> pd.DataFrame:
    records = []
    for value, rows in frame.groupby(group, observed=True, dropna=False):
        records.append(
            {
                "group": group,
                "segment": str(value),
                **binary_metrics(rows["y_true"], rows["score"], threshold),
            }
        )
    return pd.DataFrame(records)

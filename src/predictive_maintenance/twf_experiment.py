"""Ablacao exploratoria de features para hipotese TWF, apenas em FIT_SEARCH.

As colunas TWF/HDF/PWF/OSF/RNF sao usadas exclusivamente para diagnostico
posterior a predição. Nenhuma linha de VALIDATION ou TEST entra na busca.
"""
from __future__ import annotations

import copy
import json
from datetime import UTC, datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier

from predictive_maintenance.data import development_split, load_config, load_data
from predictive_maintenance.evaluation import binary_metrics
from predictive_maintenance.features import DERIVED_FEATURES, OperationalFeatures
from predictive_maintenance.training import make_sampling_pipeline, ordered_folds


# Hiperparametros CONGELADOS da run de referencia de 03/11/2026.
# Sem tuning no conjunto de validacao e sem alteracao do modelo de producao.
REFERENCE_PARAMS = {
    "n_estimators": 150, "max_depth": 6, "min_samples_leaf": 1,
    "max_features": 0.7, "criterion": "gini", "max_samples": 1.0,
}
NEW_FEATURES = {
    "wear_squared": ["Wear squared [min2]"],
    "wear_context": ["Wear rpm [min rpm]", "Wear heat gap [min K]"],
}
CANDIDATES = ("baseline", "wear_squared", "wear_context")


class WearHypothesisFeatures(TransformerMixin, BaseEstimator):
    """Feature engineering por linha, sem fit de estatisticas e sem labels."""

    def __init__(self, mode="baseline"):
        self.mode = mode

    def fit(self, X, y=None):
        if self.mode not in CANDIDATES:
            raise ValueError(f"Hipotese nao reconhecida: {self.mode}")
        return self

    def transform(self, X):
        out = X.copy()
        wear = out["Tool wear [min]"]
        if self.mode == "wear_squared":
            out["Wear squared [min2]"] = wear * wear
        elif self.mode == "wear_context":
            out["Wear rpm [min rpm]"] = wear * out["Rotational speed [rpm]"]
            out["Wear heat gap [min K]"] = wear * out["Temperature gap [K]"]
        return out


def make_candidate(config, mode, seed=42):
    if mode not in CANDIDATES:
        raise ValueError(mode)
    local = copy.deepcopy(config)
    local["numeric_features"] += DERIVED_FEATURES + NEW_FEATURES.get(mode, [])
    model = RandomForestClassifier(
        **REFERENCE_PARAMS, random_state=seed, n_jobs=1, bootstrap=True
    )
    classifier = make_sampling_pipeline(local, model, seed, neighbors=3)
    classifier.set_params(sampler="passthrough")
    return Pipeline([
        ("operational", OperationalFeatures(enabled=True)),
        ("wear_hypothesis", WearHypothesisFeatures(mode=mode)),
        ("classifier", classifier),
    ])


def summarize_twf(y, scores, twf, threshold=0.5):
    """TWF usado SOMENTE na auditoria, apos predict_proba, nunca em fit."""
    y = np.asarray(y, dtype=int)
    twf = np.asarray(twf, dtype=int)
    flagged = (y == 1) & (twf == 1)
    predicted = np.asarray(scores) >= threshold
    positive_twf = int(flagged.sum())
    detected_twf = int((flagged & predicted).sum())
    return {
        "twf_positives": positive_twf,
        "twf_tp": detected_twf,
        "twf_fn": positive_twf - detected_twf,
        "twf_recall": (detected_twf / positive_twf if positive_twf else np.nan),
    }


def run_twf_ablation(root: Path, output: Path | None = None):
    config = load_config(root)
    development, reserved_test = development_split(load_data(root, config), config)
    # Eliminar referencias aos demais blocos antes dos fits, evitando uso acidental.
    del reserved_test
    fit = development.iloc[:4800].copy()
    del development
    if fit.UDI.max() > 4800 or len(fit) != 4800:
        raise ValueError("FIT_SEARCH inesperado; nao ampliar blocos automaticamente.")
    features = config["numeric_features"] + config["categorical_features"]
    forbidden = set(config["failure_modes"] + config["identifiers"] + [config["target"]])
    if forbidden.intersection(features) or any(
        forbidden.intersection(NEW_FEATURES.get(c, [])) for c in CANDIDATES
    ):
        raise ValueError("Leakage: coluna auxiliar usada como feature.")
    X = fit[features]
    y = fit[config["target"]]
    folds = ordered_folds(X, y, n_splits=3, gap=0)
    seed = 42
    rows, oof = [], []
    for name in CANDIDATES:
        for fold_id, (train, valid) in enumerate(folds, start=1):
            candidate = make_candidate(config, name, seed)
            candidate.fit(X.iloc[train], y.iloc[train])
            probs = candidate.predict_proba(X.iloc[valid])[:, 1]
            metrics = binary_metrics(y.iloc[valid], probs, threshold=0.5)
            twf_stats = summarize_twf(
                y.iloc[valid], probs, fit.iloc[valid]["TWF"], threshold=0.5
            )
            rows.append({
                "candidate": name, "fold": fold_id,
                "fit_rows": len(train), "validation_rows": len(valid),
                **metrics, **twf_stats,
            })
            oof.append(pd.DataFrame({
                "candidate": name, "fold": fold_id,
                "UDI": fit.iloc[valid].UDI.to_numpy(),
                "y_true": y.iloc[valid].to_numpy(),
                "score": probs,
                "TWF_diagnostic_only": fit.iloc[valid]["TWF"].to_numpy(),
            }))
    result = pd.DataFrame(rows)
    predictions = pd.concat(oof, ignore_index=True)
    output = output or (root / "reports" / "twf_ablation")
    output.mkdir(parents=True, exist_ok=True)
    result.to_csv(output / "fold_metrics.csv", index=False)
    predictions.to_csv(output / "cv_predictions.csv", index=False)
    # OOF apenas dos periodos previstos por modelo que nao os viu, nao
    # reintroduzir os mesmos exemplos na escolha posterior.
    summary = []
    for name, group in predictions.groupby("candidate", sort=False):
        m = binary_metrics(group.y_true, group.score, threshold=0.5)
        t = summarize_twf(group.y_true, group.score, group.TWF_diagnostic_only)
        summary.append({"candidate": name, **m, **t})
    summary = pd.DataFrame(summary)
    summary.to_csv(output / "oof_summary.csv", index=False)
    byfold = result.groupby("candidate")[["ap", "auprc", "f1", "recall",
                                           "precision", "twf_recall"]].agg(["mean", "std"])
    byfold.to_csv(output / "cv_summary.csv")

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(CANDIDATES))
    for j, metric in enumerate(["ap", "recall", "precision", "twf_recall"]):
        vals = [result.loc[result.candidate == name, metric].mean() for name in CANDIDATES]
        ax.bar(x + (j - 1.5) * 0.18, vals, 0.18, label=metric)
    ax.set_xticks(x, CANDIDATES)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Media por fold (0–1)")
    ax.set_title("Ablacao de features TWF — CV apenas no FIT_SEARCH")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "cv_comparison.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    pivot = result.pivot(index="fold", columns="candidate", values="twf_recall")
    pivot.plot(kind="bar", ax=ax, rot=0)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Recall TWF por fold")
    ax.set_title("TWF apenas como rotulo de auditoria")
    fig.tight_layout()
    fig.savefig(output / "twf_by_fold.png", dpi=160)
    plt.close(fig)

    manifest = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "experiment": "TWF_FEATURE_ABLATION_CV_ONLY",
        "reference_run": "train_random_forest_20261103T170044Z_3b8896f7",
        "reference_params": REFERENCE_PARAMS,
        "candidates": list(CANDIDATES),
        "new_features": NEW_FEATURES,
        "fit_search_rows": len(fit),
        "folds": 3,
        "threshold": 0.5,
        "validation_used_for_tuning": False,
        "test_evaluated": False,
        "notes": [
            "TWF and other failure flags are used for diagnostics only, never model inputs.",
            "These CV folds already informed earlier hyperparameter work: exploratory evidence.",
            "RF probabilities are not calibrated in this paired feature ablation; do not compare directly with previously calibrated validation metrics.",
            "No claim that UDI is an established temporal variable.",
        ],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    report = [
        "# Ablação exploratória de features para TWF",
        "",
        "Apenas FIT_SEARCH, 4.800 linhas; 3 folds ordenados idênticos por candidato.",
        "Random Forest e threshold 0,50 congelados; sem SMOTENC, pesos ou recalibração.",
        "TWF é utilizado somente para diagnóstico de erros. VALIDATION e TEST não são usados.",
        "",
        "## Resultados OOF (somente folds de busca; não é avaliação independente)",
        "",
        "| Hipótese | AP | Recall | Precision | F1 | Recall TWF | TP TWF / total TWF |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in summary.itertuples(index=False):
        report.append(
            f"| {r.candidate} | {r.ap:.4f} | {r.recall:.4f} | "
            f"{r.precision:.4f} | {r.f1:.4f} | "
            f"{r.twf_recall:.4f} | {r.twf_tp}/{r.twf_positives} |"
        )
    report += [
        "",
        "## Decisão metodológica",
        "",
        "1. Congelar RF/threshold isola mudanças somente nas features.",
        "2. Features de desgaste ao quadrado testam não linearidade; interações desgaste-rotação e desgaste-gap térmico testam contexto físico mensurável.",
        "3. CV pareada nos mesmos folds, sem tocar VALIDATION/TEST.",
        "4. Avaliar AP (capacidade de ranqueamento), Recall/Precision/F1 (threshold 0,5) e recall TWF como diagnóstico de mecanismo.",
        "5. Não trocar o modelo de referência sem validação independente adicional.",
        "",
        "## Limitações",
        "",
        "- Hipóteses motivadas por análise anterior da validação, portanto exploratórias.",
        "- CV e seus dados não são independentes do histórico de tuning.",
        "- TWF não pode ser usado como entrada (leakage do desfecho).",
        "- UDI não representa comprovadamente tempo real.",
        "- Escassez de positivos por fold e incerteza das métricas.",
    ]
    (output / "README.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(summary[["candidate", "n", "positives", "ap", "recall", "precision",
                   "f1", "twf_positives", "twf_tp", "twf_recall"]].to_string(index=False), flush=True)
    return output

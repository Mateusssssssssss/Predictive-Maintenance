"""Rótulos e formatação de exibição; mantém os artefatos e DataFrames numéricos."""

import re

import numpy as np
import pandas as pd

LABELS = {
    "feature": "Variável",
    "group": "Variável de agrupamento",
    "segment": "Faixa / categoria",
    "bin": "Faixa de valores",
    "size": "Observações",
    "count": "Quantidade",
    "n": "Observações",
    "model": "Receita",
    "source": "Origem",
    "origin": "Origem",
    "threshold": "Limiar de alerta",
    "fold": "Fold ordenado",
    "policy": "Política de limiar",
    "params": "Parâmetros",
    "f1": "F1 (%)",
    "f1-score": "F1 (%)",
    "f2": "F2 (%)",
    "f3": "F3 (%)",
    "recall": "Recall (%)",
    "precision": "Precision (%)",
    "specificity": "Especificidade (%)",
    "accuracy": "Acurácia (%)",
    "balanced_accuracy": "Acurácia balanceada (%)",
    "ap": "AP (score)",
    "average_precision": "AP (score)",
    "roc_auc": "ROC-AUC (score)",
    "log_loss": "Log Loss",
    "brier": "Brier score",
    "mcc": "MCC",
    "kappa": "Kappa",
    "tn": "TN: não falhas liberadas",
    "fp": "FP: falsos alertas",
    "fn": "FN: falhas não detectadas",
    "tp": "TP: falhas detectadas",
    "npv": "Valor preditivo negativo (%)",
    "fpr": "Falsos positivos entre não falhas (%)",
    "fnr": "Falhas não detectadas (%)",
    "alert_rate": "Taxa de alertas (%)",
    "prevalence": "Prevalência de falhas (%)",
    "proportion": "Proporção da classe (%)",
    "positives": "Falhas confirmadas",
    "support": "Observações da classe",
    "fit_n": "Observações de treino",
    "validation_n": "Observações de validação",
    "fit_rows": "Observações de treino",
    "validation_rows": "Observações de validação",
    "fit_positives": "Falhas no treino",
    "validation_positives": "Falhas na validação",
    "pearson": "Pearson com falha",
    "spearman": "Spearman com falha",
    "pearson_fit": "Pearson com falha — treino",
    "spearman_fit": "Spearman com falha — treino",
    "association_strength_fit": "Magnitude da associação — treino",
    "association_method": "Medida de associação",
    "permutation_ap": "Queda de AP por permutação",
    "permutation_ap_drop_validation": "Queda de AP por permutação — validação",
    "permutation_std": "Desvio da permutação (score)",
    "mean_score": "Score médio",
    "observed_rate": "Taxa observada de falhas (%)",
    "tree_importance": "Importância interna da árvore",
    "keep": "Manter?",
    "remove": "Remover?",
    "n_features": "Variáveis utilizadas",
    "missing": "Ausências",
    "cardinality": "Valores distintos",
    "dtype": "Tipo de dado",
    "skew": "Assimetria",
    "iqr_outliers": "Valores fora de 1,5 IQR",
    "mean": "Média",
    "std": "Desvio",
    "rank": "Posição",
    "fit_time": "Tempo de ajuste (s)",
    "score_time": "Tempo de avaliação (s)",
    "mean_fit_time": "Tempo médio de ajuste (s)",
    "std_fit_time": "Desvio do tempo de ajuste (s)",
    "mean_score_time": "Tempo médio de avaliação (s)",
    "std_score_time": "Desvio do tempo de avaliação (s)",
    "Unnamed: 0": "Classe / média",
}
FRACTIONS = {
    "f1",
    "f1-score",
    "f2",
    "f3",
    "recall",
    "precision",
    "specificity",
    "accuracy",
    "balanced_accuracy",
    "npv",
    "fpr",
    "fnr",
    "alert_rate",
    "prevalence",
    "proportion",
    "observed_rate",
    "disagreement",
    "double_fault",
    "both_correct",
    "only_a_wrong",
    "only_b_wrong",
}
PARAMETERS = {
    "n_estimators": "Número de árvores",
    "max_depth": "Profundidade máxima",
    "min_samples_leaf": "Observações mínimas por folha",
    "max_features": "Variáveis por divisão",
    "class_weight": "Peso das classes",
    "learning_rate": "Taxa de aprendizado",
    "min_child_weight": "Peso mínimo de filho",
    "scale_pos_weight": "Peso da classe positiva",
    "subsample": "Fração de linhas por árvore",
    "colsample_bytree": "Fração de variáveis por árvore",
    "reg_alpha": "Regularização L1",
    "reg_lambda": "Regularização L2",
    "n_jobs": "Threads do estimador",
    "objective": "Objetivo",
    "eval_metric": "Métrica interna",
    "tree_method": "Método de construção",
    "criterion": "Critério de divisão",
    "bootstrap": "Amostragem com reposição",
    "min_samples_split": "Observações mínimas para dividir",
}
VALUES = {
    "xgboost": "XGBoost",
    "random_forest": "Random Forest",
    "dummy_prior": "Baseline (proporção)",
    "FIT_RESUBSTITUTION": "Treino (reavaliação no ajuste)",
    "VALIDATION_TRAIN_BLOCK": "Validação (bloco de TRAIN)",
    "FIT_PREFIX": "Treino (prefixo)",
    "CV_SELECTION": "CV de seleção",
    "CV_BASELINE": "CV do baseline",
    "VALIDATION_DIAGNOSTIC_ONLY": "Validação (somente diagnóstico)",
    "abs_spearman": "Módulo de Spearman",
    "cramers_v": "V de Cramér",
    "macro avg": "Média macro",
    "weighted avg": "Média ponderada",
    "accuracy": "Acurácia",
    "0": "Sem falha (0)",
    "1": "Falha (1)",
    "Air temperature [K]": "Temperatura do ar [K]",
    "Process temperature [K]": "Temperatura do processo [K]",
    "Rotational speed [rpm]": "Velocidade de rotação [rpm]",
    "Torque [Nm]": "Torque [Nm]",
    "Tool wear [min]": "Desgaste da ferramenta [min]",
    "Type": "Tipo",
}


def describe_column(column: str) -> tuple[str, bool, bool]:
    """Retorna rótulo, conversão de proporção e indicador de desvio."""
    if column in LABELS:
        return LABELS[column], column in FRACTIONS, False
    if column.startswith("param_"):
        key = column.removeprefix("param_").removeprefix("model__")
        return PARAMETERS.get(key, key), False, False
    base, statistic, context = column, "", ""
    for prefix, label in (("mean_", "média"), ("std_", "desvio"), ("rank_", "posição")):
        if base.startswith(prefix):
            base, statistic = base.removeprefix(prefix), label
            break
    match = re.match(r"split(\d+)_(.*)", base)
    if match:
        context = f"fold {int(match[1]) + 1}"
        base = match[2]
    for prefix, label in (
        ("validation_", "validação"),
        ("fit_", "treino"),
        ("train_", "treino CV"),
        ("test_", "validação CV"),
    ):
        if base.startswith(prefix):
            base = base.removeprefix(prefix)
            context = " — ".join(filter(None, [context, label]))
            break
    deviation = statistic == "desvio" or base.endswith("_std")
    base = base.removesuffix("_std")
    already_percent = base.endswith("_pct")
    base = base.removesuffix("_pct")
    fraction = base in FRACTIONS and statistic != "posição" and not already_percent
    label = LABELS.get(base, base.replace("_", " "))
    if deviation and (fraction or already_percent):
        label = label.replace("(%)", "").strip() + " — desvio (p.p.)"
    elif statistic:
        label += f" — {statistic}"
    elif already_percent and "(%)" not in label:
        label += " (%)"
    if context:
        label += f" — {context}"
    return label, fraction, deviation


def readable_table(frame: pd.DataFrame):
    """Formata percentuais apenas no Styler; suporta colunas MultiIndex da CV."""
    shown = frame.copy(deep=True)
    formats, renamed = {}, []
    for column in frame.columns:
        parts = column if isinstance(column, tuple) else (column,)
        base = str(parts[0])
        label, fraction, deviation = describe_column(base)
        if len(parts) > 1:
            deviation = str(parts[-1]) == "std"
            if fraction:
                label = label.replace("(%)", "").strip()
            stat = (
                ("Desvio (p.p.)" if deviation else "Média (%)")
                if fraction
                else LABELS.get(parts[-1], parts[-1])
            )
            new_column = (label, *parts[1:-1], stat)
        else:
            new_column = label
        renamed.append(new_column)
        numeric = pd.api.types.is_numeric_dtype(frame[column])
        if numeric:
            if fraction:
                suffix = " p.p." if deviation else "%"
                formats[new_column] = lambda v, s=suffix: f"{100 * v:.2f}".replace(".", ",") + s
            elif base.endswith("_pct"):
                formats[new_column] = lambda v: f"{v:.2f}%".replace(".", ",")
            else:

                def number(v):
                    if np.isfinite(v) and float(v).is_integer():
                        return f"{int(v):,}".replace(",", ".")
                    return f"{v:,.4f}".replace(",", "_").replace(".", ",").replace("_", ".")

                formats[new_column] = number
        else:
            shown[column] = shown[column].map(
                lambda v: VALUES.get(v, v) if isinstance(v, str) else v
            )
    if isinstance(frame.columns, pd.MultiIndex):
        shown.columns = pd.MultiIndex.from_tuples(renamed)
    else:
        shown.columns = renamed
    if not isinstance(shown.index, pd.MultiIndex):
        shown.index = shown.index.map(lambda v: VALUES.get(v, v) if isinstance(v, str) else v)
        shown.index.name = LABELS.get(frame.index.name, frame.index.name)
    return shown.style.format(formats, na_rep="Não definido")


def readable_parameters(parameters: dict):
    """Exibe parâmetros com descrição sem mudar os nomes usados pelo estimador."""
    frame = pd.DataFrame(
        [
            {
                "Parâmetro": name,
                "Descrição": PARAMETERS.get(name.removeprefix("model__"), name),
                "Valor": str(value),
            }
            for name, value in parameters.items()
        ]
    )
    return frame.style.hide(axis="index")


def translate_text(text: str) -> str:
    """Traduz textos de figuras sem modificar seus dados."""
    if text in VALUES and text not in ("0", "1"):
        return VALUES[text]
    if text in PARAMETERS:
        return PARAMETERS[text]
    for original in sorted(VALUES, key=len, reverse=True):
        if original not in ("0", "1", "accuracy"):
            text = text.replace(original, VALUES[original])
    text = text.replace("threshold", "limiar").replace("Threshold", "Limiar")
    return re.sub(r"(?<=\d)\.(?=\d)", ",", text)

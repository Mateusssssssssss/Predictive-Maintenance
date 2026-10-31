"""Estatísticas e figuras a partir de artefatos; não treina nem lê TEST."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FixedFormatter, FixedLocator, PercentFormatter
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

from predictive_maintenance.presentation import translate_text

COLORS = {"f1": "#3977a8", "recall": "#d96540", "precision": "#4c986c"}


class ModelReport:
    def __init__(self, root: Path, run: Path):
        self.root, self.run = root, run
        self.manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
        if self.manifest.get("test_evaluated") is not False:
            raise ValueError("Este relatório aceita somente experimentos sem avaliação de TEST.")
        self.name = self.manifest["model"]
        self.tables = root / "reports/metrics" / self.manifest["run_id"]
        self.figures = root / "reports/figures" / self.manifest["run_id"]
        self.figures.mkdir(parents=True, exist_ok=True)
        sns.set_theme(style="whitegrid")
        plt.rcParams.update(
            {"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False}
        )

    def table(self, name: str) -> pd.DataFrame:
        return pd.read_csv(self.tables / f"{name}.csv")

    def save(self, fig, name: str) -> None:
        for ax in fig.axes:
            ax.set_title(translate_text(ax.get_title()))
            ax.set_xlabel(translate_text(ax.get_xlabel()))
            ax.set_ylabel(translate_text(ax.get_ylabel()))
            for label in [*ax.texts, *ax.get_xticklabels(), *ax.get_yticklabels()]:
                label.set_text(translate_text(label.get_text()))
            # Rótulos categóricos precisam ser fixados no eixo para persistir no render.
            for axis in (ax.xaxis, ax.yaxis):
                if isinstance(axis.get_major_locator(), FixedLocator):
                    axis.set_major_formatter(
                        FixedFormatter(
                            [translate_text(value.get_text()) for value in axis.get_ticklabels()]
                        )
                    )
            if ax.get_legend():
                for label in ax.get_legend().get_texts():
                    label.set_text(translate_text(label.get_text()))
        fig.tight_layout()
        fig.savefig(self.figures / f"{name}.png", dpi=150, bbox_inches="tight")
        plt.show()
        plt.close(fig)

    def summary(self) -> pd.DataFrame:
        folds = self.table("fold_metrics")
        columns = [f"validation_{m}" for m in ["ap", "f1", "recall", "precision", "roc_auc"]]
        table = folds.groupby("model")[columns].agg(["mean", "std"])
        table.to_csv(self.tables / "cv_summary.csv")
        return table

    def cv_figures(self) -> None:
        folds = self.table("fold_metrics")
        fig, axes = plt.subplots(1, 2, figsize=(13, 4))
        for name, group in folds.groupby("model", sort=False):
            axes[0].plot(group["fold"], group["validation_ap"], marker="o", label=name)
        selected = folds[folds["model"].eq(self.name)]
        axes[1].plot(selected["fold"], selected["fit_ap"], marker="o", label="Ajuste do fold")
        axes[1].plot(
            selected["fold"], selected["validation_ap"], marker="o", label="Validação do fold"
        )
        for ax in axes:
            ax.set(
                xlabel="Fold ordenado", ylabel="AP [0, 1]", ylim=(0, 1.05), xticks=selected["fold"]
            )
            ax.legend()
        axes[0].set_title("CV de seleção: candidato e baseline")
        axes[1].set_title("Diferença entre ajuste e validação — CV")
        self.save(fig, "cv_ap_and_overfitting")

        grid = self.table("grid_results").sort_values("rank_test_ap").head(10).iloc[::-1]
        fig, ax = plt.subplots(figsize=(9, 5))
        bars = ax.barh(
            [f"Candidato {i}" for i in grid.index],
            grid["mean_test_ap"],
            xerr=grid["std_test_ap"],
            color="#3977a8",
            capsize=3,
        )
        ax.bar_label(bars, fmt="%.3f", padding=4)
        ax.set(
            xlabel="AP média ± desvio entre folds [seleção]",
            title="Top 10 da grade — CV de seleção",
            xlim=(0, 1.1),
        )
        self.save(fig, "grid_top_candidates")

        parameters = list(self.manifest["training_config"][self.name]["grid"])
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        for parameter, ax in zip(parameters, axes.flat):
            profile = self.table("grid_results")
            profile[parameter] = profile[f"param_{parameter}"].astype("string").fillna("None")
            grouped = profile.groupby(parameter, dropna=False)["mean_test_ap"].max()
            ax.bar(grouped.index.astype(str), grouped, color="#3977a8")
            ax.set(
                title=parameter.removeprefix("model__"), ylabel="Melhor média CV AP", ylim=(0, 1)
            )
        for ax in list(axes.flat)[len(parameters) :]:
            ax.set_visible(False)
        self.save(fig, "hyperparameter_profiles")

    def comparison(self) -> None:
        metrics = self.table("model_metrics")
        values = metrics.loc[metrics["origin"].eq("VALIDATION_TRAIN_BLOCK")].sort_values(
            "ap", ascending=False
        )
        fig, ax = plt.subplots(figsize=(9, 5))
        x = np.arange(len(values))
        for offset, metric in enumerate(COLORS):
            heights = values[metric].fillna(0)
            bars = ax.bar(
                x + (offset - 1) * 0.23, heights, 0.23, label=metric.title(), color=COLORS[metric]
            )
            ax.bar_label(
                bars,
                labels=[
                    "Não definido" if pd.isna(v) else f"{100 * v:.2f}%" for v in values[metric]
                ],
                padding=3,
            )
        ax.set(
            xticks=x,
            xticklabels=values["model"],
            ylim=(0, 1.15),
            ylabel="Métricas (%)",
            title=f"Baseline × {self.name} — VALIDAÇÃO / limiar {self.manifest['threshold']}",
        )
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
        ax.legend()
        self.save(fig, "comparison_f1_recall_precision")

    def confusion(self) -> None:
        p = self.table("validation_predictions")
        matrix = confusion_matrix(p.y_true, p.prediction, labels=[0, 1])
        normalized = np.divide(
            matrix,
            matrix.sum(axis=1, keepdims=True),
            out=np.zeros_like(matrix, dtype=float),
            where=matrix.sum(axis=1, keepdims=True) != 0,
        )
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, values, fmt, title in zip(
            axes, [matrix, normalized], ["d", ".1%"], ["Contagens", "Por classe real"]
        ):
            sns.heatmap(
                values,
                annot=True,
                fmt=fmt,
                cmap="Blues",
                cbar=False,
                ax=ax,
                xticklabels=["Sem falha", "Falha"],
                yticklabels=["Sem falha", "Falha"],
            )
            ax.set(xlabel="Previsto", ylabel="Real", title=f"{title} — VALIDAÇÃO")
        self.save(fig, "confusion_matrices")

    def probability_figures(self) -> None:
        p = self.table("validation_predictions")
        metrics = self.table("model_metrics")
        row = metrics.loc[
            metrics.model.eq(self.name) & metrics.origin.eq("VALIDATION_TRAIN_BLOCK")
        ].iloc[0]
        precision, recall, _ = precision_recall_curve(p.y_true, p.score)
        fpr, tpr, _ = roc_curve(p.y_true, p.score)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].plot(recall, precision, label=f"AP={row.ap:.3f}")
        axes[0].axhline(
            p.y_true.mean(),
            linestyle="--",
            color="gray",
            label=f"Prevalência={p.y_true.mean():.2%}",
        )
        axes[0].scatter(
            row.recall, row.precision, c="#d96540", label=f"threshold={row.threshold}", zorder=5
        )
        axes[0].set(xlabel="Recall (%)", ylabel="Precision (%)", title="PR — VALIDAÇÃO")
        axes[1].plot(fpr, tpr, label=f"ROC-AUC={row.roc_auc:.3f}")
        axes[1].plot([0, 1], [0, 1], "--", color="gray")
        axes[1].set(
            xlabel="Falsos positivos entre não falhas (%)",
            ylabel="Recall (%)",
            title="ROC — VALIDAÇÃO",
        )
        for ax in axes:
            ax.xaxis.set_major_formatter(PercentFormatter(xmax=1))
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
            ax.set(xlim=(0, 1), ylim=(0, 1.05))
            ax.legend()
        self.save(fig, "pr_and_roc")

        observed, predicted = calibration_curve(p.y_true, p.score, n_bins=10, strategy="quantile")
        pd.DataFrame({"mean_score": predicted, "observed_rate": observed}).to_csv(
            self.tables / "calibration.csv", index=False
        )
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].plot(predicted, observed, marker="o", label=f"Brier={row.brier:.4f}")
        axes[0].plot([0, 1], [0, 1], "--", color="gray")
        axes[0].set(
            xlabel="Score médio",
            ylabel="Fração observada de falhas",
            title="Confiabilidade — VALIDAÇÃO",
            xlim=(0, 1),
            ylim=(0, 1),
        )
        axes[0].legend()
        axes[0].yaxis.set_major_formatter(PercentFormatter(xmax=1))
        for label, color in [(0, "#3977a8"), (1, "#d96540")]:
            axes[1].hist(
                p.loc[p.y_true.eq(label), "score"],
                bins=np.linspace(0, 1, 26),
                alpha=0.5,
                color=color,
                label=f"Real={label}",
            )
        axes[1].axvline(self.manifest["threshold"], linestyle="--", color="black")
        axes[1].set(
            xlabel="Score positivo", ylabel="Contagem", title="Distribuição dos scores — VALIDAÇÃO"
        )
        axes[1].legend()
        self.save(fig, "calibration_and_scores")

    def threshold_figure(self) -> None:
        table = self.table("threshold_diagnostics")
        fig, ax = plt.subplots(figsize=(14, 5))
        x = np.arange(len(table))
        for offset, metric in enumerate(COLORS):
            bars = ax.bar(
                x + (offset - 1) * 0.25,
                table[metric].fillna(0),
                0.25,
                color=COLORS[metric],
                label=metric.title(),
            )
            ax.bar_label(
                bars,
                labels=["ND" if pd.isna(v) else f"{100 * v:.1f}%" for v in table[metric]],
                fontsize=7,
                rotation=90,
                padding=2,
            )
        reference = int(np.argmin(abs(table.threshold - self.manifest["threshold"])))
        ax.axvspan(
            reference - 0.45, reference + 0.45, color="gold", alpha=0.15, label="Referência fixa"
        )
        ax.set(
            xticks=x,
            xticklabels=[f"{v:.2f}" for v in table.threshold],
            ylim=(0, 1.25),
            xlabel="Threshold (score ≥ limiar)",
            ylabel="Métricas (%)",
            title="Sensibilidade na VALIDAÇÃO — diagnóstico, sem seleção de limiar",
        )
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
        ax.legend(ncol=4)
        self.save(fig, "threshold_sensitivity")

    def feature_figure(self) -> None:
        table = self.table("feature_association_importance").sort_values(
            "permutation_ap_drop_validation"
        )
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
        axes[0].barh(table.feature, table.association_strength_fit, color="#3977a8")
        axes[0].set(
            xlabel="|Spearman| numéricas / Cramér V para Type",
            title="Associação no FIT",
            xlim=(0, 1),
        )
        axes[1].barh(
            table.feature,
            table.permutation_ap_drop_validation,
            xerr=table.permutation_std,
            capsize=3,
            color="#d96540",
        )
        axes[1].axvline(0, color="gray")
        axes[1].set(xlabel="Queda de AP ± desvio das permutações", title="Importância na VALIDAÇÃO")
        self.save(fig, "association_vs_permutation_importance")

    def learning_figure(self) -> None:
        table = self.table("learning_diagnostic")
        fig, ax = plt.subplots(figsize=(9, 4))
        for origin, group in table.groupby("origin"):
            ax.plot(group.fit_n, group.ap, marker="o", label=origin)
        ax.set(
            xlabel="Observações no prefixo de ajuste",
            ylabel="AP [0, 1]",
            ylim=(0, 1.05),
            title="Diagnóstico com parâmetros congelados — VALIDAÇÃO fixa",
        )
        ax.legend()
        self.save(fig, "learning_diagnostic")

    def error_figure(self) -> None:
        p = self.table("validation_predictions")
        p["error"] = np.select(
            [
                p.y_true.eq(1) & p.prediction.eq(0),
                p.y_true.eq(0) & p.prediction.eq(1),
            ],
            ["FN", "FP"],
            default="Correto",
        )
        counts = (
            p.groupby(["Type", "error"])
            .size()
            .unstack(fill_value=0)
            .reindex(columns=["Correto", "FP", "FN"], fill_value=0)
        )
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        counts.plot.bar(stacked=True, ax=axes[0], color=["#9bc0d9", "#d9aa40", "#d96540"])
        axes[0].set(title="Erros por tipo — VALIDAÇÃO", ylabel="Observações", xlabel="Type")
        axes[1].scatter(
            p.UDI, p.score, c=p.y_true.map({0: "#3977a8", 1: "#d96540"}), s=10, alpha=0.6
        )
        axes[1].axhline(self.manifest["threshold"], linestyle="--", color="black")
        axes[1].set(
            title="Score por ordem — cor = alvo real",
            xlabel="UDI (não tempo)",
            ylabel="Score positivo",
            ylim=(0, 1),
        )
        self.save(fig, "errors_by_type_and_order")

"""Figuras de Optuna, ablação e pós-processamento; sem ajuste de modelos."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

from predictive_maintenance.reporting import COLORS, ModelReport


class AdvancedReport(ModelReport):
    def study_figures(self):
        trials = self.table("optuna_trials")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(trials.number + 1, trials.value, marker="o", alpha=0.6, label="Tentativa")
        ax.plot(trials.number + 1, trials.value.cummax(), linewidth=2, label="Melhor acumulado")
        ax.set(
            xlabel="Tentativa Optuna",
            ylabel=f"{self.selection_metric.upper()} média de CV [0, 1]",
            title="Evolução da busca — somente FIT_SEARCH",
            ylim=(0, 1.05),
        )
        ax.legend()
        self.save(fig, "optuna_history")
        folds = self.table("fold_metrics")
        fig, ax = plt.subplots(figsize=(9, 4))
        for label, group in folds.groupby("model"):
            ax.plot(
                group.fold, group[f"validation_{self.selection_metric}"], marker="o", label=label
            )
        ax.set(
            xlabel="Fold ordenado",
            ylabel=self.selection_metric.upper(),
            ylim=(0, 1.05),
            xticks=sorted(folds.fold.unique()),
            title="Melhor tentativa × baseline — CV de seleção",
        )
        ax.legend()
        self.save(fig, "advanced_cv")

    def ablation_figure(self):
        table = self.table("feature_ablation")
        fig, ax = plt.subplots(figsize=(9, 4))
        for label, group in table.groupby("version"):
            ax.plot(group.fold, group[self.selection_metric], marker="o", label=label)
        ax.set(
            xlabel="Fold ordenado",
            ylabel=self.selection_metric.upper(),
            ylim=(0, 1.05),
            xticks=sorted(table.fold.unique()),
            title="Ablação de variáveis — mesmos parâmetros / somente CV",
        )
        ax.legend()
        self.save(fig, "feature_ablation")

    def stage_figures(self):
        table = self.table("stage_metrics")
        x = np.arange(len(table))
        fig, ax = plt.subplots(figsize=(13, 5))
        for offset, metric in enumerate(COLORS):
            bars = ax.bar(
                x + (offset - 1) * 0.24,
                table[metric].fillna(0),
                0.24,
                color=COLORS[metric],
                label=metric.title(),
            )
            ax.bar_label(
                bars, labels=["ND" if np.isnan(v) else f"{v:.1%}" for v in table[metric]], padding=3
            )
        ax.set(
            xticks=x,
            xticklabels=[
                "Optuna + derivadas / 0,5",
                "Calibrado / 0,5",
                "Calibrado / limiar escolhido",
            ],
            ylim=(0, 1.15),
            ylabel="Métricas (%)",
            title="Etapas congeladas antes de avaliar a VALIDAÇÃO",
        )
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
        ax.legend()
        self.save(fig, "postprocessing_stages")
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for column, ax in zip(["brier", "log_loss"], axes):
            bars = ax.bar(x, table[column], color="#3977a8")
            ax.bar_label(bars, fmt="%.4f", padding=3)
            ax.set(
                xticks=x,
                xticklabels=["Base", "Calibrado", "Calibrado + limiar"],
                ylabel=column,
                title=f"{column} — validação (menor é melhor)",
            )
            ax.margins(y=0.2)
        self.save(fig, "probability_losses_by_stage")

    def threshold_selection_figure(self):
        table = self.table("threshold_selection")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(table.thresholds, table.scores, label="F1 no bloco THRESHOLD")
        ax.axvline(
            self.manifest["threshold"],
            color="#d96540",
            linestyle="--",
            label=f"Limiar escolhido = {self.manifest['threshold']:.4f}",
        )
        ax.set(
            xlabel="Limiar",
            ylabel="F1 (%)",
            ylim=(0, 1.05),
            title="Escolha do limiar — somente bloco THRESHOLD",
        )
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
        ax.legend()
        self.save(fig, "threshold_selection")

    def previous_figure(self):
        table = self.table("previous_comparison")
        x = np.arange(len(table))
        fig, ax = plt.subplots(figsize=(11, 5))
        for offset, metric in enumerate(COLORS):
            bars = ax.bar(
                x + (offset - 1) * 0.24,
                table[metric].fillna(0),
                0.24,
                color=COLORS[metric],
                label=metric.title(),
            )
            ax.bar_label(
                bars, labels=["ND" if np.isnan(v) else f"{v:.2%}" for v in table[metric]], padding=3
            )
        ax.set(
            xticks=x,
            xticklabels=table.version,
            ylabel="Métricas (%)",
            ylim=(0, 1.15),
            title=f"{self.name} — antes/depois / mesma validação / receitas e limiares distintos",
        )
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
        ax.legend()
        self.save(fig, "previous_comparison")

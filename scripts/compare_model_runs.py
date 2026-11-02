"""Compara scores de validação persistidos, sem acessar TEST ou treinar modelos."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import PercentFormatter

from predictive_maintenance.evaluation import binary_metrics
from predictive_maintenance.presentation import translate_text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["all", "xgboost", "random_forest"], default="all")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    combined = []
    names = ["xgboost", "random_forest"] if args.model == "all" else [args.model]
    for name in names:
        paths = sorted((root / "reports/experiments").glob(f"train_{name}_*/manifest.json"))
        if len(paths) < 2:
            raise ValueError("Execute uma nova receita antes de comparar.")
        old, new = [json.loads(p.read_text(encoding="utf-8")) for p in paths[-2:]]
        for field in ["dataset_sha256", "split_sha256", "cv_sha256"]:
            if old[field] != new[field]:
                raise ValueError(f"Comparação incompatível: {field}")
        rows = []
        for version, manifest in [("Anterior", old), ("Atual", new)]:
            predictions = pd.read_csv(
                root / "reports/metrics" / manifest["run_id"] / "validation_predictions.csv"
            )
            rows.append(
                {
                    "model": name,
                    "version": version,
                    "run_id": manifest["run_id"],
                    "origin": "VALIDATION_TRAIN_BLOCK",
                    **binary_metrics(predictions.y_true, predictions.score, manifest["threshold"]),
                }
            )
        result = pd.DataFrame(rows)
        tables = root / "reports/metrics" / new["run_id"]
        result.to_csv(tables / "previous_comparison.csv", index=False)
        changes = {
            m: rows[1][m] - rows[0][m]
            for m in ["auprc", "ap", "f1", "recall", "precision", "fp", "fn"]
        }
        (root / "reports/experiments" / new["run_id"] / "comparison.json").write_text(
            json.dumps(
                {
                    "previous": old["run_id"],
                    "current": new["run_id"],
                    "delta": changes,
                    "notes": "Repeated development validation; not independent confirmation. Recipes can differ in sampling, weights and hyperparameter space; changes are not isolated causal effects. Selection occurs only within CV.",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        fig, ax = plt.subplots(figsize=(10, 5))
        for offset, metric in enumerate(["f1", "recall", "precision"]):
            bars = ax.bar(
                result.index + (offset - 1) * 0.25, result[metric], 0.25, label=metric.title()
            )
            ax.bar_label(
                bars, labels=[f"{v:.2%}".replace(".", ",") for v in result[metric]], padding=3
            )
        ax.set(
            xticks=result.index,
            xticklabels=result.version,
            ylim=(0, 1.15),
            ylabel="Métricas (%)",
            title=f"{translate_text(name)} — antes e depois / mesma validação / limiar 0,5",
        )
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
        ax.legend()
        fig.tight_layout()
        figures = root / "reports/figures" / new["run_id"]
        figures.mkdir(exist_ok=True, parents=True)
        fig.savefig(figures / "previous_comparison.png", dpi=150)
        plt.close(fig)
        combined.extend(rows)
    directory = root / "reports/comparisons"
    directory.mkdir(exist_ok=True)
    destination = directory / f"{new['run_id']}.csv"
    pd.DataFrame(combined).to_csv(destination, index=False)
    print(
        pd.DataFrame(combined)[
            ["model", "version", "auprc", "ap", "f1", "recall", "precision", "fp", "fn"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()

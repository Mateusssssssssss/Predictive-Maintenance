"""Avaliação final dos artefatos congelados; nenhuma chamada a fit ou seleção."""

import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

from predictive_maintenance.data import development_split, load_config, load_data, sha256
from predictive_maintenance.evaluation import binary_metrics, segment_metrics


def bootstrap_intervals(y, scores, threshold, repeats=500, seed=42):
    """Percentis pareados por registro; hipótese IID, sem garantia temporal/grupos."""
    rng = np.random.default_rng(seed)
    keys = ["ap", "auprc", "roc_auc", "f1", "recall", "precision", "brier"]
    samples = []
    for _ in range(repeats):
        indices = rng.integers(0, len(y), len(y))
        metrics = binary_metrics(np.asarray(y)[indices], np.asarray(scores)[indices], threshold)
        samples.append([metrics[k] for k in keys])
    values = np.asarray(samples)
    return pd.DataFrame(
        {
            "metric": keys,
            "lower_95": np.nanpercentile(values, 2.5, axis=0),
            "upper_95": np.nanpercentile(values, 97.5, axis=0),
            "valid_repeats": np.isfinite(values).sum(axis=0),
        }
    )


def evaluate_test(root: Path) -> Path:
    """Usa somente runs fixadas; reexecuções leem o snapshot já calculado."""
    specification = root / "configs/test_evaluation.json"
    settings = json.loads(specification.read_text(encoding="utf-8"))
    config = load_config(root)
    records = {}
    for name, run_id in settings["runs"].items():
        path = root / "reports/experiments" / run_id
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        if manifest["model"] != name or manifest["test_evaluated"] is not False:
            raise ValueError("Run incompatível com avaliação final isolada.")
        if manifest["dataset_sha256"] != sha256(root / config["source"]):
            raise ValueError("Dataset difere do utilizado no treinamento.")
        if manifest["model_sha256"] != sha256(root / manifest["model_path"]):
            raise ValueError("Artefato do modelo foi alterado.")
        records[name] = manifest
    output = root / "reports/test" / ("test_" + sha256(specification)[:12])
    output.mkdir(parents=True, exist_ok=True)
    provenance = {
        "settings": settings,
        "config_sha256": sha256(specification),
        "dataset_sha256": sha256(root / config["source"]),
        "models": {
            name: {key: m[key] for key in ["run_id", "model_sha256", "threshold"]}
            for name, m in records.items()
        },
    }
    existing = output / "manifest.json"
    if existing.exists():
        saved = json.loads(existing.read_text(encoding="utf-8"))
        if any(saved[key] != value for key, value in provenance.items()):
            raise ValueError("Snapshot de TEST incompatível; não sobrescrever resultados.")
        for relative, digest in saved["artifact_sha256"].items():
            if sha256(output / relative) != digest:
                raise ValueError("Artefato TEST foi alterado.")
        return output
    development, test = development_split(load_data(root, config), config)
    features = config["numeric_features"] + config["categorical_features"]
    y = test[config["target"]].to_numpy()
    predictions, metrics, intervals, classes, segments, comparison = [], [], [], [], [], []
    for name, m in records.items():
        membership = pd.read_csv(
            root / "reports/experiments" / m["run_id"] / "split_membership.csv"
        )
        reserved = membership.loc[membership.partition.eq("TEST_RESERVED"), "UDI"]
        np.testing.assert_array_equal(reserved, test.UDI)
        model = joblib.load(root / m["model_path"])
        scores = model.predict_proba(test[features])[:, 1]
        threshold = m["threshold"]
        decisions = (scores >= threshold).astype(int)
        np.testing.assert_array_equal(decisions, model.predict(test[features]))
        frame = test[["UDI", "Type"]].copy()
        frame["model"], frame["y_true"] = name, y
        frame["score"], frame["prediction"] = scores, decisions
        predictions.append(frame)
        result = {"model": name, "partition": "TEST", **binary_metrics(y, scores, threshold)}
        metrics.append(result)
        comparison.append(result)
        old = pd.read_csv(root / "reports/metrics" / m["run_id"] / "validation_predictions.csv")
        comparison.append(
            {
                "model": name,
                "partition": "VALIDATION",
                **binary_metrics(old.y_true, old.score, threshold),
            }
        )
        intervals.append(
            bootstrap_intervals(
                y, scores, threshold, settings["bootstrap_repeats"], settings["seed"]
            ).assign(model=name)
        )
        report = classification_report(
            y,
            decisions,
            labels=[0, 1],
            target_names=["Sem falha", "Falha"],
            output_dict=True,
            zero_division=0,
        )
        classes.append(pd.DataFrame(report).T.rename_axis("class").reset_index().assign(model=name))
        segments.append(segment_metrics(frame, "Type", threshold).assign(model=name))
    # Baseline prior usa apenas prevalência do bloco FIT_SEARCH de cada receita.
    for name, m in records.items():
        prior = development.iloc[: m["fit_rows"]][config["target"]].mean()
        metrics.append(
            {
                "model": f"baseline_prior_{name}",
                "partition": "TEST",
                **binary_metrics(y, np.full(len(y), prior), 0.5),
            }
        )
    for filename, frame in {
        "predictions": pd.concat(predictions, ignore_index=True),
        "metrics": pd.DataFrame(metrics),
        "validation_test": pd.DataFrame(comparison),
        "bootstrap_intervals": pd.concat(intervals, ignore_index=True),
        "classification_report": pd.concat(classes, ignore_index=True),
        "segments_type": pd.concat(segments, ignore_index=True),
        "membership": test[["UDI"]].assign(partition="TEST"),
    }.items():
        frame.to_csv(output / f"{filename}.csv", index=False)
    provenance.update(
        {
            "evaluated_at_utc": datetime.now(UTC).isoformat(),
            "test_rows": len(test),
            "positives": int(y.sum()),
            "test_evaluated": True,
            "refit": False,
            "threshold_tuning_on_test": False,
            "code_sha256": sha256(Path(__file__)),
            "artifact_sha256": {p.name: sha256(p) for p in sorted(output.glob("*.csv"))},
            "limitations": [
                "UDI não comprova tempo",
                "Bootstrap IID por registro; dependência pode invalidar intervalos",
                "TEST revelado: novos ajustes exigem outra confirmação independente",
            ],
        }
    )
    existing.write_text(json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8")
    return output


def test_figures(output: Path):
    predictions = pd.read_csv(output / "predictions.csv")
    metrics = pd.read_csv(output / "metrics.csv").set_index("model")
    folder = output / "figures"
    folder.mkdir(exist_ok=True)

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(folder / f"{name}.png", dpi=150, bbox_inches="tight")
        plt.show()
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, (name, rows) in zip(axes, predictions.groupby("model")):
        matrix = confusion_matrix(rows.y_true, rows.prediction, labels=[0, 1])
        ax.imshow(matrix, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(
                    j,
                    i,
                    str(matrix[i, j]),
                    ha="center",
                    va="center",
                    color="white" if matrix[i, j] > matrix.max() / 2 else "black",
                )
        ax.set(
            xticks=[0, 1],
            yticks=[0, 1],
            xticklabels=["Sem falha", "Falha"],
            yticklabels=["Sem falha", "Falha"],
            xlabel="Previsto",
            ylabel="Real",
            title=f"{name} · limiar {metrics.loc[name, 'threshold']:.4f}",
        )
    save(fig, "confusion_matrices")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for name, rows in predictions.groupby("model"):
        p, r, _ = precision_recall_curve(rows.y_true, rows.score)
        fpr, tpr, _ = roc_curve(rows.y_true, rows.score)
        observed, predicted = calibration_curve(
            rows.y_true, rows.score, n_bins=8, strategy="quantile"
        )
        axes[0].plot(r, p, label=f"{name} · AP {metrics.loc[name, 'ap']:.3f}")
        axes[1].plot(fpr, tpr, label=f"{name} · AUC {metrics.loc[name, 'roc_auc']:.3f}")
        axes[2].plot(predicted, observed, marker="o", label=name)
    axes[0].axhline(predictions.y_true.mean(), ls="--", color="gray", label="Prevalência TEST")
    axes[1].plot([0, 1], [0, 1], "--", color="gray")
    axes[2].plot([0, 1], [0, 1], "--", color="gray")
    for ax, title, xlabel, ylabel in zip(
        axes,
        ["Precisão × revocação", "ROC", "Calibração (descritiva)"],
        ["Revocação", "Taxa de falsos positivos", "Probabilidade média prevista"],
        ["Precisão", "Taxa de verdadeiros positivos", "Frequência de falha"],
    ):
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel, xlim=(0, 1), ylim=(0, 1.03))
        ax.legend(fontsize=8)
    save(fig, "pr_roc_calibration")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, (name, rows) in zip(axes, predictions.groupby("model")):
        for label, subset in rows.groupby("y_true"):
            ax.hist(
                subset.score,
                bins=np.linspace(0, 1, 21),
                alpha=0.6,
                label="Falha" if label else "Sem falha",
            )
        ax.axvline(metrics.loc[name, "threshold"], color="black", ls="--", label="Limiar congelado")
        ax.set(title=name, xlabel="Probabilidade prevista", ylabel="Registros", yscale="symlog")
        ax.legend()
    save(fig, "score_distributions")

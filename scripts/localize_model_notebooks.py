"""Aplica apresentação portuguesa aos notebooks existentes, preservando suas células."""

import re
from pathlib import Path

import nbformat


def localize(notebook):
    for cell in notebook.cells:
        if cell.cell_type == "code":
            source = cell.source
            anchor = "from predictive_maintenance.reporting import ModelReport"
            if anchor in source and "import readable_table" not in source:
                source = source.replace(
                    anchor,
                    anchor
                    + "\nfrom predictive_maintenance.presentation import readable_table, readable_parameters",
                )
            source = re.sub(
                r"display\((report\.table\('[^']+'\))\)", r"display(readable_table(\1))", source
            )
            for expression in (
                "folds",
                "metrics",
                "report.summary()",
                "grid.sort_values('rank_test_ap')[columns].head(10)",
            ):
                source = source.replace(
                    f"display({expression})", f"display(readable_table({expression}))"
                )
            for expression in (
                "manifest['training_config'][MODEL_NAME]['grid']",
                "manifest['fixed_params']",
                "manifest['best_params']",
            ):
                source = source.replace(
                    f"display({expression})", f"display(readable_parameters({expression}))"
                )
            source = source.replace("F1={validation.f1:.3f}", "F1={validation.f1:.2%}")
            source = source.replace(
                "Recall={validation.recall:.3f}", "Recall={validation.recall:.2%}"
            )
            source = source.replace(
                "Precision={validation.precision:.3f}", "Precision={validation.precision:.2%}"
            )
            if (
                "from predictive_maintenance.presentation import" in source
                and "def translated_markdown" not in source
            ):
                source = source.replace(
                    "from predictive_maintenance.presentation import readable_table, readable_parameters",
                    "from predictive_maintenance.presentation import readable_table, readable_parameters, translate_text"
                    "\n\ndef translated_markdown(text):\n    return Markdown(translate_text(text))\n",
                )
            source = source.replace("display(Markdown(", "display(translated_markdown(")
            cell.source = source
        else:
            cell.source = cell.source.replace(
                "Scores apresentados em 0–1, contagens em inteiros.",
                "Nas tabelas e barras, F1/Recall/Precision e taxas aparecem em %, com vírgula decimal; "
                "AP e ROC-AUC permanecem scores em 0–1. Desvios de percentuais são pontos percentuais (p.p.). "
                "Os artefatos numéricos não são alterados.",
            )
            cell.source = cell.source.replace("`indef.` nas barras", "`Não definido` nas barras")
    return notebook


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    for path in sorted((root / "notebooks").glob("metrics_*.ipynb")):
        notebook = localize(nbformat.read(path, as_version=4))
        nbformat.validate(notebook)
        nbformat.write(notebook, path)
        print(f"Apresentação atualizada: {path.name}")

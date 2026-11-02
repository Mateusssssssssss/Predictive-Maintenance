"""Atualiza relatórios existentes para AUPRC/SMOTENC, mantendo os notebooks anteriores."""

from pathlib import Path
from shutil import copy2

import nbformat as nbf


def update(notebook, name):
    xgb = name == "xgboost"
    for cell in notebook.cells:
        s = cell.source
        if cell.cell_type == "code":
            s = s.replace(
                "display(readable_table(comparison))",
                "display(readable_table(comparison[['version', 'threshold', 'auprc', 'ap', 'f1', 'recall', 'precision', 'fp', 'fn']]))",
            )
            s = s.replace(
                "columns = ['rank_test_ap', 'mean_test_ap', 'std_test_ap', 'mean_train_ap',",
                "selection_metric = manifest.get('selection_metric', 'ap')\ncolumns = [f'rank_test_{selection_metric}', f'mean_test_{selection_metric}', f'std_test_{selection_metric}', f'mean_train_{selection_metric}', 'mean_test_ap', 'mean_test_auprc',",
            )
            s = s.replace(
                "grid.sort_values('rank_test_ap')[columns]",
                "grid.sort_values(f'rank_test_{selection_metric}')[list(dict.fromkeys(columns))]",
            )
            s = s.replace(
                "AP={validation.ap:.3f};", "AUPRC={validation.auprc:.3f}; AP={validation.ap:.3f};"
            )
            if xgb:
                s = s.replace("selected_folds.validation_ap", "selected_folds.validation_auprc")
                s = s.replace("**AP de seleção:**", "**AUPRC de seleção:**")
                s = s.replace(
                    "print(f\"AP média de seleção: {manifest['best_cv_ap']:.4f}\")",
                    "print(f\"AUPRC média de seleção: {manifest['best_cv_auprc']:.4f}\")",
                )
        elif xgb:
            s = s.replace(
                "**Critério:** maior média de AP", "**Critério:** maior média de AUPRC trapezoidal"
            )
            s = s.replace(
                "Sem SMOTE,\nundersampling,",
                "SMOTENC somente no treino de cada fold; sem undersampling,",
            )
            s = s.replace("Sem SMOTE,", "SMOTENC somente no treino do fold; sem")
            s = s.replace(
                "Árvores sem scaling.", "Escala padronizada antes das distâncias do SMOTENC."
            )
            s = s.replace(
                "árvores sem scaling.", "sensores padronizados para distâncias do SMOTENC."
            )
            s = s.replace("AP médio", "AUPRC trapezoidal médio")
            s = s.replace("32 candidatos", "32 candidatos")
            if "scale_pos_weight=10" in s:
                s = (
                    "A grade investiga árvores, learning rate, profundidade, min_child_weight e "
                    "razão de síntese SMOTENC (0,25 ou 0,5 falhas/não falhas após a operação). "
                    "k_neighbors=3 fica fixo; o menor treino da CV tem 45 falhas. "
                    "scale_pos_weight=1 evita somar reponderação à reamostragem nesta receita. "
                    "Sensores são imputados e padronizados antes de SMOTENC; Type é imputado "
                    "e preservado como categoria, recebendo one-hot somente após a síntese. "
                    "Tudo ocorre dentro do treino de cada fold. Validação mantém os registros originais. "
                    "Subsample, colsample_bytree, L1 e L2 permanecem fixos. Sem early stopping."
                )
            s = s.replace("A unidade é queda de AP;", "A unidade é queda de AUPRC trapezoidal;")
        cell.source = s
        if cell.cell_type == "markdown" and "## 6. PR, ROC" in s and "pontuação constante" not in s:
            cell.source += (
                "\n\n**Nota sobre AUPRC:** uma pontuação constante pode gerar poucos pontos na curva PR; "
                "a interpolação trapezoidal pode resultar em área alta mesmo com AP igual à prevalência. "
                "Por isso AP e AUPRC são mostradas separadamente, inclusive no baseline."
            )
    if not any("## 12. Comparação com a execução anterior" in c.source for c in notebook.cells):
        notebook.cells.extend(
            [
                nbf.v4.new_markdown_cell(
                    "## 12. Comparação com a execução anterior\n\nMesmos registros e folds. "
                    "AUPRC trapezoidal e AP são calculadas com predict_proba[:, 1]; a execução anterior "
                    "já usava predict_proba. Comparação na validação de desenvolvimento reutilizada; "
                    "não demonstra generalização independente. O efeito conjunto de sampler, pesos e "
                    "critério de busca não pode ser atribuído somente ao SMOTENC. TEST permanece reservado."
                ),
                nbf.v4.new_code_cell(
                    "comparison = report.table('previous_comparison')\ndisplay(readable_table(comparison))\n"
                    "from IPython.display import Image\ndisplay(Image(filename=str(report.figures / 'previous_comparison.png')))\n"
                    "before, after = comparison.iloc[0], comparison.iloc[1]\n"
                    "display(translated_markdown(f'AUPRC: {before.auprc:.4f} → {after.auprc:.4f}; ' "
                    "f'AP: {before.ap:.4f} → {after.ap:.4f}; ' "
                    "f'F1: {before.f1:.2%} → {after.f1:.2%}. ' "
                    "'Não houve ajuste de limiar nesta comparação.'))"
                ),
            ]
        )
        if xgb:
            notebook.cells.extend(
                [
                    nbf.v4.new_markdown_cell(
                        "## 13. Auditoria do SMOTENC\n\nContagens reais do sampler escolhido "
                        "em cada treino e no refit. As contagens de CV são uma reprodução apenas do sampler "
                        "com os parâmetros escolhidos; não acrescentam ajuste de classificador. "
                        "Nenhum registro de validação foi reamostrado. As razões são minoria/maioria, "
                        "não proporção da minoria no total. Não há undersampling."
                    ),
                    nbf.v4.new_code_cell(
                        "display(readable_table(report.table('sampling_counts')))"
                    ),
                ]
            )
    return notebook


def main():
    root = Path(__file__).resolve().parents[1]
    for name in ["xgboost", "random_forest"]:
        path = root / "notebooks" / f"metrics_{name}.ipynb"
        manifests = sorted((root / "reports/experiments").glob(f"train_{name}_*/manifest.json"))
        old_run = manifests[0].parent
        backup = old_run / "notebook_before_smotenc.ipynb"
        if not backup.exists():
            copy2(path, backup)
        notebook = update(nbf.read(path, as_version=4), name)
        for cell in notebook.cells:
            if cell.cell_type == "code":
                cell.outputs, cell.execution_count = [], None
        nbf.validate(notebook)
        nbf.write(notebook, path)


if __name__ == "__main__":
    main()

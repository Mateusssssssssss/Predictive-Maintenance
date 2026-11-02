"""Atualiza as duas apresentações para a busca condicional de SMOTENC e pesos."""

from pathlib import Path
from shutil import copy2

import nbformat as nbf


def upgrade_balance(notebook, name):
    metric = "AUPRC trapezoidal" if name == "xgboost" else "AP"
    weight_parameter = "scale_pos_weight" if name == "xgboost" else "class_weight"
    trees = 150 if name == "xgboost" else 300
    for cell in notebook.cells:
        source = cell.source
        if cell.cell_type == "markdown":
            if source.startswith("## 1. Protocolo"):
                source = f"""## 1. Protocolo e origem das métricas

- Alvo Machine failure; classe positiva 1. Cinco sensores e Type, sem IDs ou modos de falha.
- Mesma reserva da EDA: FIT 6.400, validação posterior 1.600 e TEST reservado 2.000.
- Três folds de prefixos crescentes em FIT. Sem shuffle ou gap; UDI não é data real.
- Critério: maior média de **{metric} entre candidatos com SMOTENC**. Desempate usa primeiro candidato.
- Elegíveis: SMOTENC e SMOTENC + pesos. Controles sem balanceamento e só pesos não podem vencer o refit.
- Imputação e padronização numérica, SMOTENC dentro dos treinos e one-hot depois da síntese.
- Todos os braços usam o mesmo pré-processamento; sampler desligado nos controles.
- A validação conserva seus registros e prevalência. Sem reamostragem na inferência.
- Refit somente em FIT de 6.400. Limiar fixo score >= 0,5; métricas de área usam predict_proba[:, 1].
- AP e AUPRC trapezoidal são métricas distintas. A CV do escolhido tem otimismo de seleção.
- Validação já usada no desenvolvimento; não é confirmação independente. TEST não é avaliado."""
            elif source.startswith("## 2. Grade"):
                source = f"""## 2. Grade condicional, custo e melhores parâmetros

72 candidatos × 3 folds = 216 ajustes de busca, mais 1 refit por modelo.
São 8 controles sem balanceamento, 16 só pesos, 16 SMOTENC e 32 SMOTENC + pesos.
Há ainda baseline e três diagnósticos por tamanho com parâmetros congelados.

{trees} árvores ficam **fixas** nesta rodada para concentrar orçamento no balanceamento.
A grade base tem 8 configurações. Pesos positivos 3 e 10, razões de síntese 0,25 e 0,5,
e k_neighbors=3 fixo (45 falhas no menor treino da CV).

`{weight_parameter}` dá peso maior à falha: XGBoost usa multiplicadores 3/10;
Random Forest usa dicionários {{0: 1, 1: 3}} / {{0: 1, 1: 10}}.
SMOTENC sozinho usa peso neutro. A combinação é comparada, sem presumir que vence.

`mean_test_*` significa validação dos folds em FIT, não o TEST reservado.
Grade condicional completa, incluindo controles e parâmetros, está no manifesto e abaixo."""
            elif source.startswith("A grade investiga"):
                source = (
                    "Os quatro braços usam os mesmos hiperparâmetros base. Razões de síntese significam "
                    "falhas/não falhas após a operação, não a fração da minoria no total. "
                    "A padronização serve às distâncias do sampler. Pesos explícitos evitam o "
                    "recálculo automático de balanced sobre dados já reamostrados. "
                    "Critério Gini/bootstrap no RF e subsampling/regularização no XGBoost ficam fixos. "
                    "Não há early stopping, undersampling ou ajuste de limiar."
                )
            elif "Cada modelo pesquisa 32" in source:
                source = source.replace("32 candidatos", "72 candidatos").replace(
                    "96 ajustes", "216 ajustes"
                )
        else:
            if (
                "display(readable_parameters(manifest['training_config'][MODEL_NAME]['grid']))"
                in source
                and "Grade condicional completa" not in source
            ):
                source = source.replace(
                    "display(readable_parameters(manifest['training_config'][MODEL_NAME]['grid']))",
                    "display(translated_markdown('**Grade condicional completa:**'))\n"
                    "for number, block in enumerate(manifest['search_grid'], start=1):\n"
                    "    display(translated_markdown(f'Bloco {number}'))\n"
                    "    display(readable_parameters(block))\n"
                    "print('Estratégia selecionada:', translate_text(manifest['balance_strategy']))\n"
                    "print('SMOTENC obrigatório no refit:', manifest['requires_smotenc'])",
                )
            # Remove repetições da linha de áreas acumuladas por migrações anteriores.
            repeated = "AUPRC={validation.auprc:.3f}; AUPRC={validation.auprc:.3f}; "
            while repeated in source:
                source = source.replace(repeated, "AUPRC={validation.auprc:.3f}; ")
        cell.source = source
    if not any("## 13. Auditoria do SMOTENC" in c.source for c in notebook.cells):
        notebook.cells.extend(
            [
                nbf.v4.new_markdown_cell(
                    "## 13. Auditoria do SMOTENC\n\nContagens do sampler escolhido por treino "
                    "e refit; auditoria dos folds sem novo ajuste de classificador. "
                    "Validação não é reamostrada."
                ),
                nbf.v4.new_code_cell("display(readable_table(report.table('sampling_counts')))"),
            ]
        )
    if not any("## 14. Estratégias de balanceamento" in c.source for c in notebook.cells):
        notebook.cells.extend(
            [
                nbf.v4.new_markdown_cell(f"""## 14. Estratégias de balanceamento — decisão na CV

Melhor configuração de cada braço segundo {metric}, com dispersão entre folds.
Somente candidatos com SMOTENC são elegíveis para refit, conforme o requisito.
Os controles podem ter score maior; isso será mostrado, sem ocultar ou chamá-los de escolhidos.
As tabelas desta seção são de CV de seleção, não do bloco de validação.
O verde destaca a escolha; cinza identifica controles. Não há confirmação estatística independente."""),
                nbf.v4.new_code_cell(
                    "strategies = report.table('strategy_cv_summary')\n"
                    "metric = report.selection_metric\n"
                    "columns = ['strategy', 'eligible_for_refit', 'selected', 'candidate_index', "
                    "f'mean_test_{metric}', f'std_test_{metric}', 'mean_test_f1', 'mean_test_recall', 'mean_test_precision', 'params']\n"
                    "display(readable_table(strategies.sort_values(f'mean_test_{metric}', ascending=False)[columns]))\n"
                    "report.strategy_figures()\n"
                    "overall = int(manifest['global_best_candidate_index'])\n"
                    "selected = int(manifest['best_candidate_index'])\n"
                    "display(translated_markdown(f'Melhor índice irrestrito: {overall}; selecionado com SMOTENC: {selected}. ' "
                    "'A escolha usa CV, nunca as métricas do bloco de validação.'))"
                ),
            ]
        )
    return notebook


def main():
    root = Path(__file__).resolve().parents[1]
    archive = root / "reports/notebook_snapshots/before_balance_search"
    archive.mkdir(parents=True, exist_ok=True)
    for name in ["xgboost", "random_forest"]:
        path = root / "notebooks" / f"metrics_{name}.ipynb"
        backup = archive / path.name
        if not backup.exists():
            copy2(path, backup)
        notebook = upgrade_balance(nbf.read(path, as_version=4), name)
        for cell in notebook.cells:
            if cell.cell_type == "code":
                cell.outputs, cell.execution_count = [], None
        nbf.validate(notebook)
        nbf.write(notebook, path)


if __name__ == "__main__":
    main()

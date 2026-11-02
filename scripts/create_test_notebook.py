"""Cria o notebook de avaliação final autorizada; execução reutiliza snapshot."""

from pathlib import Path

import nbformat as nbf

root = Path(__file__).resolve().parents[1]
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))


def code(text):
    cells.append(nbf.v4.new_code_cell(text))


md("""# Avaliação final · TEST · XGBoost e Random Forest

Avaliação solicitada pelo usuário após o desenvolvimento. Os dois artefatos,
features, calibrações e limiares estão fixados em `configs/test_evaluation.json`.
Não há treinamento, SMOTENC no TEST, busca de parâmetros ou ajuste de limiar neste notebook.
O pipeline aplica apenas transformações aprendidas no treino e `predict_proba`.
Machine failure=1 representa falha.

TEST corresponde aos UDI 8.001–10.000, reservado antes do desenvolvimento.
UDI não é timestamp verificado: este resultado não comprova antecipação de falha futura.
Após esta leitura, o TEST está revelado. Mudanças motivadas por seus resultados
precisam de outra amostra independente para confirmação final.
""")
code("""from pathlib import Path
import sys
import json
import pandas as pd
from IPython.display import display, Markdown

ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / 'configs/test_evaluation.json').is_file())
sys.path.insert(0, str(ROOT / 'src'))
from predictive_maintenance.test_evaluation import evaluate_test, test_figures
from predictive_maintenance.presentation import readable_table

OUTPUT = evaluate_test(ROOT)
manifest = json.loads((OUTPUT / 'manifest.json').read_text(encoding='utf-8'))
def table(name):
    return pd.read_csv(OUTPUT / f'{name}.csv')
display(Markdown(f"**Snapshot:** `{OUTPUT.relative_to(ROOT)}` · **Registros:** {manifest['test_rows']} · **Falhas:** {manifest['positives']}"))
display(manifest['models'])
assert manifest['refit'] is False
assert manifest['threshold_tuning_on_test'] is False
""")
md("""## Métricas completas e baseline

AP (Average Precision) e AUPRC trapezoidal são áreas diferentes e usam probabilidades.
Precisão, revocação, F1 e matriz de confusão usam o limiar de cada receita, escolhido antes do TEST.
Brier e log loss avaliam probabilidades; menores são melhores. O baseline prevê a prevalência
do bloco de ajuste do respectivo modelo e decide no limiar 0,5. Acurácia isolada é pouco
informativa com classes desbalanceadas. Valores indefinidos são apresentados como tal.
Em scores constantes do baseline, a integração trapezoidal interpola o segmento
entre os extremos e pode produzir área próxima de 0,5; AP é igual à prevalência
(1,95% neste TEST). Essa AUPRC do baseline não significa boa discriminação.
""")
code("""metrics = table('metrics')
display(readable_table(metrics[['model', 'n', 'positives', 'prevalence', 'threshold', 'ap', 'auprc', 'roc_auc', 'f1', 'recall', 'precision']]))
display(readable_table(metrics[['model', 'tn', 'fp', 'fn', 'tp', 'specificity', 'npv', 'fpr', 'fnr', 'alert_rate']]))
display(readable_table(metrics[['model', 'accuracy', 'balanced_accuracy', 'mcc', 'kappa', 'brier', 'log_loss']]))
""")
md("""## Comparação com validação

Os modelos e limiares são os mesmos nas duas partições. Validação foi usada repetidamente
durante o desenvolvimento; TEST é a avaliação final desta receita. As populações e
prevalências podem diferir: uma diferença não isola uma causa nem prova superioridade estatística.
""")
code("""comparison = table('validation_test')
display(readable_table(comparison[['model', 'partition', 'n', 'positives', 'prevalence', 'threshold', 'ap', 'auprc', 'f1', 'recall', 'precision', 'fp', 'fn', 'brier']]))
for name, rows in comparison.groupby('model'):
    before = rows.loc[rows.partition.eq('VALIDATION')].iloc[0]
    final = rows.loc[rows.partition.eq('TEST')].iloc[0]
    display(Markdown(f"**{name}:** F1 {before.f1:.3f} → {final.f1:.3f}; AP {before.ap:.3f} → {final.ap:.3f}; "
                     f"no TEST: {int(final.tp)} falhas detectadas, {int(final.fn)} não detectadas e {int(final.fp)} falsos alertas."))
""")
md("""## Matrizes, curvas PR/ROC, calibração e distribuição dos scores

Calibração no gráfico é apenas descritiva; nenhuma curva ajusta o modelo no TEST.
Histogramas usam eixo vertical simétrico logarítmico para exibir as duas classes.
""")
code("test_figures(OUTPUT)")
md("""## Relatório por classe e por tipo de produto

Os segmentos são diagnósticos descritivos, sem seleção de políticas por segmento.
Poucas falhas podem tornar as estimativas instáveis. O relatório por classe usa a convenção
do sklearn de zero para precisão/F1 indefinidos; a tabela principal preserva indefinidos.
""")
code("""classification = table('classification_report')
# A acurácia global já está na tabela principal; não é uma classe com suporte próprio.
display(readable_table(classification.loc[classification['class'].ne('accuracy')]))
display(readable_table(table('segments_type')))
""")
md("""## Incerteza das métricas

Intervalos percentis de 95% com 500 reamostragens pareadas por registro, semente 42,
sem reajustar modelos. São aproximados e condicionais à receita já treinada.
Pressupõem registros independentes; dependência entre máquinas/ordem pode invalidá-los.
Não cobrem incerteza do treinamento nem demonstram diferença significativa entre modelos.
""")
code("""intervals = table('bootstrap_intervals')
intervals['metric'] = intervals['metric'].replace({'ap': 'AP', 'auprc': 'AUPRC trapezoidal', 'roc_auc': 'ROC-AUC', 'f1': 'F1', 'recall': 'Revocação', 'precision': 'Precisão', 'brier': 'Brier score'})
display(readable_table(intervals))
""")
md("""## Artefatos e proveniência

CSV numéricos, previsões por UDI, composição do TEST, métricas por classe/Type,
comparação com validação, intervalos e figuras ficam no snapshot em `reports/test/`.
O manifesto registra hashes do dataset, modelos e tabelas, além dos limiares originais.
Reexecutar o notebook reutiliza os resultados e verifica sua integridade.
Os manifestos originais de treinamento permanecem como registro histórico anterior ao TEST.
""")
code("display(manifest)")
nb = nbf.v4.new_notebook(
    cells=cells,
    metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
)
nbf.validate(nb)
nbf.write(nb, root / "notebooks/metrics_test.ipynb")
print("Criado notebooks/metrics_test.ipynb")

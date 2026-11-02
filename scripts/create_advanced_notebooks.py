"""Regenera notebooks de métricas com artefatos da rodada Optuna e pós-processamento."""

import textwrap
from pathlib import Path
from shutil import copy2

import nbformat as nbf


def build(root, name):
    cells = []

    def md(source):
        cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(source).strip()))

    def code(source):
        cells.append(nbf.v4.new_code_cell(textwrap.dedent(source).strip()))

    md(f"""# Métricas · {name} · Optuna, derivadas, calibração e limiar

    Relatório de treino e validação. TEST permanece reservado. Os gráficos leem artefatos;
    executar o notebook não repete treinamento. Para nova busca, usar scripts/train_advanced.py.""")
    md("""## 1. Contrato, blocos e mudança de protocolo

    Machine failure=1 é positivo. Usam-se cinco sensores e Type, sem IDs ou modos de falha.
    Desenvolvimento: 8.000 registros, sendo FIT_SEARCH 1–4.800 (221 falhas), CALIBRATION
    4.801–5.600 (27), THRESHOLD 5.601–6.400 (17) e VALIDAÇÃO 6.401–8.000 (35).
    TEST 8.001–10.000 segue reservado. UDI é ordem, não tempo comprovado; gap=0.

    Optuna usa três folds crescentes somente em FIT_SEARCH; critério AUPRC trapezoidal
    no XGBoost e AP no RF. Sem pruning, para avaliar todos os folds de regimes distintos.
    O modelo é reajustado em 4.800, congelado e calibrado por sigmoid em outro bloco;
    depois o limiar é escolhido por F1 em um terceiro bloco, com o modelo calibrado congelado.
    Não há refit após escolher o limiar. AUPRC e AP usam predict_proba[:, 1].

    A rodada anterior ajustava 6.400 registros: a busca/CV e quantidade de ajuste agora
    são diferentes. A validação externa ao ajuste é a mesma, mas já reutilizada no
    desenvolvimento; não é confirmação independente. Mudanças não isolam um único efeito.""")
    code(f"""
    from pathlib import Path
    import sys
    import json
    from IPython.display import display, Markdown

    ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / 'configs/advanced.json').is_file())
    sys.path.insert(0, str(ROOT / 'src'))
    from predictive_maintenance.training import latest_run
    from predictive_maintenance.advanced_reporting import AdvancedReport
    from predictive_maintenance.presentation import readable_table, readable_parameters, translate_text

    def show(text):
        display(Markdown(translate_text(text)))

    MODEL_NAME = '{name}'
    RUN_ID = None  # Informe uma execução para revisar um snapshot específico.
    run = ROOT / 'reports/experiments' / RUN_ID if RUN_ID else latest_run(ROOT, MODEL_NAME)
    report = AdvancedReport(ROOT, run)
    manifest = report.manifest
    assert manifest['experiment_type'] == 'OPTUNA_FEATURES_CALIBRATION_THRESHOLD'
    assert manifest['test_evaluated'] is False
    print('Execução:', manifest['run_id'])
    print('Execução anterior:', manifest['previous_run'])
    print('TEST avaliado:', manifest['test_evaluated'])
    display(manifest['versions'])
    """)
    md("""## 2. Variáveis derivadas e SMOTENC

    Diferença térmica = temperatura de processo − ar (K).
    Potência mecânica = torque × rpm × 2π/60 (W).
    Interação desgaste × torque (min·Nm).
    As fórmulas são determinísticas, calculadas dos sensores dentro do pipeline.
    Não usam o rótulo ou estatísticas globais. Imputação e StandardScaler são ajustados
    em cada treino, seguido de SMOTENC; Type permanece categoria até o one-hot.
    A síntese de derivados pode não preservar identidades físicas exatas.
    A disponibilidade dos sensores antes da decisão real continua uma hipótese a confirmar.""")
    code("""
    display(manifest['derived_features'])
    display(readable_table(report.table('sampling_counts')))
    """)
    md("""## 3. Optuna: orçamento, espaço e escolhido

    24 tentativas TPE por modelo × três folds = 72 fits de busca, mais refit, baseline
    e três fits da ablação. Busca sequencial com semente; folds usam dois workers,
    cada modelo usa uma thread. O espaço inclui pesos, razão/vizinhos do SMOTENC,
    número de árvores, profundidade e regularização. SMOTENC obrigatório em todas as tentativas.
    O escolhido é o melhor da busca realizada, não ótimo global. Arquivo SQLite local
    registra o estudo; CSV registra tentativas completas, inclusive parâmetros e scores.""")
    code("""
    display(manifest['settings']['space'])
    display(readable_parameters(manifest['best_params']))
    print('Critério:', manifest['selection_metric'])
    show(f"Melhor média CV: {manifest['best_cv_score']:.4f}")
    trials = report.table('optuna_trials')
    columns = [c for c in trials.columns if c in ['number','value','duration','state'] or c.startswith('params_')]
    display(readable_table(trials[columns]))
    report.study_figures()
    """)
    md("""## 4. Estatísticas de CV e ablação de features

    A CV do escolhido tem otimismo de seleção, sem CV aninhada. O resumo mostra
    média e desvio amostral, sem interpretar três folds como intervalo de confiança.
    A ablação remove somente as derivadas, mantendo os parâmetros escolhidos. É um
    diagnóstico pareado em CV, não comparação de dois modelos igualmente otimizados.""")
    code("""
    display(readable_table(report.table('fold_metrics')))
    display(readable_table(report.summary()))
    display(readable_table(report.table('feature_ablation')))
    report.ablation_figure()
    """)
    md("""## 5. Calibração e ajuste de limiar

    CalibratedClassifierCV usa sigmoid e FrozenEstimator: base não reaprende com as
    800 observações de calibração. TunedThresholdClassifierCV usa cv=prefit e refit=False
    sobre o modelo calibrado congelado, somente nas outras 800 observações.
    O objetivo provisório é F1, pois custos e capacidade operacional não foram definidos.
    A grade usa scores distintos do bloco THRESHOLD e referências 0/0,5/1.
    Poucas falhas (27 e 17) limitam a estabilidade dessas duas etapas.
    Um sigmoid ajustado não comprova boa calibração no futuro.""")
    code("""
    show(f"Limiar escolhido em TRAIN: {manifest['threshold']:.5f}; F1 no bloco de seleção: {manifest['threshold_selection_f1']:.2%}")
    print('Política:', manifest['threshold_policy'])
    display(readable_table(report.table('threshold_selection')))
    report.threshold_selection_figure()
    """)
    md("""## 6. Etapas na mesma validação, sem seleção posterior

    As três receitas abaixo foram congeladas antes de avaliar este bloco: modelo com
    Optuna/derivadas em limiar 0,5; calibrado em limiar 0,5; calibrado com limiar escolhido.
    Essas métricas não escolhem uma nova versão vencedora. A calibração sigmoid pode
    preservar a ordenação e as áreas, mudando a escala dos scores. Alterar apenas
    o limiar não altera AP/AUPRC dos mesmos scores.""")
    code("""
    display(readable_table(report.table('stage_metrics')))
    report.stage_figures()
    """)
    md("""## 7. Modelo final versus baseline e matriz

    FIT_RESUBSTITUTION é desempenho no próprio ajuste, sujeito a sobreajuste.
    VALIDATION_TRAIN_BLOCK é o bloco posterior de 1.600. Baseline usa threshold 0,5;
    a receita final usa o limiar aprendido em TRAIN. A tabela identifica ambos.
    Scores de área em 0–1, taxas em %, ausências como Não definido. Não há métricas TEST.""")
    code("""
    metrics = report.table('model_metrics')
    display(readable_table(metrics))
    report.comparison()
    display(readable_table(report.table('classification_report')))
    report.confusion()
    """)
    md("""## 8. PR/AUPRC/AP, ROC e diagnóstico probabilístico

    AP não é AUPRC trapezoidal. Scores constantes podem ter área trapezoidal alta por
    interpolação de poucos pontos; AP do baseline é a prevalência. Brier/Log Loss e
    curva de confiabilidade complementam a avaliação; não garantem calibração geral.""")
    code("""
    report.probability_figures()
    display(readable_table(report.table('calibration')))
    """)
    md("""## 9. Sensibilidade na validação — somente diagnóstico

    O limiar destacado já veio do bloco THRESHOLD de TRAIN. A varredura abaixo não
    modifica o pacote nem é usada para escolher novo limiar nesta validação.""")
    code("""
    display(readable_table(report.table('threshold_diagnostics')))
    report.threshold_figure()
    """)
    md("""## 10. Associações, importância e erros

    Associação no FIT_SEARCH; permutação na validação como diagnóstico posterior,
    sem seleção de variáveis. Permutar um sensor também afeta as derivadas dele no
    pipeline completo: a importância se refere à entrada original, não isola uma derivada.
    Grupos e faixas incluem suporte; limites são aprendidos no ajuste, sem modificar TEST.""")
    code("""
    display(readable_table(report.table('feature_association_importance')))
    report.feature_figure()
    display(readable_table(report.table('segment_type')))
    display(readable_table(report.table('segment_numeric')))
    report.error_figure()
    """)
    md("""## 11. Comparação com a rodada anterior

    Mesmas observações e alvos de validação, CSV original preservado. Busca, tamanho
    do ajuste, features, calibração e limiar mudaram juntos: não atribuir uma diferença
    somente ao Optuna ou à engenharia. As etapas não foram escolhidas pela validação.
    O pacote anterior continua disponível; a versão atual não é confirmação independente.""")
    code("""
    comparison = report.table('previous_comparison')
    display(readable_table(comparison[['version','threshold','auprc','ap','f1','recall','precision','brier','log_loss','fp','fn']]))
    report.previous_figure()
    before, after = comparison.iloc[0], comparison.iloc[-1]
    show(f"F1: {before.f1:.2%} → {after.f1:.2%}; Recall: {before.recall:.2%} → {after.recall:.2%}; "
         f"Precision: {before.precision:.2%} → {after.precision:.2%}. "
         f"AP: {before.ap:.4f} → {after.ap:.4f}; AUPRC: {before.auprc:.4f} → {after.auprc:.4f}.")
    """)
    md("""## 12. Artefato e proveniência

    O pacote inclui derivadas, preparação, SMOTENC, classificador, calibrador e limiar.
    Carregar apenas joblib de origem confiável. Inferência não reamostra ou reajusta.
    Base treinada em 4.800, calibrador em 800, limiar em 800; não há refit posterior.""")
    code("""
    print('Pipeline:', manifest['model_path'])
    print('SHA256:', manifest['model_sha256'])
    print('Equivalência após serialização:', manifest['serialization_verified'])
    print('Tabelas:', report.tables.relative_to(ROOT))
    print('Figuras:', report.figures.relative_to(ROOT))
    display(manifest['limitations'])
    """)
    notebook = nbf.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3",
            }
        },
    )
    path = root / "notebooks" / f"metrics_{name}.ipynb"
    archive = root / "reports/notebook_snapshots/before_advanced"
    archive.mkdir(parents=True, exist_ok=True)
    if path.exists() and not (archive / path.name).exists():
        copy2(path, archive / path.name)
    nbf.validate(notebook)
    temporary = path.with_name(f".{path.stem}.advanced.ipynb")
    nbf.write(notebook, temporary)
    temporary.replace(path)


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    for name in ["xgboost", "random_forest"]:
        build(root, name)

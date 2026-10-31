"""Cria um notebook revisável por modelo sem sobrescrever os existentes."""

import textwrap
from pathlib import Path

import nbformat as nbf
from localize_model_notebooks import localize


def build(root: Path, name: str) -> None:
    cells = []

    def md(source):
        cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(source).strip()))

    def code(source):
        cells.append(nbf.v4.new_code_cell(textwrap.dedent(source).strip()))

    display_name = "XGBoost" if name == "xgboost" else "Random Forest"
    md(f"""
    # Métricas e treinamento · {display_name}

    GridSearchCV reproduzível, baseline e diagnósticos de **treino e validação**.
    **TEST não é avaliado neste notebook.** Os gráficos leem artefatos persistidos;
    abrir ou executar o notebook com a configuração padrão não repete a busca.
    """)
    md("""
    ## 1. Protocolo e origem das métricas

    - Alvo `Machine failure`, positivo `1`. Entradas: cinco sensores e `Type`.
    - Mesma reserva da EDA: 8.000 registros de desenvolvimento e 2.000 em TEST reservado.
    - Dentro dos 8.000: prefixo **FIT de 6.400** para GridSearchCV, seguido de **1.600 em VALIDAÇÃO**.
    - Três folds com prefixos crescentes dentro de FIT, sem shuffle ou gap. UDI é ordem,
      não data real; não se demonstra independência temporal nem entre máquinas.
    - **Critério:** maior média de AP na CV; empate exato usa primeiro candidato da grade.
      AP é Average Precision, não integração trapezoidal da PR. O vencedor é o melhor
      da grade pesquisada, sem garantia de ótimo global.
    - **Refit:** melhores parâmetros reajustados apenas nas 6.400 observações FIT.
      A validação de 1.600 não participa do ajuste nem do GridSearchCV.
    - **Decisão:** `score >= 0.5`, referência fixa e não limiar operacional ótimo.
    - Imputação e one-hot dentro do pipeline/folds; árvores sem scaling. Sem SMOTE,
      undersampling, seleção automática de features, calibração ajustada ou early stopping.
    - TRAIN completo já participou da EDA: a validação é interna ao desenvolvimento,
      não uma confirmação independente. Os scores do vencedor na CV têm otimismo de seleção.
    """)
    code(f"""
    from pathlib import Path
    import sys
    import json
    from IPython.display import display, Markdown

    ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents)
                if (p / 'configs/training.json').is_file() and (p / 'pyproject.toml').is_file())
    sys.path.insert(0, str(ROOT / 'src'))
    from predictive_maintenance.training import run_experiment, latest_run
    from predictive_maintenance.reporting import ModelReport
    from predictive_maintenance.data import sha256

    MODEL_NAME = '{name}'
    RUN_TRAINING = False  # True executa uma nova busca e preserva as anteriores.
    RUN_ID = None  # Informe um ID salvo para revisar exatamente aquela execução.
    if RUN_TRAINING:
        run = run_experiment(ROOT, MODEL_NAME)
    elif RUN_ID:
        run = ROOT / 'reports/experiments' / RUN_ID
    else:
        run = latest_run(ROOT, MODEL_NAME)
    report = ModelReport(ROOT, run)
    manifest = report.manifest
    if manifest['model'] != MODEL_NAME:
        raise ValueError('Execução pertence a outro modelo.')
    print('Execução:', manifest['run_id'])
    print('Dataset SHA256:', manifest['dataset_sha256'])
    print('CSV atual corresponde:', sha256(ROOT / manifest['data_config']['source']) == manifest['dataset_sha256'])
    print('TEST avaliado:', manifest['test_evaluated'])
    display(manifest['versions'])
    """)
    md("""
    ## 2. Grade, custo e melhores parâmetros

    Cada modelo pesquisa 32 candidatos × 3 folds = 96 ajustes de busca, mais um refit.
    Há ainda 4 ajustes do baseline e 3 diagnósticos de tamanho com parâmetros congelados.
    A busca usa 2 workers e cada estimador usa 1 thread para limitar concorrência.

    **Nomenclatura sklearn:** `mean_test_*` e `split*_test_*` nas tabelas da busca
    significam validação dos folds dentro de FIT, nunca o conjunto TEST reservado.
    """)
    if name == "xgboost":
        md("""
        A grade investiga número de árvores × learning rate, profundidade,
        `min_child_weight` e peso positivo. `scale_pos_weight=10` é uma hipótese fixa
        moderada frente a 1, não uma razão calculada sobre validação/TEST.
        `subsample=0.8`, `colsample_bytree=0.9`, L1=0 e L2=1 são **fixos**, não otimizados.
        Não há early stopping; o número de árvores participa da busca conjunta.
        """)
    else:
        md("""
        A grade investiga quantidade de árvores, profundidade, suporte mínimo de folhas,
        número de features por divisão e `class_weight`. `balanced` calcula os pesos
        com os rótulos do treino de cada fit. Critério Gini, bootstrap e
        `min_samples_split=2` são **fixos**, não otimizados nesta rodada.
        """)
    code("""
    display(manifest['training_config'][MODEL_NAME]['grid'])
    display(Markdown('**Parâmetros fixos:**'))
    display(manifest['fixed_params'])
    display(Markdown('**Melhores parâmetros na grade:**'))
    display(manifest['best_params'])
    print(f"AP média de seleção: {manifest['best_cv_ap']:.4f}")
    print(f"Tempo da busca: {manifest['search_seconds']:.1f} s")
    grid = report.table('grid_results')
    columns = ['rank_test_ap', 'mean_test_ap', 'std_test_ap', 'mean_train_ap',
               'mean_test_f1', 'mean_test_recall', 'mean_test_precision', 'mean_fit_time', 'params']
    display(grid.sort_values('rank_test_ap')[columns].head(10))
    report.cv_figures()
    """)
    md("""
    **Leitura dos perfis:** cada barra mostra o melhor AP médio entre combinações
    daquele valor, mantendo os demais parâmetros livres na grade. Não é efeito
    isolado de hiperparâmetro. Os desvios do top 10 são dispersão entre folds,
    não intervalos de confiança ou evidência de superioridade estatística.
    """)
    md("""
    ## 3. Estatísticas por fold e baseline

    `CV_SELECTION` refere-se ao candidato escolhido nas mesmas amostras usadas na
    seleção, sem CV aninhada. Não são predições OOF nem resultado de teste.
    O baseline Dummy usa a proporção observada no treino de cada fold.
    Precision sem alertas é 0 no scorer da CV; nos diagnósticos de segmentos é indefinida (NaN).
    """)
    code("""
    folds = report.table('fold_metrics')
    display(folds)
    display(report.summary())
    selected_folds = folds.loc[folds.model.eq(MODEL_NAME)]
    display(Markdown(f"**AP de seleção:** {selected_folds.validation_ap.mean():.3f} ± "
                    f"{selected_folds.validation_ap.std(ddof=1):.3f} (desvio amostral entre folds). "
                    'Deslocamentos entre blocos e baixo suporte positivo limitam a interpretação.'))
    """)
    md("""
    ## 4. FIT reajustado versus VALIDAÇÃO

    `FIT_RESUBSTITUTION` é desempenho sobre o próprio ajuste, sujeito a sobreajuste.
    `VALIDATION_TRAIN_BLOCK` é o bloco posterior fora da busca. Mesmos dados e limiar
    para o candidato e o baseline. Scores apresentados em 0–1, contagens em inteiros.
    `indef.` nas barras indica ausência de alertas; a altura 0 é apenas convenção visual.
    """)
    code("""
    metrics = report.table('model_metrics')
    display(metrics)
    report.comparison()
    validation = metrics.loc[metrics.model.eq(MODEL_NAME) & metrics.origin.eq('VALIDATION_TRAIN_BLOCK')].iloc[0]
    fit = metrics.loc[metrics.model.eq(MODEL_NAME) & metrics.origin.eq('FIT_RESUBSTITUTION')].iloc[0]
    display(Markdown(f"**Validação:** AP={validation.ap:.3f}; F1={validation.f1:.3f}; "
        f"Recall={validation.recall:.3f}; Precision={validation.precision:.3f}. "
        f"Diferença de AP entre ajuste e validação = {fit.ap - validation.ap:.3f}. "
        f"São {int(validation.positives)} falhas em {int(validation.n)} observações. "
        'A diferença pode refletir sobreajuste e mudança entre blocos; não prova causalidade.'))
    """)
    md("""
    ## 5. Matriz de confusão e relatório por classe

    Eixo vertical real, horizontal previsto; classe positiva = falha. Matrizes em
    contagens e normalizadas pela classe real. `zero_division=0` no relatório por classe.
    """)
    code("""
    display(report.table('classification_report'))
    report.confusion()
    display(Markdown(f"**Carga de revisão no bloco:** {int(validation.tp + validation.fp)} alertas; "
                    f"{int(validation.fp)} falsos positivos e {int(validation.fn)} falhas perdidas. "
                    'Custos e capacidade operacional ainda não foram informados.'))
    """)
    md("""
    ## 6. PR, ROC, confiabilidade e distribuição dos scores

    AP e ROC usam scores positivos. A prevalência é a referência da PR. ROC complementa
    a análise de classe rara. A curva de confiabilidade usa até 10 grupos por quantis;
    não ajusta calibrador e não comprova calibração. Pesos de classe podem alterar os scores.
    """)
    code("""
    report.probability_figures()
    display(report.table('calibration'))
    display(Markdown(f"**Diagnóstico probabilístico:** Brier={validation.brier:.4f}; "
                    f"Log Loss={validation.log_loss:.4f}. "
                    'Amostra com poucas falhas limita a estabilidade da curva de confiabilidade.'))
    """)
    md("""
    ## 7. Sensibilidade ao threshold — sem tuning

    O destaque mostra a referência 0,5. Varrer limiares na validação é **diagnóstico**,
    sem mudar o pacote ou declarar um ótimo. Recall/Precision/F1 mudam; AP dos mesmos
    scores não muda. Uma escolha operacional futura exige objetivo e validação própria.
    """)
    code("""
    display(report.table('threshold_diagnostics'))
    report.threshold_figure()
    """)
    md("""
    ## 8. Associação versus importância preditiva

    Associação calculada em FIT: Spearman para sensores, Cramér V para Type (magnitude
    com medidas distintas, não equivalentes). Importância por permutação no bloco de
    validação, agregada por entrada original passando pelo pipeline completo.
    A unidade é queda de AP; as barras de erro são desvio das permutações, não confiança.
    Correlação entre sensores pode repartir importâncias. Nada foi removido com esse diagnóstico.
    """)
    code("""
    display(report.table('feature_association_importance'))
    report.feature_figure()
    """)
    md("""
    ## 9. Diagnóstico por tamanho do prefixo

    Parâmetros congelados da busca, prefixos ordenados e a mesma validação. Prefixos
    de tamanhos distintos abrangem regimes diferentes: **não é uma learning curve i.i.d.
    controlada**. É diagnóstico descritivo; não escolhe novamente os parâmetros.
    """)
    code("""
    display(report.table('learning_diagnostic'))
    report.learning_figure()
    """)
    md("""
    ## 10. Erros por segmento e ordem dos registros

    Contagens, taxas reais, TP/FP/FN/TN e métricas por Type e faixas de sensores.
    Limites das faixas aprendidos apenas em FIT; extremos cobrem valores futuros.
    NaN indica denominador inexistente. Não somar suportes de features diferentes:
    as mesmas observações aparecem em cada análise. UDI não representa tempo decorrido.
    """)
    code("""
    display(report.table('segment_type'))
    display(report.table('segment_numeric'))
    report.error_figure()
    """)
    md("""
    ## 11. Artefato, reprodutibilidade e limites

    O pipeline local corresponde ao refit em FIT (6.400), não a um modelo reajustado
    nas 8.000 observações ou selecionado como vencedor do projeto. Não existe avaliação TEST.
    O binário é local e ignorado no Git; carregar somente artefatos de origem confiável.
    """)
    code("""
    print('Pipeline:', manifest['model_path'])
    print('SHA256:', manifest['model_sha256'])
    print('Equivalência após serialização:', manifest['serialization_verified'])
    print('Política de limiar:', manifest['threshold_policy'])
    print('Figuras:', report.figures.relative_to(ROOT))
    print('Tabelas:', report.tables.relative_to(ROOT))
    display(manifest['limitations'])
    display(Markdown('**Próxima decisão:** comparar os dois candidatos sob este mesmo protocolo, '
        'considerando AP, falhas perdidas, falsos alarmes, estabilidade e custo. '
        'Não ajustar features ou threshold com TEST. Nenhum vencedor de produção foi declarado.'))
    """)
    notebook = localize(nbf.v4.new_notebook(cells=cells))
    notebook.metadata.kernelspec = {
        "display_name": "Python 3 (ipykernel)",
        "language": "python",
        "name": "python3",
    }
    destination = root / "notebooks" / f"metrics_{name}.ipynb"
    if destination.exists():
        raise FileExistsError(f"Preserve o notebook existente: {destination}")
    nbf.write(notebook, destination)


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    for model_name in ["xgboost", "random_forest"]:
        build(project_root, model_name)

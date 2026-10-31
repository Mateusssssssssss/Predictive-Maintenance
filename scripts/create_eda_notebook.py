"""Gera o notebook inicial; recusa sobrescrever edições existentes."""

import json
import textwrap
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    cells = []

    def cell(kind: str, source: str) -> None:
        entry = {
            "cell_type": kind,
            "metadata": {},
            "source": textwrap.dedent(source).strip() + "\n",
            "id": f"eda-{len(cells):02d}",
        }
        if kind == "code":
            entry.update(execution_count=None, outputs=[])
        cells.append(entry)

    def md(source: str) -> None:
        cell("markdown", source)

    def code(source: str) -> None:
        cell("code", source)

    md("""
    # 01 · Análise exploratória — manutenção preditiva

    **Objetivo:** avaliar qualidade, distribuições, desbalanceamento e hipóteses para
    futura modelagem. Nenhum classificador é treinado aqui. As saídas são calculadas
    a partir do CSV local e persistidas em relatórios por execução.
    """)
    md("""
    ## 1. Contrato do problema e limites

    - **Unidade:** observação operacional; cada linha não identifica necessariamente
      uma máquina física independente.
    - **Alvo:** `Machine failure`; positivo `1` = falha registrada naquela observação.
    - **Momento pretendido:** sensores disponíveis antes da decisão de manutenção.
      O CSV não comprova antecedência nem fornece horizonte futuro ou datas.
      A classificação do estado de falha não demonstra antecipação de falha futura.
    - **População:** registros locais; aplicação industrial real não validada.
    - **Origem provável:** schema compatível com AI4I 2020, base sintética descrita pela
      [UCI](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset).
      A identidade byte a byte com a distribuição oficial não foi verificada.
    - **Métrica futura proposta:** Average Precision (AP) para ordenação, complementada
      por Recall, Precision e F1. Custos de erros e capacidade de alertas estão em aberto.
    - **Vazamento:** TWF/HDF/PWF/OSF/RNF são rótulos auxiliares, excluídos dos preditores.
      UDI e Product ID também ficam fora das entradas de futuros modelos.
    """)
    code("""
    from pathlib import Path
    import sys
    import json
    import platform
    import importlib.metadata
    from datetime import datetime, timezone
    from uuid import uuid4
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from IPython.display import display, Markdown

    ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents)
                if (p / 'configs/data.json').is_file() and (p / 'pyproject.toml').is_file())
    sys.path.insert(0, str(ROOT / 'src'))
    from predictive_maintenance.data import load_config, load_data, development_split, sha256
    from predictive_maintenance.eda import quality_table, outlier_table, outcome_rates

    sns.set_theme(style='whitegrid', context='notebook')
    plt.rcParams.update({'figure.dpi': 110, 'axes.spines.top': False, 'axes.spines.right': False})
    COLORS = {0: '#3977a8', 1: '#d96540'}
    config = load_config(ROOT)
    TARGET = config['target']
    NUM = config['numeric_features']
    FEATURES = NUM + config['categorical_features']
    RUN_ID = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid4().hex[:8]
    FIGURES = ROOT / 'reports/figures' / RUN_ID
    TABLES = ROOT / 'reports/metrics' / RUN_ID
    EXPERIMENT = ROOT / 'reports/experiments' / RUN_ID
    for path in (FIGURES, TABLES, EXPERIMENT):
        path.mkdir(parents=True, exist_ok=False)

    def save_table(table, name):
        table.to_csv(TABLES / f'{name}.csv', encoding='utf-8', index=True)
        display(table)

    def finish_figure(fig, name):
        fig.tight_layout()
        fig.savefig(FIGURES / f'{name}.png', dpi=160, bbox_inches='tight')
        plt.show()
        plt.close(fig)

    print(f'Execução: {RUN_ID} | Python {platform.python_version()}')
    """)
    md("""
    ## 2. Schema e reserva de TEST

    Reservar o bloco final na ordem UDI, sem embaralhar, é uma escolha inicial
    conservadora para sinais sequenciais. **UDI não é timestamp.** A fração 80/20
    deixa a maior parte para investigação e 2.000 registros reservados no arquivo
    atual; não foi escolhida por desempenho ou prevalência de TEST.

    Sem horizonte ou duração conhecida da dependência, não há gap. Revisar divisão,
    gap e validação por blocos antes da modelagem; não adotar CV aleatória por padrão.
    A leitura global valida apenas o schema. Todas as distribuições e relações abaixo
    usam **TRAIN**. TEST não é explorado.
    """)
    code("""
    raw = load_data(ROOT, config)
    train, reserved_test = development_split(raw, config)
    split = pd.concat([
        train[['UDI']].assign(partition='TRAIN'),
        reserved_test[['UDI']].assign(partition='TEST'),
    ], ignore_index=True)
    split.to_csv(EXPERIMENT / 'split_membership.csv', index=False)
    summary = pd.DataFrame({
        'conjunto': ['CSV completo (somente estrutura)', 'TRAIN explorado', 'TEST reservado'],
        'linhas': [len(raw), len(train), len(reserved_test)],
        'colunas': [raw.shape[1], train.shape[1], reserved_test.shape[1]],
    }).set_index('conjunto')
    save_table(summary, 'partition_sizes')
    dataset_hash = sha256(ROOT / config['source'])
    print('SHA256 do CSV:', dataset_hash)
    del raw, reserved_test  # Evitar exploração acidental do TEST.
    """)
    md("""
    ## 3. Dicionário, tipos e disponibilidade

    `Type` representa variante de qualidade L/M/H. Os modos representam desgaste,
    dissipação térmica, potência, sobrecarga e falha aleatória; servem somente à
    auditoria dos rótulos. A disponibilidade dos sensores antes da decisão precisa
    ser confirmada para uma aplicação real.
    """)
    code("""
    dictionary = pd.read_csv(ROOT / 'docs/data_dictionary.csv').set_index('feature')
    save_table(dictionary.join(quality_table(train)), 'data_dictionary_and_quality')
    display(train[FEATURES + [TARGET]].head(8))
    """)
    md("""
    ## 4. Qualidade e plausibilidade

    Duplicação de linhas completas, entradas e identificadores são conceitos distintos.
    Entradas iguais com alvos diferentes merecem auditoria. Extremos de 1,5×IQR
    sinalizam caudas; não são necessariamente erros e não serão removidos.
    Limites físicos específicos da máquina não foram fornecidos.
    """)
    code("""
    repeated = train.duplicated(FEATURES, keep=False)
    contradictory = train.groupby(FEATURES, dropna=False)[TARGET].nunique().gt(1).sum()
    checks = pd.Series({
        'linhas_duplicadas_completas': int(train.duplicated().sum()),
        'linhas_com_entradas_repetidas': int(repeated.sum()),
        'grupos_com_alvos_contraditorios': int(contradictory),
        'UDI_repetido': int(train['UDI'].duplicated().sum()),
        'Product_ID_repetido': int(train['Product ID'].duplicated().sum()),
        'celulas_ausentes': int(train.isna().sum().sum()),
        'processo_mais_frio_que_ar': int((train[NUM[1]] < train[NUM[0]]).sum()),
        'temperatura_absoluta_nao_positiva': int((train[NUM[:2]] <= 0).any(axis=1).sum()),
        'rotacao_nao_positiva': int((train['Rotational speed [rpm]'] <= 0).sum()),
        'tipo_diverge_prefixo': int(train['Product ID'].astype('string').str[0].ne(train['Type']).sum()),
    }, name='contagem').to_frame()
    save_table(checks, 'quality_checks')
    extremes = outlier_table(train, NUM)
    save_table(extremes, 'outlier_flags')
    save_table(train[NUM].describe().T, 'numeric_summary')
    display(Markdown(f"**Leitura:** {checks.loc['celulas_ausentes', 'contagem']:,} células ausentes; "
        f"{int(extremes['flagged'].sum()):,} sinalizações IQR somadas entre variáveis "
        '(uma linha pode aparecer várias vezes). Nenhum registro foi removido.'))
    """)
    md("""
    ### 4.1. Consistência dos rótulos

    A união dos modos deveria ser confrontada com a definição do alvo. Divergências
    não serão corrigidas automaticamente. Os modos podem coexistir: suas contagens
    não devem ser somadas como falhas distintas.
    """)
    code("""
    any_mode = train[config['failure_modes']].eq(1).any(axis=1)
    audit = pd.crosstab(train[TARGET].rename('Machine failure'), any_mode.rename('algum_modo_ativo'))
    save_table(audit, 'label_consistency')
    save_table(train[config['failure_modes']].sum().rename('ocorrencias').to_frame(), 'failure_modes_train')
    mismatches = int(any_mode.ne(train[TARGET].eq(1)).sum())
    display(Markdown(f'**Leitura:** {mismatches} registros de TRAIN divergem entre alvo e '
                     'união dos modos. Confirmar origem e política de anotação antes de corrigir.'))
    """)
    md("""
    ## 5. Desbalanceamento do alvo

    O denominador é o número de observações em TRAIN. Acurácia isolada pode ocultar
    falhas perdidas; ainda não há avaliação de classificador.
    """)
    code("""
    counts = train[TARGET].value_counts().reindex([0, 1], fill_value=0)
    classes = counts.rename('n').to_frame()
    classes['fraction'] = classes['n'] / len(train)
    save_table(classes, 'target_distribution')
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(['Sem falha (0)', 'Falha (1)'], counts, color=[COLORS[0], COLORS[1]])
    ax.bar_label(bars, labels=[f'{n:,} ({n / len(train):.2%})' for n in counts], padding=4)
    ax.set(ylabel='Observações de TRAIN', title='Alvo — TRAIN', ylim=(0, counts.max()*1.18))
    finish_figure(fig, 'target_distribution')
    prevalence = float(train[TARGET].mean())
    display(Markdown(f'**Leitura:** prevalência em TRAIN = **{prevalence:.2%}**. '
                     'Considerar AP, Recall e Precision na futura modelagem.'))
    """)
    md("""
    ## 6. Distribuições dos sensores

    As densidades são normalizadas por classe para comparar forma, sem representar
    a prevalência populacional. Boxplots mostram quartis e extremos, sem exclusões.
    """)
    code("""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for feature, ax in zip(NUM, axes.flat):
        sns.histplot(data=train, x=feature, hue=TARGET, hue_order=[0, 1],
                     palette=COLORS, bins=35, stat='density', common_norm=False,
                     element='step', ax=ax)
        ax.set_title(feature)
    axes.flat[-1].set_visible(False)
    finish_figure(fig, 'sensor_distributions_by_target')
    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    for feature, ax in zip(NUM, axes):
        sns.boxplot(data=train, x=TARGET, y=feature, color='#9bc0d9', ax=ax)
        ax.set_xlabel('Falha (0/1) — TRAIN')
    finish_figure(fig, 'sensor_boxplots')
    medians = train.groupby(TARGET)[NUM].median().T
    save_table(medians, 'sensor_medians_by_class')
    display(Markdown('**Leitura:** compare medianas e caudas na tabela acima. '
        'A sobreposição das classes impede inferir uma separação perfeita a partir '
        'de um único sensor; diferenças descritivas não medem desempenho fora da amostra.'))
    """)
    md("""
    ## 7. Taxas por categoria e faixas

    Taxa = falhas / observações do grupo. Exibir denominadores para não interpretar
    grupos pequenos como padrões estáveis. Quantis são aprendidos só em TRAIN,
    para exploração; em modelagem, aprender faixas dentro dos folds.
    """)
    code("""
    type_rates = outcome_rates(train, 'Type', TARGET)
    save_table(type_rates, 'rates_by_type')
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(type_rates.index.astype(str), type_rates['failure_rate'] * 100, color='#3977a8')
    ax.bar_label(bars, labels=[f'{r.failure_rate:.1%} (n={r.n})' for r in type_rates.itertuples()], padding=4)
    ax.set(xlabel='Type', ylabel='Falhas / observações (%)', title='Taxa por tipo — TRAIN')
    ax.margins(y=0.25)
    finish_figure(fig, 'rates_by_type')

    rate_tables = []
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for feature, ax in zip(NUM, axes.flat):
        bins = pd.qcut(train[feature], q=5, duplicates='drop')
        local = train.assign(faixa=bins if bins.notna().any() else 'constante/ausente')
        rates = outcome_rates(local, 'faixa', TARGET)
        rate_tables.append(rates.reset_index().assign(feature=feature))
        bars = ax.bar(np.arange(len(rates)), rates['failure_rate'] * 100, color='#3977a8')
        ax.bar_label(bars, labels=[f'{r.failure_rate:.1%}\\nn={r.n}' for r in rates.itertuples()], fontsize=8, padding=3)
        labels = [f'{v.left:.1f} a {v.right:.1f}' if isinstance(v, pd.Interval) else str(v) for v in rates.index]
        ax.set_xticks(np.arange(len(rates)), labels, rotation=35, ha='right', fontsize=8)
        ax.set(title=feature, ylabel='Taxa de falha (%)')
        ax.margins(y=0.35)
    axes.flat[-1].set_visible(False)
    finish_figure(fig, 'rates_by_numeric_bins')
    save_table(pd.concat(rate_tables, ignore_index=True), 'rates_by_numeric_bins')
    display(Markdown('**Leitura:** rótulos das faixas arredondados para uma casa; '
        'intervalos exatos estão na tabela. Variação nas taxas sugere hipóteses de '
        'associação e não regras causais. As mesmas observações aparecem em cada '
        'feature: não somar denominadores entre variáveis.'))
    """)
    md("""
    ## 8. Associações e redundância — TRAIN

    Pearson mede associação linear; Spearman, monotônica. Com alvo binário, Pearson
    é ponto-bisserial. `Type` não recebe código numérico arbitrário. Relações fracas
    não descartam interações, e associação não demonstra causalidade.

    **Associação versus importância:** os painéis comparam duas associações.
    Importância preditiva fica adiada porque requer modelo e validação. Quando houver
    modelo, comparar associação com permutation importance em validação interna de TRAIN,
    em painéis alinhados. Correlação não substitui importância.
    """)
    code("""
    association = pd.DataFrame({
        'pearson': train[NUM + [TARGET]].corr(method='pearson')[TARGET].drop(TARGET),
        'spearman': train[NUM + [TARGET]].corr(method='spearman')[TARGET].drop(TARGET),
    }).sort_values('spearman', key=lambda s: s.abs())
    save_table(association, 'feature_target_associations')
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for method, ax in zip(association.columns, axes):
        bars = ax.barh(association.index, association[method], color='#3977a8')
        ax.bar_label(bars, fmt='%.3f', padding=4)
        ax.axvline(0, color='gray', linewidth=1)
        ax.set(title=f'{method.title()} com alvo — TRAIN', xlabel='Coeficiente [-1, 1]', xlim=(-1, 1))
    finish_figure(fig, 'target_associations')
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(train[NUM].corr(method='spearman'), annot=True, fmt='.2f', cmap='vlag',
                center=0, vmin=-1, vmax=1, ax=ax)
    ax.set_title('Redundância entre sensores — Spearman / TRAIN')
    finish_figure(fig, 'sensor_correlation')
    strongest = association['spearman'].abs().idxmax()
    display(Markdown(f'**Leitura:** maior associação monotônica em módulo: `{strongest}` '
                     f'({association.loc[strongest, "spearman"]:.3f}). '
                     'Isso não determina utilidade preditiva nem seleção automática.'))
    """)
    md("""
    ## 9. Interações e ordem dos registros

    Investigar regimes de torque/rotação e diferença térmica. Eles podem refletir
    mecanismos sintéticos. UDI indica ordem de registro, **não tempo transcorrido**.
    Blocos são descritivos; não são folds ou janelas temporais já validados.
    """)
    code("""
    derived = train.assign(temperature_gap_K=train[NUM[1]] - train[NUM[0]])
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for label in (0, 1):
        group = derived.loc[derived[TARGET].eq(label)]
        axes[0].scatter(group['Rotational speed [rpm]'], group['Torque [Nm]'],
                        s=9 if label == 0 else 22, alpha=0.2 if label == 0 else 0.8,
                        color=COLORS[label], label=f'Falha={label}')
        axes[1].scatter(group['temperature_gap_K'], group['Rotational speed [rpm]'],
                        s=9 if label == 0 else 22, alpha=0.2 if label == 0 else 0.8,
                        color=COLORS[label], label=f'Falha={label}')
    axes[0].set(xlabel='Rotação (rpm)', ylabel='Torque (Nm)', title='Rotação × torque — TRAIN')
    axes[1].set(xlabel='Temperatura processo − ar (K)', ylabel='Rotação (rpm)', title='Diferença térmica × rotação — TRAIN')
    for ax in axes:
        ax.legend()
    finish_figure(fig, 'sensor_interactions')

    ordered = train.sort_values('UDI').assign(bloco=lambda d: np.arange(len(d)) // 500 + 1)
    block_rates = outcome_rates(ordered, 'bloco', TARGET)
    save_table(block_rates, 'rates_by_record_block')
    fig, axes = plt.subplots(2, 1, figsize=(13, 7))
    for feature in NUM[:2]:
        axes[0].plot(ordered['UDI'], ordered[feature], linewidth=0.8, label=feature)
    axes[0].set(xlabel='UDI (ordem)', ylabel='Temperatura (K)', title='Sinais na ordem original — TRAIN')
    axes[0].legend()
    bars = axes[1].bar(block_rates.index.astype(str), block_rates['failure_rate'] * 100, color='#3977a8')
    axes[1].bar_label(bars, labels=[f'{r.failure_rate:.1%}\\nn={r.n}' for r in block_rates.itertuples()], fontsize=8, padding=3)
    axes[1].set(xlabel='Bloco de até 500 registros em TRAIN', ylabel='Taxa de falha (%)')
    axes[1].margins(y=0.3)
    finish_figure(fig, 'ordered_signal_and_block_rates')
    """)
    md("""
    ## 10. Síntese para revisão e proveniência

    As conclusões são calculadas em TRAIN. Definição do problema e protocolo
    precisam ser revisados antes de modelar e acessar o conjunto reservado.
    """)
    code('''
    findings = f"""
    - TRAIN contém **{len(train):,} observações** e **{int(train[TARGET].sum()):,} falhas ({prevalence:.2%})**.
    - **{int(train.isna().sum().sum())} células ausentes** e **{int(train.duplicated(FEATURES).sum())} entradas duplicadas** além da primeira ocorrência.
    - **{mismatches} divergências de rótulo** entre alvo e união dos modos; sem correções automáticas.
    - **{int(extremes['flagged'].sum())} sinalizações IQR** somadas entre variáveis; nenhum registro removido.
    - Maior associação Spearman em módulo: **{strongest}**; investigar interações antes de seleção.
    - TEST permanece reservado. Nenhum modelo, threshold ou importance foi ajustado.

    **Próximas decisões:** confirmar antecedência dos sensores, horizonte e identidade
    de equipamento; auditar rótulos; revisar blocos/gap; estabelecer baseline e pipeline
    em TRAIN. AP é proposta para ordenação; Recall/Precision/F1 devem acompanhar custos
    e capacidade de atender alertas. Sem avaliação de generalização nesta etapa.
    """
    display(Markdown(findings))
    (EXPERIMENT / 'findings.md').write_text(findings.strip() + '\\n', encoding='utf-8')
    versions = {name: importlib.metadata.version(name) for name in
                ['pandas', 'numpy', 'matplotlib', 'seaborn', 'nbformat', 'nbclient', 'ipykernel']}
    manifest = {
        'run_id': RUN_ID, 'dataset': config['source'], 'dataset_sha256': dataset_hash,
        'config': config, 'python': platform.python_version(), 'versions': versions,
        'train_rows': len(train), 'test_rows': int((split['partition'] == 'TEST').sum()),
        'explored_partition': 'TRAIN', 'models_fitted': False,
        'split_membership': 'split_membership.csv', 'randomness': 'nenhuma; operações determinísticas',
        'code_sha256': {str(p.relative_to(ROOT)): sha256(p) for p in [
            ROOT / 'scripts/create_eda_notebook.py', ROOT / 'scripts/execute_eda.py',
            *sorted((ROOT / 'src/predictive_maintenance').glob('*.py')),
        ]},
    }
    (EXPERIMENT / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    print('Tabelas:', TABLES.relative_to(ROOT))
    print('Figuras:', FIGURES.relative_to(ROOT))
    print('Proveniência:', EXPERIMENT.relative_to(ROOT))
    ''')
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    destination = root / "notebooks/01_eda.ipynb"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError("Notebook já existe; preserve as edições antes de regenerar.")
    destination.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    for directory in [
        "data/raw",
        "data/interim",
        "data/processed",
        "models",
        "reports/figures",
        "reports/metrics",
        "reports/experiments",
    ]:
        path = root / directory
        path.mkdir(parents=True, exist_ok=True)
        (path / ".gitkeep").touch()


if __name__ == "__main__":
    main()

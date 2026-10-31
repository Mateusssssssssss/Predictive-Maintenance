# Predictive Maintenance

Exploração de dados operacionais para estudar classificação de falha em equipamento.
O projeto contém **EDA e notebooks separados para XGBoost e Random Forest**, com
GridSearchCV, estatísticas e gráficos de treino/validação. **TEST permanece reservado**,
sem predições ou avaliação nesta etapa.

Comece por [notebooks/01_eda.ipynb](notebooks/01_eda.ipynb).

- [XGBoost — treino e métricas](notebooks/metrics_xgboost.ipynb)
- [Random Forest — treino e métricas](notebooks/metrics_random_forest.ipynb)

## Problema e dados

`manutencao.csv` é o original fornecido, preservado na raiz. Seu schema é compatível
com a base sintética [AI4I 2020 / UCI](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset),
mas a identidade com a distribuição oficial não foi verificada. Uma linha representa
uma observação operacional; não necessariamente uma máquina independente. O alvo
é `Machine failure`, positivo `1`.

Faltam datas, horizonte e antecedência comprovada dos sensores. A EDA não comprova
previsão de falha futura ou aplicação industrial real. UDI e Product ID são excluídos
dos preditores. TWF/HDF/PWF/OSF/RNF são rótulos auxiliares de falha: usá-los como
entradas permitiria vazamento do desfecho.

## Ambiente e execução

Na raiz, com Python 3.11 ou superior, no PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe scripts/execute_eda.py
.\.venv\Scripts\python.exe -m jupyterlab notebooks/01_eda.ipynb
```

Selecione o kernel `.venv` ao abrir no editor. A execução automatizada usa o kernel
`python3` instalado pelo ipykernel no ambiente. Os caminhos funcionam a partir da
raiz ou da pasta `notebooks/`. `requirements-lock.txt` registra versões exatas da
entrega em Windows/Python 3.12: para reproduzir nesse ambiente, instale o arquivo antes
de `pip install -e . --no-deps`.

Cada execução preserva relatórios anteriores em pastas próprias de `reports/` e
atualiza as saídas do notebook. Não é necessário executar `create_eda_notebook.py`;
ele rastreia a criação inicial e recusa sobrescrever um notebook existente.

## Estrutura

```text
configs/data.json                    schema, features e separação inicial
data/{raw,interim,processed}/         espaços para evolução dos dados
docs/data_dictionary.csv             descrição, unidades e papéis das colunas
docs/experiment_contract.md          pressupostos e próximas decisões
notebooks/01_eda.ipynb                EDA executada para revisão
notebooks/metrics_xgboost.ipynb       busca e métricas do XGBoost
notebooks/metrics_random_forest.ipynb busca e métricas do Random Forest
src/predictive_maintenance/data.py    carregamento, validação e split
src/predictive_maintenance/eda.py     tabelas reutilizáveis
scripts/execute_eda.py                execução limpa e persistência
tests/                               contratos de dados e separação
reports/{figures,metrics,experiments}/ gráficos, tabelas e manifestos por execução
models/<run_id>/pipeline.joblib       pipelines locais de candidatos (fora do Git)
```

## Protocolo e análise

Os últimos 20% em UDI ficam reservados como TEST antes da exploração. A divisão
inicial evita misturar vizinhos de sinais sequenciais; não representa validação
temporal comprovada nem foi otimizada por resultados. Não há gap, pois horizonte
e duração da dependência ainda não foram definidos. Toda análise que orienta
decisões usa os primeiros 80%, identificados como TRAIN.

A EDA cobre dicionário, tipos, cardinalidade, ausências, duplicatas, plausibilidade,
extremos IQR, consistência entre alvo e modos, desbalanceamento, distribuições por
classe, taxas por tipo/faixa com denominadores, Pearson/Spearman, redundância,
interações e ordem dos registros. Nenhum registro foi removido automaticamente.
Correlação não equivale a causalidade ou importância preditiva.

O notebook termina com achados calculados e pendências para revisão. Manifestos
registram hash do CSV, configuração, versões, código e membros dos conjuntos.
[reports/validation.md](reports/validation.md) registra as verificações da entrega.

## Verificação

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check src tests scripts
.\.venv\Scripts\python.exe -m ruff format --check src tests scripts
```

## GridSearchCV: XGBoost e Random Forest

```powershell
.\.venv\Scripts\python.exe -u scripts/train_models.py --model all
.\.venv\Scripts\python.exe scripts/execute_model_notebooks.py
```

Use `--model xgboost` ou `--model random_forest` para treinar separadamente.
Os notebooks têm `RUN_TRAINING=False`: executar suas células lê os resultados
salvos e gera os gráficos, sem repetir a busca. Para iniciar uma nova busca pelo
notebook, altere a flag explicitamente. `RUN_ID` permite revisar um experimento
específico; `None` abre a execução mais recente daquele modelo. Não é preciso
rodar os geradores de notebook: eles preservam arquivos existentes.

Dentro dos 8.000 registros de TRAIN: os primeiros 6.400 formam FIT para busca e
os últimos 1.600 formam VALIDAÇÃO. A busca usa 3 folds de prefixos crescentes
apenas no FIT. A validação posterior não participa do GridSearchCV ou do refit.
Os dois modelos usam os mesmos registros e folds. O TRAIN inteiro já participou
da EDA: a validação é interna ao desenvolvimento, sem independência da exploração.

`configs/training.json` define as grades completas. Cada modelo pesquisa 32
combinações × 3 folds = 96 fits de busca, mais 1 refit, 4 ajustes de baseline e
3 diagnósticos de tamanho. O critério é maior AP média de CV; empate exato usa o
primeiro candidato. O campeão da grade não é um ótimo global. Os scores da CV
do vencedor têm otimismo de seleção; não há CV aninhada ou OOF reportado.

Random Forest pesquisa árvores, profundidade, tamanho mínimo de folha, fração de
features por divisão e pesos de classe. XGBoost pesquisa árvores, learning rate,
profundidade, min_child_weight e scale_pos_weight. A grade é inicial e compacta,
com limites explícitos. Critério Gini/bootstrap no RF e subsampling/regularização
no XGBoost ficam fixos, identificados no manifesto. Não se afirma que parâmetros
fixos foram pesquisados. Não há early stopping ou reamostragem.

Imputação mediana e categórica, seguida de one-hot, ficam dentro do pipeline de
cada fold. Árvores não recebem scaling. UDI, Product ID e modos de falha não entram
no modelo. Pesos podem modificar a distribuição dos scores; calibração não foi
demonstrada. A decisão usa `score >= 0.5` como referência fixa, sem tuning de limiar.

Cada notebook apresenta grade completa/top 10, parâmetros escolhidos, baseline,
média e desvio entre folds, ajuste versus validação, barras F1/Recall/Precision,
matriz de confusão, PR/AP, ROC, Brier/Log Loss, confiabilidade, sensibilidade de
limiar, associação versus importância por permutação, diagnóstico por tamanho e
erros por categoria/faixa. Os desvios entre folds e permutações não são intervalos
de confiança. O diagnóstico por prefixo mistura regimes e tamanho de amostra;
não é uma learning curve i.i.d. controlada.

Nas tabelas originais do GridSearchCV, `mean_test_*` e `split*_test_*` significam
**validação dos folds dentro de FIT**, não o TEST reservado. Nas métricas gerais,
`FIT_RESUBSTITUTION` e `VALIDATION_TRAIN_BLOCK` identificam os conjuntos explicitamente.
AP difere da área trapezoidal da PR. Precision indefinida aparece como NaN quando
não há alertas nos diagnósticos; o scorer CV usa zero_division=0.

Os pipelines de cada candidato ficam em `models/<run_id>/pipeline.joblib`, ignorados
no Git. Representam refit em **6.400 registros**, não em TRAIN completo e não um
vencedor final do projeto. As predições foram verificadas antes/depois da serialização.
Não carregar binários de origem desconhecida. Grids, folds, resultados, predições de
validação, versões, parâmetros e hashes ficam em `reports/` por execução.
Consulte [reports/training_validation.md](reports/training_validation.md).
As definições e convenções estão em [docs/metric_definitions.csv](docs/metric_definitions.csv).

Os notebooks de métricas apresentam rótulos em português, com termos de manutenção,
vírgula decimal e contagens com separador de milhar. F1, Recall, Precision e taxas
são exibidos em porcentagem; seus desvios, em pontos percentuais. AP e ROC-AUC ficam
em escala 0–1. A formatação usa `presentation.readable_table`, inclusive no resumo
de CV com colunas em múltiplos níveis; valores ausentes aparecem como “Não definido”.
Os CSVs numéricos e os pipelines salvos mantêm os valores originais.

## Limites e próxima decisão

Comparar AP, Recall, Precision, estabilidade e carga de alertas entre os candidatos.
Confirmar momento da previsão, horizonte, identificação de equipamento e custos
antes de uma aplicação real. UDI não é timestamp; não há gap ou garantia de
independência entre máquinas. Sensibilidade de threshold é diagnóstico, não seleção
operacional. Nenhum vencedor de produção, API, tracking ou avaliação TEST foi entregue.

Referências de implementação: [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html),
[TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html),
[permutation importance](https://scikit-learn.org/stable/modules/generated/sklearn.inspection.permutation_importance.html),
[XGBoost sklearn API](https://xgboost.readthedocs.io/en/release_3.3.0/python/sklearn_estimator.html).
O [AGENTS.md](AGENTS.md) adapta o guia fornecido ao escopo autorizado.

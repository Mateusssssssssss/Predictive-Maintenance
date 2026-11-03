# Predictive Maintenance

Exploração de dados operacionais para estudar classificação de falha em equipamento.
O projeto contém **EDA e notebooks separados para XGBoost e Random Forest**, com
GridSearchCV e uma rodada com Optuna, estatísticas e gráficos de treino/validação.
**A avaliação final no TEST foi autorizada pelo usuário** e tem notebook próprio,
com modelos e limiares congelados. Menções a TEST reservado nas rodadas abaixo
descrevem o estado histórico durante o desenvolvimento.

Comece por [notebooks/01_eda.ipynb](notebooks/01_eda.ipynb).

- [XGBoost — treino e métricas](notebooks/metrics_xgboost.ipynb)
- [Random Forest — treino e métricas](notebooks/metrics_random_forest.ipynb)
- [Avaliação final dos dois modelos — TEST](notebooks/metrics_test.ipynb)

Modelo serializado de referência: `models/best_model.joblib` (Random Forest da
run `train_random_forest_20261101T001858Z_680167a8`, com melhor AP e F1 na
validação entre as duas receitas finais). É uma cópia integral do pipeline já
treinado, com features, preprocessamento, calibração e limiar 0,5; não houve
novo treinamento. Proveniência e SHA-256: `models/best_model.json`.
Carregar com `joblib.load('models/best_model.joblib')` no ambiente do projeto
instalado (`pip install -e .`); usar as seis colunas originais dos preditores.
`predict_proba(X)[:, 1]` retorna probabilidades e `predict(X)` aplica o limiar salvo.

## Rodada com Optuna e avaliação final

Ferramentas utilizadas: pandas/NumPy para dados e estatísticas; matplotlib/seaborn
para figuras; scikit-learn para pipelines, CV ordenada, métricas, calibração sigmoid
(`CalibratedClassifierCV` com `FrozenEstimator`) e ajuste de limiar por F1
(`TunedThresholdClassifierCV`, modelo pré-ajustado); imbalanced-learn para SMOTENC;
XGBoost e RandomForestClassifier para classificação; Optuna/TPE para 24 tentativas
por modelo, três folds cada, semente 42 e sem pruning; joblib para persistência;
nbformat/nbclient para notebooks executados; pytest e Ruff para verificações.
Versões resolvidas estão em `requirements-lock.txt` e nos manifestos de treino.

O pipeline cria diferença de temperaturas (processo − ar), potência mecânica
(torque × rpm × 2π/60) e interação desgaste × torque. Imputação, StandardScaler,
SMOTENC e codificação respeitam os folds de treino. A padronização atende à distância
usada pelo SMOTENC. Type permanece categórico até a síntese. O espaço Optuna inclui
razão/vizinhos do SMOTENC, pesos positivos e parâmetros de árvores. Peso neutro e
SMOTENC com pesos concorrem; toda receita final contém SMOTENC.
AP no RF e AUPRC trapezoidal no XGBoost usam `predict_proba[:, 1]`.

Blocos por ordem UDI: ajuste/busca 1–4.800 (221 falhas), calibração 4.801–5.600
(27), escolha de limiar 5.601–6.400 (17), validação 6.401–8.000 (35), TEST
8.001–10.000. O modelo ajustado em 4.800 é congelado antes da calibração e novamente
antes do limiar; não há refit posterior. A quantidade de ajuste mudou em relação
à rodada anterior (6.400), portanto as CVs antigas e novas não são comparáveis.
A validação foi reutilizada no desenvolvimento; o TEST avalia a receita congelada.
As poucas falhas nos blocos de calibração/limiar aumentam a incerteza.
Derivadas sintetizadas podem não preservar identidades físicas exatas.

Na mesma validação, XGBoost passou de AP 0,5103/F1 0,5060 para AP 0,7624/F1
0,6909; RF passou de AP 0,7499/F1 0,6944 para AP 0,8697/F1 0,8525. Isso não isola
o efeito de cada ferramenta. No XGBoost, o limiar escolhido antes da validação
reduziu recall de 60,0% para 54,3%, mas falsos positivos de 27 para 1. A etapa
calibrada em 0,5 obteve F1 0,7077, superior ao limiar ajustado (0,6909) nessa
validação; isso fica registrado, sem trocar a política com base na validação.
Não há garantia de que aplicar uma ferramenta melhore todas as métricas.

O notebook `metrics_test.ipynb` avalia as duas runs explicitamente fixadas em
`configs/test_evaluation.json`, verifica hashes e composição da partição, aplica
os limiares originais e salva snapshot em `reports/test/`. Inclui baseline prior
calculado do ajuste, métricas completas, comparação validação/TEST, matrizes de
confusão, PR, ROC, calibração, distribuição de scores, relatório por classe/Type
e intervalos bootstrap de 95% (500 réplicas, hipótese IID por registro).
O TEST não é reamostrado pelo SMOTENC. Reexecuções leem o snapshot e verificam
integridade. Os manifestos históricos de treino não são reescritos: o manifesto
da avaliação registra `test_evaluated=true`. Após revelar TEST, novos ajustes
motivados por ele exigem outra confirmação independente.

Resultado final em 2.000 registros, com 39 falhas (1,95%):

| Modelo | AP | AUPRC | F1 | Recall | Precisão | FP | FN |
| --- | --- | --- | --- | --- | --- | --- | --- |
| XGBoost | 0,7313 | 0,7306 | 70,97% | 56,41% | 95,65% | 1 | 17 |
| Random Forest | 0,7701 | 0,7696 | 79,41% | 69,23% | 93,10% | 2 | 12 |

RF detectou 27/39 falhas e XGBoost 22/39. No RF, F1 caiu de 85,25% na validação
para 79,41% no TEST; no XGBoost passou de 69,09% para 70,97%, enquanto AP caiu
de 0,7624 para 0,7313. As métricas variam por partição e não melhoram todas juntas.
Baseline prior tem AP 0,0195; sua AUPRC trapezoidal de 0,50975 decorre da
interpolação com scores constantes, não de discriminação útil. Consulte os
intervalos e a proveniência em `reports/test/test_b4b4cee2c103/`.

Comandos utilizados nesta implementação (PowerShell, raiz do projeto):

```powershell
.\.venv\Scripts\python.exe -u scripts/train_advanced.py --model all
.\.venv\Scripts\python.exe scripts/create_advanced_notebooks.py
.\.venv\Scripts\python.exe scripts/create_test_notebook.py
.\.venv\Scripts\python.exe -u scripts/execute_model_notebooks.py --model test
.\.venv\Scripts\python.exe -u scripts/execute_model_notebooks.py --model all
.\.venv\Scripts\python.exe -m pytest -q --basetemp=.cache/pytest-final-retry
.\.venv\Scripts\python.exe -m ruff check src scripts tests
.\.venv\Scripts\python.exe -m ruff format --check src scripts tests
.\.venv\Scripts\python.exe -m pip check
```

Verificação: 23 testes passaram e `pip check` não encontrou dependências quebradas.
O CSV encontrado em `data/raw/` foi copiado de volta à raiz, mantendo ambas as
cópias e confirmando o mesmo SHA-256 utilizado no treino.
Referências: [calibração](https://scikit-learn.org/stable/modules/calibration.html),
[limiares](https://scikit-learn.org/stable/modules/classification_threshold.html),
[Optuna samplers](https://optuna.readthedocs.io/en/stable/reference/samplers/index.html).

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

`configs/training.json` define as grades completas. Cada modelo pesquisa 72
combinações × 3 folds = 216 fits de busca, mais 1 refit, 4 ajustes de baseline e
3 diagnósticos de tamanho. O critério é maior AUPRC trapezoidal média de CV para
XGBoost e maior AP média de CV para Random Forest; empate exato usa o
primeiro candidato, **entre candidatos com SMOTENC**, conforme o requisito atual.
Controles sem SMOTENC não são elegíveis para refit. O escolhido não é um ótimo global. Os scores da CV
do vencedor têm otimismo de seleção; não há CV aninhada ou OOF reportado.

Random Forest pesquisa profundidade, tamanho mínimo de folha, fração de features
por divisão, razão SMOTENC e pesos de classe. XGBoost pesquisa learning rate,
profundidade, min_child_weight, razão SMOTENC e scale_pos_weight. A grade é inicial e compacta,
com limites explícitos. Critério Gini/bootstrap no RF e subsampling/regularização
no XGBoost ficam fixos, identificados no manifesto. Não se afirma que parâmetros
fixos foram pesquisados. São fixas 300 árvores no RF e 150 no XGBoost nesta rodada,
para concentrar o orçamento no balanceamento. Não há early stopping.

Imputação mediana e categórica, seguida de one-hot, ficam dentro do pipeline de
cada fold. Nos dois modelos, sensores são padronizados para as distâncias de SMOTENC;
Type permanece categórico até a síntese, recebendo one-hot depois dela. Controles
usam o mesmo pré-processamento com sampler desligado. UDI, Product ID e modos de falha não entram
no modelo. Pesos podem modificar a distribuição dos scores; calibração não foi
demonstrada. A decisão usa `score >= 0.5` como referência fixa, sem tuning de limiar.

Cada notebook apresenta grade completa/top 10, parâmetros escolhidos, baseline,
média e desvio entre folds, ajuste versus validação, barras F1/Recall/Precision,
matriz de confusão, PR/AP/AUPRC trapezoidal, ROC, Brier/Log Loss, confiabilidade, sensibilidade de
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

## Histórico: primeira rodada com SMOTENC e AUPRC no XGBoost

SMOTENC é a variante para sensores numéricos e `Type` categórico; não é SMOTETomek
ou uma combinação com undersampling. A grade usa razões minoria/maioria de 0,25 e
0,5 após síntese (não proporção da minoria no total). `k_neighbors=3` é fixo e
compatível com as 45 falhas do menor treino da CV. São hipóteses iniciais de
balanceamento moderado, comparadas apenas na CV. `scale_pos_weight=1` fica fixo
para evitar somar automaticamente pesos altos à reamostragem. São 32 candidatos.

Somente o treino de cada fold recebe síntese. Validação mantém os registros e
prevalência originais. Contagens antes/depois e sintéticos criados estão em
`sampling_counts.csv`. Scaling, imputação e sampler pertencem ao pipeline completo
serializado. A inferência não reamostra dados.

AUPRC trapezoidal é `auc(recall, precision)`; AP continua registrada separadamente.
Ambas usam `predict_proba[:, 1]`, já utilizado na receita anterior. AUPRC é a métrica
do GridSearchCV, não o parâmetro interno `eval_metric=logloss` do XGBoost.
Não foi ajustado threshold. A mudança combina sampler, pesos e critério de seleção:
uma diferença não pode ser atribuída somente à síntese sem ablação controlada.

Depois de uma nova rodada com `scripts/train_models.py --model all`, gere o antes/depois:

```powershell
.\.venv\Scripts\python.exe scripts/compare_model_runs.py
.\.venv\Scripts\python.exe scripts/execute_model_notebooks.py
```

As últimas duas execuções por modelo são comparadas somente se hashes do dataset,
split e folds coincidirem. AUPRC da receita antiga é calculada dos scores de
validação já salvos; não altera os CSVs anteriores. Ambos os notebooks incluem
tabela e gráfico antes/depois. A validação de desenvolvimento está sendo reutilizada;
isso não equivale a confirmação independente. TEST continua reservado.

## Rodada atual: SMOTENC e pesos nos dois modelos

Cada busca é uma lista de quatro grades condicionais: sem balanceamento (8),
só pesos (16), SMOTENC (16) e SMOTENC + pesos (32). Os dois primeiros braços são
**controles, sem elegibilidade para refit**; os dois últimos concorrem ao pacote,
mantendo SMOTENC nos dois modelos. A seleção restrita é explicitada no manifesto,
que também registra o melhor índice irrestrito. Não ocultamos controles superiores.

No XGBoost, scale_pos_weight assume 1 na receita SMOTENC e 3/10 na combinação.
No Random Forest, a receita SMOTENC usa class_weight=None, e a combinação usa
{0: 1, 1: 3} ou {0: 1, 1: 10}. São pesos positivos fixos como hipóteses de busca;
não são estimativas de custos nem razão calculada sobre validação/TEST.
Não usamos balanced no conjunto sintético, que recalcularia os pesos após a síntese.
As razões de SMOTENC 0,25/0,5 e os três vizinhos continuam dentro dos folds.

As 8 configurações base de cada modelo são idênticas entre os quatro braços:
isso permite comparar balanceamento sem grades base diferentes na mesma rodada.
O espaço difere de rodadas anteriores, portanto um ganho antes/depois não pode
ser atribuído exclusivamente ao sampler ou peso. Uma combinação com pesos maiores
pode aumentar falsos alertas ou piorar a ordenação; a CV, não uma presunção, decide.

`strategy_cv_summary.csv` registra o melhor candidato por braço, métricas de CV,
elegibilidade e indicação do selecionado. Os notebooks incluem gráficos de área e
F1/Recall/Precision por estratégia, auditoria do sampler e comparação com a rodada
anterior. Somente o escolhido é reajustado em FIT e avaliado no bloco de validação.
A validação repetida continua sendo de desenvolvimento; TEST não é acessado.

Para atualizar a apresentação de notebooks existentes:

```powershell
.\.venv\Scripts\python.exe scripts/update_balance_notebooks.py
.\.venv\Scripts\python.exe scripts/compare_model_runs.py
.\.venv\Scripts\python.exe scripts/execute_model_notebooks.py
```

Snapshots anteriores ficam em reports/notebook_snapshots/before_balance_search/.
Treinamento e validação continuam pelos comandos anteriores; não há avaliação TEST.
Resultados desta rodada: [reports/balance_search_comparison.md](reports/balance_search_comparison.md).

Referências dos pesos: [XGBoost scale_pos_weight](https://xgboost.readthedocs.io/en/latest/python/python_api.html)
e [RandomForestClassifier class_weight](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html).

Referências: [SMOTENC para dados mistos](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTENC.html)
e [diferença entre AP e integração trapezoidal](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html).

Referências de implementação: [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html),
[TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html),
[permutation importance](https://scikit-learn.org/stable/modules/generated/sklearn.inspection.permutation_importance.html),
[XGBoost sklearn API](https://xgboost.readthedocs.io/en/release_3.3.0/python/sklearn_estimator.html).
O [AGENTS.md](AGENTS.md) adapta o guia fornecido ao escopo autorizado.


## Seleção de balanceamento sem SMOTENC obrigatório (alteração proposta)

As rodadas históricas e os valores de TEST acima **não foram recalculados**.
Em novas execuções, `configs/training.json` permite refit do candidato com
maior métrica de CV independentemente de utilizar SMOTENC. A busca avançada
com Optuna também compara quatro estratégias: sem balanceamento, somente
pesos, somente SMOTENC e SMOTENC com pesos. O sampler é aplicado somente
nos folds de treino quando selecionado; sem SMOTENC, o pipeline usa
`sampler="passthrough"` e registra zero amostras sintéticas.

Optuna reserva as primeiras quatro tentativas para cobrir as estratégias e
depois prossegue com a busca TPE, sob o mesmo orçamento total configurado.
Essa alteração **não implica ganho de desempenho** e não autoriza usar o TEST
já revelado para escolher nova receita. Novos resultados são exploratórios
até avaliação independente adicional.


## Diagnóstico dos falsos negativos — Random Forest (03/11/2026)

O modelo de referência com threshold 0,50 obteve, no bloco de validação de desenvolvimento, **Precision 100%**, **Recall 74,29%** e **F1 85,25%** (26 TP, 9 FN e 0 FP em 1.600 registros).

O diagnóstico dos **nove falsos negativos** mostrou que **7 apresentam o indicador auxiliar TWF** (falha associada ao desgaste da ferramenta), enquanto **nenhum dos 26 verdadeiros positivos** apresenta esse indicador. Os falsos negativos também tiveram mediana de apenas **1 exemplo com falha entre os 20 vizinhos mais próximos do FIT_SEARCH**, contra **7 para os verdadeiros positivos**, usando cinco sensores padronizados e o mesmo tipo de produto.

Isso sugere investigar limitações na identificação de certos mecanismos de falha, sobretudo TWF, e possível sobreposição local entre registros normais e de falha. **Não é uma demonstração causal nem uma nova validação independente.** TWF/HDF/PWF/OSF/RNF são **rótulos auxiliares usados somente na análise de erros**, nunca features do classificador.

- [Notebook reproduzível de análise de erros](notebooks/04_error_analysis.ipynb)
- [Relatório com metodologia, resultados e limitações](reports/random_forest_error_analysis_20261103.md)
- [Execução de referência e artefato das predições](https://github.com/Mateusssssssssss/Predictive-Maintenance/actions/runs/41679733671)

O notebook requer a extração do artefato de previsões indicado acima. Nenhuma alteração no modelo de referência ou consulta ao TEST foi feita para essa análise.


## Experimento direcionado a TWF: ablação de features (03/11/2026)

**Motivo desta decisão:** o diagnóstico em [`reports/random_forest_error_analysis_20261103.md`](reports/random_forest_error_analysis_20261103.md) identificou 9 falsos negativos no Random Forest de referência, sendo **7 associados à flag TWF**, enquanto nenhum dos 26 verdadeiros positivos dessa validação tinha TWF marcado. Nos 20 vizinhos de treino mais próximos, os FN tinham mediana de apenas 1 exemplo positivo, contra 7 para os TP. Isso **sugere** sobreposição entre classes e possível insuficiência das features atuais para identificar certas falhas por desgaste, mas **não prova** causalidade nem escassez como causa única.

**Decisões experimentais e justificativas:**
1. **Preservar o baseline** (Random Forest da run `train_random_forest_20261103T170044Z_3b8896f7`): fixamos `n_estimators=150`, `max_depth=6`, `min_samples_leaf=1`, `max_features=0.7`, `criterion=gini`, `max_samples=1.0`, semente 42, sem SMOTENC e sem pesos extras. Assim isolamos o efeito das novas variáveis, sem misturar alterações de hiperparâmetros ou de balanceamento.
2. **Testar hipóteses mensuráveis**: baseline com diferença de temperaturas, potência e desgaste×torque já existentes; candidato `wear_squared` adiciona desgaste² (hipótese de não linearidade); candidato `wear_context` adiciona desgaste×rpm e desgaste×diferença térmica (hipótese de interação operacional). Todas são transformações determinísticas por linha, sem aprender dos rótulos.
3. **Não usar rótulos auxiliares como features**: `TWF/HDF/PWF/OSF/RNF` descrevem mecanismos de falha e vazariam informação do alvo; elas aparecem **apenas como indicadores de auditoria posterior à predição**.
4. **Comparar por CV pareada e sem tocar VALIDATION/TEST**: três folds ordenados no prefixo `FIT_SEARCH` de 4.800 linhas; mesmas partições, hiperparâmetros e threshold 0,50 em todos os candidatos. O novo experimento não acessa os blocos de calibração, threshold, validação histórica nem o TEST. A ordem UDI não equivale a tempo comprovado.
5. **Avaliar AP, Recall, Precision, F1 e Recall de TWF**: AP avalia separação dos scores sem depender de threshold; métricas por classe capturam a compensação entre falsos negativos e falsos positivos. Recall de TWF é diagnóstico de mecanismo de falha, não objetivo de treinamento.
6. **Não promover automaticamente a nova variante**: os achados foram motivados pela validação já examinada; mesmo o CV em FIT_SEARCH é exploratório porque esse conjunto já participou do tuning histórico. Uma alegação de melhoria confirmada requer dados novos independentes.

**Execução reproduzível:** `python scripts/run_twf_ablation.py`. Código em [`src/predictive_maintenance/twf_experiment.py`](src/predictive_maintenance/twf_experiment.py); teste em `tests/test_twf_experiment.py`; relatórios, tabelas e gráficos ficam em `reports/twf_ablation/` (disponibilizados como artefatos do GitHub Actions, sem sobrescrever resultados históricos). A execução não atualiza `models/best_model.joblib` nem o threshold em produção.


### Resultado da ablação TWF — decisão após execução

Execução comprovada: [GitHub Actions — TWF CV](https://github.com/Mateusssssssssss/Predictive-Maintenance/actions/runs/41686277340), com testes aprovados e artefatos registrados em [`reports/twf_feature_ablation_20261103.md`](reports/twf_feature_ablation_20261103.md). Três candidatos foram comparados por predições OOF de CV somente em FIT_SEARCH (3.600 previsões, 188 falhas, sem calibração):

| Feature set | AP | Recall | Precision | F1 | TWF detectadas |
|---|---:|---:|---:|---:|---:|
| Baseline congelado | **0,5399** | 34,57% | 90,28% | 50,00% | 1/18 |
| + desgaste² | 0,5235 | 34,57% | 89,04% | 49,81% | 1/18 |
| + desgaste×rpm e desgaste×gap térmico | 0,5045 | 34,57% | 91,55% | 50,19% | 1/18 |

**Decisão fundamentada: não incorporar as novas features.** Nenhuma aumentou o Recall total nem o Recall específico de TWF. A variante de interações reduziu AP apesar de melhorar marginalmente Precision; a variante quadrática também reduziu AP. O modelo de referência e o threshold permanecem inalterados.

**Nova limitação observada:** os três folds continham 28, 25 e **135** falhas, respectivamente; no terceiro fold o Recall do baseline caiu para **17,78%** (versus 75% e 80% nos dois anteriores). Isso sinaliza forte heterogeneidade entre os blocos, mas não comprova drift temporal porque UDI não é uma variável temporal validada. Antes de novos hiperparâmetros, vale examinar rótulos, dinâmica do desgaste e validação com histórico real por equipamento. Resultados OOF aqui **não são diretamente comparáveis** aos 74,29% de Recall da validação histórica com calibração.

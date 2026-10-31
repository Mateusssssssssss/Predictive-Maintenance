# XGBoost e Random Forest — treino e validação

Execução local em 30/10/2026 (America/Fortaleza); IDs usam UTC.
Solicitação explícita: **não avaliar TEST nesta etapa**.

## Protocolo executado

- CSV original preservado; mesmo SHA256 da EDA.
- Desenvolvimento: primeiros 8.000 registros em UDI; TEST: últimos 2.000, reservado.
- FIT: primeiros 6.400 de desenvolvimento; VALIDAÇÃO: 1.600 seguintes, com 35 falhas.
- Busca: 3 folds ordenados em FIT, com prefixos de ajuste de 1.600/3.200/4.800
  e blocos posteriores de validação de 1.600. Sem shuffle ou gap.
- Mesmo hash de membros do split e folds para os dois candidatos.
- Seleção por maior média de Average Precision na CV; 32 configurações por modelo,
  96 fits de busca e 1 refit por modelo. Desempate: primeiro candidato da grade.
- Baseline Dummy prior, imputação e one-hot em pipeline; sem reamostragem ou scaling.
- Threshold de referência fixo: score >= 0,5. Sem tuning ou calibrador ajustado.
- Permutation importance e diagnósticos usam validação; não alteraram o pacote.
- Pipelines reajustados em FIT de 6.400; não houve refit nas 8.000 observações.
- Nenhuma predição, métrica ou figura do conjunto TEST.

## Resultados reais

| Modelo | AP média CV ± desvio amostral | AP validação | F1 validação | Recall validação | Precision validação | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| XGBoost | 0,4955 ± 0,0884 | 0,6796 | 0,6429 | 0,7714 | 0,5510 | 22 | 8 |
| Random Forest | 0,5076 ± 0,1215 | 0,7205 | 0,5660 | 0,4286 | 0,8333 | 3 | 20 |

O baseline AP na validação foi 0,021875 (prevalência), com Recall e F1 iguais a 0
e Precision indefinida por ausência de alertas. O scorer CV usa zero_division=0.

Random Forest ordenou melhor neste bloco segundo AP, mas no limiar de referência
perdeu mais falhas. XGBoost detectou 27 das 35 falhas, gerando 49 alertas; Random
Forest detectou 15, gerando 18 alertas. Custos e capacidade operacional não foram
informados. Não há vencedor de produção ou superioridade estatística demonstrada.

## Parâmetros escolhidos na grade

- **XGBoost:** n_estimators=150, max_depth=4, learning_rate=0.05,
  min_child_weight=1, scale_pos_weight=10. Subsample=0.8, colsample_bytree=0.9,
  reg_alpha=0 e reg_lambda=1 ficaram fixos. Busca: 35,2 s.
- **Random Forest:** n_estimators=300, max_depth=10, min_samples_leaf=1,
  max_features=sqrt, class_weight=None. Gini, bootstrap e min_samples_split=2
  ficaram fixos. Busca: 67,4 s.

Tempos são apenas da busca e refit; não incluem todos os diagnósticos e notebooks.

## Artefatos

- XGBoost: `train_xgboost_20261031T004336Z_06512d36`.
- Random Forest: `train_random_forest_20261031T004452Z_db4ffe4c`.
- Cada notebook: 25 células, 11 de código executadas, zero saídas de erro.
- Cada execução: 11 figuras PNG e 12 tabelas CSV, mais manifesto e membros dos conjuntos.
- Grids completos com parâmetros por candidato, scores de ajuste/validação por fold,
  dispersões e tempos persistidos em `grid_results.csv`.
- Pipeline local por execução em `models/`, fora do Git. Hash verificado e predições
  equivalentes antes/depois da serialização (tolerância absoluta 1e-12).
- Figuras de comparação, threshold, importância, matrizes e perfis inspecionadas visualmente.
- Dicionário de métricas: `docs/metric_definitions.csv`.

## Verificações executadas

- `python -m pytest -q --basetemp=.cache/pytest-training-all`: **15 passed**.
  São testes de implementação; não avaliação do conjunto TEST.
- `python -m ruff check src tests scripts`: **All checks passed**.
- `python -m ruff format --check src tests scripts`: **14 files already formatted**.
- `python -m pip check`: **No broken requirements found**.
- Schema dos dois notebooks validado com nbformat; outputs verificados sem erros.
- Hashes do split e CV idênticos entre candidatos; hashes dos pipelines conferidos.
- `git diff --check`: sem erros de whitespace; normalização LF/CRLF sinalizada pelo Git.

## Limitações

UDI é ordem, não data real; não há identificação confirmada de máquina ou horizonte.
Sem gap e sem CV aninhada. Scores do vencedor na CV têm otimismo de seleção.
A validação pertence ao TRAIN já explorado na EDA e tem somente 35 falhas.
Desvios entre folds e permutações não são intervalos de confiança. Curvas por
tamanho misturam regimes e quantidade de amostras. Scores não têm calibração comprovada.
Não houve uso de TEST para seleção ou avaliação.

O kernel Jupyter local exigiu execução fora do isolamento do Windows. Os avisos
de loop Proactor/ZMQ e transporte TCP local, descritos na validação da EDA,
continuaram presentes e não impediram a execução. Nenhum serviço foi publicado.

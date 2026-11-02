# Busca conjunta: SMOTENC e pesos nos dois modelos

Nesta rodada, os dois classificadores treinam SMOTENC e SMOTENC + pesos. Também
há controles sem balanceamento e só pesos, sem elegibilidade para refit. O requisito
de manter SMOTENC é explícito; o manifesto registra tanto o escolhido elegível
quanto o melhor índice irrestrito. Não se esconde eventual controle superior.

## Protocolo executado

- Mesmo CSV, FIT 6.400, validação posterior 1.600 e TEST 2.000 reservado.
- Três folds ordenados dentro de FIT; sem shuffle ou gap, UDI não é tempo comprovado.
- 72 candidatos por modelo: 8 sem balanceamento, 16 só pesos, 16 SMOTENC e 32 combinados.
- 48 candidatos elegíveis por modelo; seleção por AUPRC no XGBoost e AP no RF.
- 216 ajustes de busca por modelo, 432 no total, mais refits/baseline/diagnósticos.
- XGBoost: scale_pos_weight=1 na receita SMOTENC e 3/10 na combinação.
- RF: class_weight=None na receita SMOTENC e {0: 1, 1: 3}/{0: 1, 1: 10} na combinação.
- Razões minoria/maioria após síntese: 0,25/0,5. Três vizinhos, somente no treino.
- Sensores imputados e padronizados; Type categórico até a síntese, one-hot depois.
- Árvores fixas nesta rodada: 150 no XGBoost e 300 no RF. O espaço base é igual
  entre os quatro braços de cada modelo, com oito configurações base.
- Limiar fixo 0,5; scores positivos de predict_proba; nenhum tuning de threshold.

## XGBoost

| Melhor configuração de cada estratégia | AUPRC média CV | Elegível? |
|---|---:|---|
| Sem balanceamento | 0,4908 | Não — controle |
| Só pesos | 0,4926 | Não — controle |
| SMOTENC | **0,5353** | Sim — escolhido |
| SMOTENC + pesos | 0,5305 | Sim |

O escolhido também foi o máximo irrestrito (índice 26). Parâmetros: 150 árvores
fixas, profundidade 2, learning_rate=0,05, min_child_weight=5,
sampling_strategy=0,25, scale_pos_weight=1. O melhor combinado usou peso 3,
profundidade 4, min_child_weight=1 e a mesma taxa de aprendizado e razão de síntese.
Os pesos 3 e 10 foram efetivamente treinados, não apenas sugeridos.

O combinado teve Recall médio CV de 60,44%, contra 59,15% do SMOTENC sozinho,
mas menor AUPRC. A diferença de área é pequena e não demonstra superioridade
estatística com apenas três folds; a seleção usa o máximo pontual definido.

Na validação, o escolhido reproduziu a rodada anterior: AUPRC 0,4917, AP 0,5103,
F1 50,60%, Recall 60,00%, Precision 43,75%, FP 27 e FN 14. Não houve melhora.
Não se forçou um peso alto no pacote quando ele perdeu pelo critério de seleção.

## Random Forest

| Melhor configuração de cada estratégia | AP média CV | Elegível? |
|---|---:|---|
| Sem balanceamento | **0,5069** | Não — controle |
| Só pesos | 0,4925 | Não — controle |
| SMOTENC | **0,4632** | Sim — escolhido entre elegíveis |
| SMOTENC + pesos | 0,4336 | Sim |

O máximo irrestrito foi o controle sem balanceamento (índice 4). Por requisito de
manter SMOTENC nos dois modelos, o refit usa o melhor elegível, índice 30. Não se
afirma que esse refit venceu o controle na CV.

Escolhido: 300 árvores fixas, max_depth=None, max_features=1,0,
min_samples_leaf=4, sampling_strategy=0,25, class_weight=None. A melhor combinação
com pesos usou {0: 1, 1: 3}, max_depth=10, max_features=sqrt e folha mínima 4.
O peso 10 também foi pesquisado em ambos os braços ponderados.

## Métricas do bloco de validação, limiar 0,5

| Modelo / versão | AUPRC | AP | F1 | Recall | Precision | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| XGBoost anterior | 0,4917 | 0,5103 | 50,60% | 60,00% | 43,75% | 27 | 14 |
| XGBoost atual | 0,4917 | 0,5103 | 50,60% | 60,00% | 43,75% | 27 | 14 |
| RF anterior | 0,7180 | 0,7205 | 56,60% | 42,86% | 83,33% | 3 | 20 |
| RF atual | 0,7479 | 0,7499 | 69,44% | 71,43% | 67,57% | 12 | 10 |

RF: AP aumentou 0,0294, AUPRC 0,0300, F1 12,84 p.p. e Recall 28,57 p.p.;
Precision caiu 15,77 p.p. Houve 10 falhas perdidas a menos, com 9 falsos alertas
a mais. A validação tem apenas 35 falhas. Esse resultado não altera retrospectivamente
a seleção de CV nem transforma a validação de desenvolvimento em teste independente.

## Rastreabilidade e verificações

XGBoost: train_xgboost_20261031T235601Z_fa7abc82. As tabelas grid_results.csv,
strategy_cv_summary.csv e sampling_counts.csv identificam candidatos, scores,
elegibilidade, escolhido e contagens reais de síntese. Artefatos anteriores preservados.
RF: train_random_forest_20261101T000049Z_f81b0b13 (ID UTC; data local 31/10/2026).
Tempo de busca/refit: 118,0 s no XGBoost e 277,8 s no RF; não inclui todos os diagnósticos.
Os dois pipelines passaram pela checagem de equivalência após serialização.
Ambos foram carregados e conferidos com sampler SMOTENC efetivo, não passthrough.
Hashes de pipeline, split e folds verificados; sem avaliação TEST.
Cada notebook atualizado tem 31 células, 14 de código executadas, sem saídas de erro.
As figuras por estratégia e antes/depois foram inspecionadas visualmente.
Os 28 artefatos numéricos originais da primeira entrega permaneceram idênticos por SHA256.
Ruff check passou e a formatação conferiu 19 arquivos.

Comandos da rodada com .venv/Scripts/python.exe:

```text
-m pytest -q --basetemp=.cache/pytest-balance-grid  # 19 passed
-u scripts/train_models.py --model all
scripts/update_balance_notebooks.py
scripts/compare_model_runs.py
-u scripts/execute_model_notebooks.py
-m ruff check src tests scripts
-m ruff format --check src tests scripts
```

Não há avaliação TEST. Scores CV do escolhido têm otimismo de seleção. A validação
de desenvolvimento é reutilizada, não confirmação independente. Comparações com
rodadas anteriores têm espaços de busca distintos; diferenças não são efeitos
causais isolados de SMOTENC ou pesos. Scores com ponderação/síntese não têm calibração
comprovada. Nenhum vencedor de produção foi declarado.

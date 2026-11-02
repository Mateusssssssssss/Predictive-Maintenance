# Rodada SMOTENC / AUPRC — 31/10/2026

**A nova receita do XGBoost não melhorou a validação.** Random Forest reproduziu
os resultados anteriores. Nenhum TEST foi avaliado e nenhuma receita anterior foi
substituída ou declarada vencedora final.

## Protocolo

Mesmo CSV, FIT de 6.400, validação posterior de 1.600 (35 falhas), três folds
ordenados e limiar fixo 0,5. Hashes de dataset, split e folds conferidos entre as
versões. Os 2.000 registros TEST permanecem reservados.

XGBoost usa pipeline com imputação/StandardScaler numérico, imputação categórica,
SMOTENC, one-hot e classificador. O sampler só atua em fit. AUPRC trapezoidal é o
critério de GridSearchCV; AP continua separada. Todos os scores vêm de
predict_proba[:, 1], já utilizado antes. eval_metric interno permanece logloss.

Grade: árvores 150/300, profundidade 2/4, learning rate 0,05/0,15,
min_child_weight 1/5 e sampling_strategy 0,25/0,5. k_neighbors=3 é fixo,
compatível com as 45 falhas do menor fold. scale_pos_weight=1 evita combinar
automaticamente reamostragem com o peso 10 anterior. São 32 candidatos × 3 folds.
Random Forest mantém a mesma grade e seleção por AP, sem sampler.

## Validação antes/depois

| Modelo / versão | AUPRC trapezoidal | AP | F1 | Recall | Precision | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| XGBoost anterior | 0,6763 | 0,6796 | 64,29% | 77,14% | 55,10% | 22 | 8 |
| XGBoost SMOTENC | 0,4917 | 0,5103 | 50,60% | 60,00% | 43,75% | 27 | 14 |
| Random Forest anterior | 0,7180 | 0,7205 | 56,60% | 42,86% | 83,33% | 3 | 20 |
| Random Forest atual | 0,7180 | 0,7205 | 56,60% | 42,86% | 83,33% | 3 | 20 |

XGBoost: queda de AUPRC de 0,1845 e de AP de 0,1693; F1 caiu 13,68 p.p., Recall
17,14 p.p. e Precision 11,35 p.p. Foram 5 falsos alertas e 6 falhas perdidas a mais.
Não houve retuning após observar a queda.

A AP média de seleção de XGBoost subiu de 0,4955 para 0,5357, mas não se traduziu
em ganho no bloco de validação. A AUPRC média de seleção da nova receita foi 0,5353.
Os scores de CV dos vencedores têm otimismo de seleção; não comparar AP antiga
diretamente com AUPRC nova como se fossem a mesma métrica.

## Escolhido na grade e auditoria

XGBoost: 150 árvores, profundidade 2, learning rate 0,05, min_child_weight=5 e
sampling_strategy=0,25. O refit passou de 265 para 1.533 falhas, preservando 6.135
não falhas: 1.268 amostras sintéticas. A razão é minoria/maioria, não fração da minoria
no total. Validação não recebe síntese. Contagens por fold estão em sampling_counts.csv.

As contagens dos folds são auditorias reexecutadas somente do sampler escolhido,
sem novos fits de classificador. O sampler do refit registra as contagens reais.

## Artefatos e reprodução

- XGBoost anterior: train_xgboost_20261031T004336Z_06512d36.
- XGBoost atual: train_xgboost_20261031T234210Z_700d4c05.
- Random Forest anterior: train_random_forest_20261031T004452Z_db4ffe4c.
- Random Forest atual: train_random_forest_20261031T234425Z_5a1caa57.
- Comparação numérica em reports/comparisons/ e previous_comparison.csv nas novas execuções.
- Manifestos incluem seleção, sampler, versões, hashes e equivalência após serialização.
- Notebooks anteriores preservados como notebook_before_smotenc.ipynb nas execuções originais.
- Ambos os notebooks atualizados foram executados e validados com zero saídas de erro:
  XGBoost com 29 células (13 de código), Random Forest com 27 (12 de código).
- Os 28 artefatos numéricos originais conferidos por SHA256 permaneceram idênticos.
- Figuras PR/ROC e antes/depois inspecionadas visualmente. Cada nova execução tem
  12 figuras; XGBoost tem 14 tabelas e Random Forest 13.
- Uma falha transitória de gravação foi resolvida usando arquivo temporário validado
  e substituição atômica no executor dos notebooks; não exigiu repetir treinamento.

Comandos executados com .venv/Scripts/python.exe:

```text
-m pytest -q --basetemp=.cache/pytest-smotenc  # 17 passed
-u scripts/train_models.py --model all
scripts/compare_model_runs.py
-u scripts/execute_model_notebooks.py
-m pip check                               # sem dependências quebradas
-m ruff check src tests scripts
-m ruff format --check src tests scripts
```

## Limites

A validação é reutilizada do desenvolvimento, não confirmação independente.
Sampler, pesos e critério de seleção mudaram juntos; a queda não isola o efeito
de SMOTENC nem prova que nenhuma outra configuração de síntese possa funcionar.
UDI não comprova tempo real, faltam horizonte/grupos e não há gap. Nenhum limiar
operacional foi escolhido nesta rodada.

AUPRC trapezoidal pode ser alta para scores constantes por interpolação de poucos
pontos: o baseline atual tem AUPRC 0,5109 e AP 0,0219 na validação. Por isso mostramos
AP e AUPRC separadas, inclusive para o baseline. Não interpretar a área do baseline
como habilidade discriminativa.

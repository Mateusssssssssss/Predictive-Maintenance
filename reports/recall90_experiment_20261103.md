# Experimento: Recall >= 90% e Precision >= 80% (03/11/2026)

**Status:** rodada executada no GitHub Actions, somente em desenvolvimento. TEST reservado não foi consultado.
**Execução:** https://github.com/Mateusssssssssss/Predictive-Maintenance/actions/runs/41682064961
**Artefato:** https://github.com/Mateusssssssssss/Predictive-Maintenance/actions/runs/41682064961/artifacts/13253931018

## Protocolo
- 48 tentativas Optuna por modelo × 3 folds; busca nos 4.800 registros FIT_SEARCH.
- Calibração em 800 registros; threshold em outros 800; validação em 1.600 (35 falhas).
- Objetivo de threshold por prioridade: satisfazer simultaneamente Recall >= 0.90 e Precision >= 0.80; sem candidato elegível, escolher maior Precision entre os que alcançam o Recall mínimo; caso contrário maximizar Recall.
- O threshold é escolhido **apenas no bloco THRESHOLD**, não na validação.
- A validação já foi reutilizada durante o projeto; resultados exploratórios, não confirmação independente.

## Resultados na validação (threshold pré-selecionado)

| Modelo | Threshold | TP | FP | FN | Recall | Precision | F1 | AP | AUPRC | Brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Random Forest | 0.009717 | 31 | 63 | 4 | 88.57% | 32.98% | 48.06% | 0.867487 | 0.866909 | 0.005684 |
| XGBoost | 0.004748 | 34 | 230 | 1 | 97.14% | 12.88% | 22.74% | 0.800405 | 0.799025 | 0.009725 |

**Nenhum dos dois modelos cumpriu as duas metas na validação.**

## Seleção de threshold (bloco THRESHOLD, 17 falhas)

| Modelo | TP | FP | FN | Recall | Precision | Ambas as metas? |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Random Forest | 16 | 35 | 1 | 94.12% | 31.37% | Não |
| XGBoost | 16 | 131 | 1 | 94.12% | 10.88% | Não |

## Melhor compromisso alternativo na validação (análise exploratória, não usar para escolher novo threshold)
- Random Forest: com Precision >= 80%, Recall máximo 80% (threshold ~0.03383).
- XGBoost: com Precision >= 80%, Recall máximo 71.43% (threshold ~0.66692).
- Para Recall >= 90%: melhor Precision observada RF ~31.68% (Recall 91.43%); XGBoost ~24.43% (Recall 91.43%).

## Hiperparâmetros selecionados pela CV
- Random Forest: `balancing_strategy=none`, `n_estimators=500`, `max_depth=10`, `min_samples_leaf=1`, `min_samples_split=2`, `max_features=0.7`, `criterion=entropy`, `max_samples=0.7`. CV AP = 0.709089.
- XGBoost: `balancing_strategy=smotenc_weights`, `sampling_strategy=0.5`, `k_neighbors=3`, `positive_weight=5`, `n_estimators=600`, `max_depth=8`, `learning_rate=0.0639289`, `min_child_weight=3`, `subsample=1`, `colsample_bytree=1`, `reg_alpha=0`, `reg_lambda=10`. CV AUPRC = 0.668575.

## Limitações e recomendação
- Apenas 17 positivos para seleção de threshold e 35 na validação, tornando resultados voláteis.
- Métricas em um mesmo bloco de validação usado no histórico; não é uma comparação causal entre buscas.
- Baixar threshold altera o balanço entre falsos positivos e falsos negativos; não melhora necessariamente a ordenação do modelo (AP).
- **Não incorporar automaticamente o threshold exploratório nem reabrir o TEST antigo.** Avaliar custos de falhas vs alertas, obtenção de novos dados independentes e validação robusta, antes de qualquer escolha operacional.

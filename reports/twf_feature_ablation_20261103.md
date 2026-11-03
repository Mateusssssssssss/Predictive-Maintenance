# Resultados — ablação exploratória TWF (03/11/2026)

**Status:** executado, 3 hipóteses avaliadas, sem melhoria observada na detecção de TWF. O modelo de referência foi mantido.

**Execução:** https://github.com/Mateusssssssssss/Predictive-Maintenance/actions/runs/41686277340

**Artefato:** https://github.com/Mateusssssssssss/Predictive-Maintenance/actions/runs/41686277340/artifacts/13254444084

## Pergunta científica

Os 9 falsos negativos observados na validação histórica incluíam 7 com indicador TWF. Investigamos a hipótese de que **interações não lineares de desgaste com rotação e temperatura** poderiam melhorar a detecção de falhas em CV, sem adicionar informações do desfecho ao modelo.

## Decisões e por quê

1. **Congelar os parâmetros do Random Forest** da run de referência `train_random_forest_20261103T170044Z_3b8896f7` e manter threshold 0,50, sem SMOTENC/pesos extras: isola o impacto das features testadas; não mistura tuning novo com ablação.
2. **Testar baseline, desgaste², desgaste×rpm e desgaste×diferença de temperatura**: a análise anterior indicou dificuldades com TWF e desgaste; as novas variáveis são construídas apenas com sensores operacionais medidos na linha, respeitando disponibilidade em inferência.
3. **Usar somente os 4.800 registros FIT_SEARCH**: os conjuntos CALIBRATION, THRESHOLD, VALIDATION histórico e TEST não foram usados. Três folds ordenados por UDI, compartilhados por todos os candidatos; UDI não é tempo comprovado.
4. **Usar TWF só na avaliação dos erros**: o indicador é uma coluna ligada ao desfecho, logo inseri-lo como preditor caracterizaria vazamento de dados.
5. **Medição por AP, AUPRC, F1, Recall, Precision e Recall TWF**: diferencia capacidade de ordenação (independente do limiar) de classificação binária e avaliação específica do mecanismo que motivou a hipótese.
6. **Não promover uma feature sem evidência**: o experimento é exploratório e requer dados novos independentes para confirmar qualquer ganho.

## Resultados agregados OOF da CV (3.600 amostras, 188 falhas)

| Versão | TP | FP | FN | Recall | Precision | F1 | AP | AUPRC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 65 | 7 | 123 | 34,57% | 90,28% | 50,00% | **0,5399** | **0,5394** |
| Desgaste² | 65 | 8 | 123 | 34,57% | 89,04% | 49,81% | 0,5235 | 0,5224 |
| Desgaste×rpm / desgaste×gap térmico | 65 | 6 | 123 | 34,57% | 91,55% | 50,19% | 0,5045 | 0,5036 |

**Resultado TWF:** das 18 falhas com `TWF=1` nos folds de avaliação, o baseline detectou 1/18, desgaste² detectou 1/18 e as interações detectaram 1/18. **Não houve aumento do Recall TWF**.

## Diagnóstico adicional: mudanças no perfil dos folds

O número de falhas do bloco de validação variou muito entre os três folds (28, 25 e **135**), e a prevalência passou de ~2% para 11,25% no terceiro fold. O Recall do baseline, por fold, foi 75%, 80% e **17,78%**.

Isso revela forte variação entre blocos ordenados. Não representa necessariamente drift temporal, já que UDI **não é uma data validada**; também não prova causalidade. É uma razão para priorizar testes de estabilidade e um desenho de validação por período/equipamento real quando os dados estiverem disponíveis.

## Interpretação e decisão

- Acrescentar desgaste² prejudicou ligeiramente AP e Precision.
- Acrescentar desgaste×rpm/gap reduziu AP e aumentou marginalmente Precision (menos um falso positivo), mas não alterou Recall total nem Recall TWF.
- **Manter Random Forest anterior, sem atualização de features, hiperparâmetros ou threshold.**
- Próxima hipótese: examinar limites de previsão antecipada de TWF a partir desses sensores estáticos, definição do alvo e dados de histórico por equipamento. Em dados novos, avaliar features temporais legitimamente disponíveis antes da falha.
- Essas métricas são **OOF de FIT_SEARCH sem calibração**, não devem ser comparadas diretamente com a validação histórica calibrada (Recall 74,29%, Precision 100%).
- Como as hipóteses foram motivadas por uma validação já observada, esta CV é exploratória e não confirma generalização.

Arquivos reproduzíveis no artefato: `fold_metrics.csv`, `cv_summary.csv`, `oof_summary.csv`, `cv_predictions.csv`, `cv_comparison.png`, `twf_by_fold.png`, `manifest.json`.

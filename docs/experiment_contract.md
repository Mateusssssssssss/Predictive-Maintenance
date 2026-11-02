# Contrato inicial

O guia externo orientou organização e metodologia; seu checklist não obriga a
implementar toda a infraestrutura de produção. O escopo atual contém EDA, utilitários,
schema, testes e proveniência. O escopo foi ampliado pelo usuário para XGBoost e
Random Forest com GridSearchCV e notebooks separados. Não há API, MLflow,
resampling ou avaliação do conjunto TEST nesta etapa.

O CSV tem schema compatível com AI4I 2020. A documentação oficial descreve dados
sintéticos e sinais sequenciais, justificando evitar embaralhamento nesta etapa.
A equivalência com a distribuição oficial não foi verificada.
Fonte: https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset

O bloco final em UDI é reservado antes da EDA. A fração 80/20 não foi otimizada:
deixa 8.000 observações para desenvolvimento e 2.000 para teste no CSV atual.
UDI não demonstra tempo real; faltam timestamp, horizonte e identificação de
equipamento. Não há gap: revisar após definir horizonte e dependência dos sinais.
Não há garantia de independência temporal ou entre máquinas. Os rótulos do TEST
passam somente por validação de schema, sem exploração de suas distribuições.

Antes do treino, confirmar antecedência dos sensores, unidade, população e custo
dos erros; auditar divergências dos rótulos. Estudar validação em blocos dentro de
TRAIN, baseline Dummy e modelos simples com transformações em pipeline. AP é proposta
para ordenação; Recall/Precision/F1 e capacidade de alertas precisam orientar a decisão.
Threshold e transformações aprendidas deverão usar apenas TRAIN e seus folds.

## Protocolo de treino autorizado

Por solicitação explícita, executar somente treino e validação, sem TEST.
Dos 8.000 registros explorados na EDA, usar prefixo FIT de 6.400 para três folds
crescentes com TimeSeriesSplit e bloco posterior de 1.600 para validação dos candidatos.
Os três folds têm 1.600 registros de validação e prefixos FIT de 1.600, 3.200 e 4.800.
Não há gap: dependência e horizonte reais continuam desconhecidos. Este protocolo
estuda ordem de registros e não demonstra previsão temporal futura.

Selecionar parâmetros pela média de AP na CV. Random Forest e XGBoost têm 32
combinações por modelo em configs/training.json. Refit somente nos 6.400 registros;
nenhum ajuste aprende com o bloco posterior de validação. Baseline Dummy usa os
mesmos folds. Transformações são ajustadas dentro do pipeline/fold.

Limiar fixo de referência 0,5 com score >= limiar, sem tuning. Diagnósticos de
threshold e permutation importance usam validação e não mudam features ou pacote.
Associações são calculadas em FIT. Não há seleção automática, SMOTE ou calibrador.

CV do vencedor está sujeita a otimismo de seleção. A validação pertence ao TRAIN
já explorado na EDA e pode orientar comparação futura; não é confirmação independente.
TEST permanece sem métricas ou predições. Nenhum candidato é declarado vencedor
de produção ou reajustado nas 8.000 observações nesta fase.

## Evolução solicitada em 31/10/2026

XGBoost passa a usar SMOTENC dentro de cada treino, com sensores imputados e
padronizados para distâncias e Type preservado antes do one-hot. A busca escolhe
entre razões minoria/maioria de 0,25 e 0,5, com três vizinhos fixos (45 falhas no
menor fold). scale_pos_weight=1 evita reponderação adicional nesta receita.
AUPRC trapezoidal passa a ser o refit do GridSearchCV, calculada com predict_proba;
AP continua como métrica distinta. O eval_metric interno do XGBoost continua logloss.
Random Forest mantém seleção por AP e sem sampler, para referência.

Os dados, folds e limiar 0,5 são preservados. A avaliação antes/depois usa scores
salvos e validação reutilizada, sem confirmação independente e sem acesso a TEST.
As mudanças de sampler, pesos e métrica ocorrem juntas; não isolam causalmente o
efeito do SMOTENC. Mesmo se piorar, preservar e reportar o resultado sem retunar
com base nessa validação.

## Ampliação: pesos e SMOTENC nos dois classificadores

Agora ambos os modelos treinam receitas SMOTENC e SMOTENC + pesos em uma grade
condicional. O refit exige SMOTENC por requisito do usuário. Só pesos e sem
balanceamento ficam como controles; a melhor posição irrestrita é registrada,
mesmo quando divergir do escolhido elegível. Não confundir essa restrição com
superioridade sobre todos os candidatos ou com avaliação final.

XGBoost pesquisa scale_pos_weight 3/10 na combinação; Random Forest pesquisa
class_weight {0: 1, 1: 3} / {0: 1, 1: 10}. SMOTENC sozinho usa peso neutro.
São multiplicadores positivos propostos para busca, não custos estimados ou
parâmetros ótimos prévios. Razões de síntese 0,25/0,5 e três vizinhos, sempre no
treino de cada fold. Não há balanceamento da validação.

Cada modelo tem 8 configurações base e 9 combinações de sampler/peso: 72
candidatos, 216 ajustes de busca e 1 refit. Para concentrar orçamento, são fixas
150 árvores no XGBoost e 300 no RF. Manter a mesma grade base nos quatro braços.
O espaço mudou em relação à rodada anterior; diferenças antes/depois não isolam
um único mecanismo. Seleção: AUPRC no XGBoost, AP no RF, apenas em CV.
Limiar 0,5, FIT 6.400, validação 1.600 e TEST 2.000 reservado caracterizam essa rodada histórica.

## Extensão Optuna e avaliação final autorizada

O usuário autorizou engenharia de features, Optuna, calibração e ajuste de limiar.
Agora FIT_SEARCH=4.800, CALIBRATION=800, THRESHOLD=800 e VAL=1.600,
em blocos ordenados, com três folds crescentes exclusivamente em FIT_SEARCH.
Optuna usa 24 tentativas por modelo, TPE/seed42, sem pruning; toda tentativa usa
SMOTENC e pesos neutros ou positivos. Critérios AP/RF e AUPRC/XGBoost permanecem.
Calibração sigmoid usa modelo congelado; F1 escolhe o limiar no bloco separado,
também com modelo congelado. Nenhum refit ocorre depois dessas decisões.

A pedido posterior explícito, TEST está autorizado para avaliação final dos dois
artefatos fixados em configs/test_evaluation.json. Sem ajuste de parâmetros,
features, calibração ou limiares no TEST. Avaliação e hashes ficam em reports/test/;
manifestos anteriores preservam o estado histórico. O TEST revelado não deverá
servir como nova validação seguida de alegação de confirmação independente.

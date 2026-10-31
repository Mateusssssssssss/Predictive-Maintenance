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

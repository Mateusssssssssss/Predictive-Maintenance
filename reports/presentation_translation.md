# Apresentação das métricas em português

Aplicação do exemplo fornecido pelo usuário, adaptando fraude/transações para
falhas/observações de manutenção. Alteração apenas de apresentação, sem treinamento
ou avaliação de TEST.

- Cabeçalhos, nomes dos conjuntos e receitas em português nos dois notebooks.
- F1, Recall, Precision, especificidade, prevalência e taxas em porcentagem.
- AP e ROC-AUC mantidos como scores; desvios de taxas em pontos percentuais (p.p.).
- Vírgula decimal, milhares nas contagens e “Não definido” para métricas ausentes.
- Resumo CV com colunas MultiIndex, grade e parâmetros com descrições legíveis.
- Gráficos com nomes dos modelos, sensores e limiar traduzidos; barras e eixos de
  métricas percentuais usam a escala correspondente.
- Função reutilizável: `src/predictive_maintenance/presentation.py`.

Verificações: formatação HTML simples e MultiIndex conferida; DataFrames numéricos
preservados; comparação SHA256 dos 28 artefatos existentes (24 CSVs, 2 pipelines e
2 manifestos) sem diferenças. Figuras de comparação inspecionadas visualmente.
Notebooks reexecutados somente para apresentar artefatos existentes, com
`RUN_TRAINING=False`. Verificações Ruff passaram (16 arquivos formatados).

Os manifestos de treinamento continuam registrando o código usado na execução
original. A evolução posterior da apresentação não altera o experimento treinado.

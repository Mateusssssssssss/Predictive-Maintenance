# Predictive Maintenance — orientações do projeto

Adaptado do guia fornecido pelo usuário (30/10/2026). O escopo atual é estrutura
e EDA, ampliado para Random Forest e XGBoost com GridSearchCV e notebooks separados.
API e tracking continuam fora do escopo.

Escopo ampliado pelo usuário: engenharia de features, Optuna, calibração e limiar
em blocos disjuntos de treino; avaliação TEST agora explicitamente autorizada.
Fixar runs e limiares antes de ler TEST; não ajustar decisões com base nele.
Registrar a avaliação final separadamente, preservando manifestos de treino.

- Preservar `manutencao.csv`; o dado original fica na raiz por compatibilidade.
- Código reutilizável em `src/predictive_maintenance`; notebooks em `notebooks`.
- Usar `pyproject.toml` como fonte de dependências e caminhos relativos à raiz.
- Não inventar origem confirmada, resultados, execução ou desempenho.
- Separar desenvolvimento e TEST antes de análises que orientem decisões.
- Não explorar o TEST para selecionar features, modelos ou limiares.
- Excluir UDI, Product ID e modos TWF/HDF/PWF/OSF/RNF dos preditores.
- Machine failure=1 é a classe positiva. Sem timestamp/horizonte, não alegar
  antecipação de falha futura nem validação temporal comprovada.
- Salvar notebook executado, figuras, tabelas e proveniência em reports/.
- Não remover outliers automaticamente nem tratar associação como causalidade.
- Quando houver modelos, incluir notebook de métricas que leia artefatos, baseline,
  validação apropriada e avaliação final isolada. Reamostragem e transformações
  aprendidas devem ocorrer dentro de cada fold de treino.
- Não adicionar serviços, API, modelos vazios ou infraestrutura sem necessidade.
- Verificar mudanças com testes relevantes e registrar os comandos reais no README.
- Ambos os modelos comparam SMOTENC e SMOTENC + pesos na CV. Os candidatos finais
  exigem SMOTENC; só pesos e sem balanceamento são controles, sem elegibilidade
  para refit. Type permanece categórico até a síntese, sempre dentro dos folds.
- AUPRC trapezoidal é o critério do XGBoost e AP o do Random Forest. Ambas usam
  scores de predict_proba. Manter métricas e receitas anteriores identificadas.

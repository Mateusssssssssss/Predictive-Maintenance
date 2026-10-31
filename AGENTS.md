# Predictive Maintenance — orientações do projeto

Adaptado do guia fornecido pelo usuário (30/10/2026). O escopo atual é estrutura
e EDA, ampliado para Random Forest e XGBoost com GridSearchCV e notebooks separados.
API e tracking continuam fora do escopo.

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

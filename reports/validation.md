# Verificação da entrega inicial

Data local: 30/10/2026 (America/Fortaleza). Os identificadores de execução usam UTC.

## Execução e artefatos

- Execução final: `20261031T002852Z_aa8e0ca3`.
- Comando: `.venv/Scripts/python.exe -u scripts/execute_eda.py`.
- Notebook validado com nbformat: 23 células, 11 de código executadas, zero saídas de erro.
- Saídas finais: 9 PNGs e 13 tabelas CSV, além de manifesto, membros do split e achados.
- A primeira execução foi preservada em `20261031T002733Z_0175f1a6`.
- Figuras de distribuição, taxas por faixa e associações inspecionadas visualmente;
  rótulos das faixas ajustados para uma casa decimal, sem alterar os intervalos calculados.
- CSV original preservado; SHA256:
  `dc6630cd9b1f0f853922fad78a1b6436570d3f1ec863f1dd5c4340ac56bc8a8e`.

## Evidências em TRAIN

- CSV: 10.000 linhas e 14 colunas; TRAIN: 8.000; TEST reservado: 2.000.
- TRAIN: 300 falhas (3,75%), nenhuma célula ausente e nenhuma entrada preditora duplicada.
- 25 divergências entre alvo e união dos modos de falha; nenhuma correção automática.
- 410 sinalizações IQR somadas entre as cinco variáveis, sem remoção de registros.
- Nenhum modelo ajustado, nenhuma métrica de desempenho e nenhuma distribuição de TEST explorada.

## Checks

- `.venv/Scripts/python.exe -m pytest -q`: **11 passed**.
- `.venv/Scripts/python.exe -m ruff check src tests scripts`: **All checks passed**.
- `.venv/Scripts/python.exe -m ruff format --check src tests scripts`: **7 files already formatted**.
- `.venv/Scripts/python.exe -m pip check`: **No broken requirements found**.
- `.venv/Scripts/python.exe -m jupyterlab --version`: **4.6.5**.
- `git diff --check`: sem erros de whitespace; aviso de normalização LF/CRLF do Git no README.

## Ambiente

Windows, Python 3.12.14, ambiente `.venv`; versões exatas em `requirements-lock.txt`
e versões analíticas no manifesto. A instalação, os testes com temporários e a
execução do kernel exigiram execução fora do isolamento do Windows. O notebook
inicia somente um kernel local, sem API de aplicação ou publicação.

Jupyter emitiu avisos de compatibilidade do loop Proactor com ZMQ (usa uma thread
seletora adicional) e de transporte TCP sem criptografia do kernel local. Não
impediram a execução; não foram ocultados. A execução não demonstra configuração
segura para kernel remoto ou serviço exposto à rede.

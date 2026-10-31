# Dados

`manutencao.csv` é o original fornecido, mantido na raiz sem alterações.
`configs/data.json` aponta para ele; não há download automático nem cópia nesta fase.

- `raw/`: reservado para fontes originais imutáveis.
- `interim/`: reservado para dados intermediários.
- `processed/`: reservado para dados preparados.

Novos dados nessas pastas são ignorados no Git. O CSV original, cuja origem provável
é sintética, fica disponível para versionamento explícito. Cada execução registra
seu SHA256 e os membros de TRAIN/TEST em `reports/experiments/`.

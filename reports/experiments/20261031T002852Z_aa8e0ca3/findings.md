- TRAIN contém **8,000 observações** e **300 falhas (3.75%)**.
- **0 células ausentes** e **0 entradas duplicadas** além da primeira ocorrência.
- **25 divergências de rótulo** entre alvo e união dos modos; sem correções automáticas.
- **410 sinalizações IQR** somadas entre variáveis; nenhum registro removido.
- Maior associação Spearman em módulo: **Torque [Nm]**; investigar interações antes de seleção.
- TEST permanece reservado. Nenhum modelo, threshold ou importance foi ajustado.

**Próximas decisões:** confirmar antecedência dos sensores, horizonte e identidade
de equipamento; auditar rótulos; revisar blocos/gap; estabelecer baseline e pipeline
em TRAIN. AP é proposta para ordenação; Recall/Precision/F1 devem acompanhar custos
e capacidade de atender alertas. Sem avaliação de generalização nesta etapa.

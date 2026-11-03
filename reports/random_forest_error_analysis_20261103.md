# Diagnóstico de erros — Random Forest de referência (03/11/2026)

## Origem e escopo

- Modelo: `train_random_forest_20261103T170044Z_3b8896f7`; threshold de 0,50.
- Predições: arquivo `metrics/<run_id>/validation_predictions.csv` do artefato da [execução GitHub Actions #41679733671](https://github.com/Mateusssssssssss/Predictive-Maintenance/actions/runs/41679733671).
- Dados originais: `manutencao.csv`, versão do repositório; as colunas auxiliares de mecanismo de falha são usadas **somente para diagnóstico**, não como preditoras.
- Avaliação: bloco VALIDATION de desenvolvimento (UDI 6401–8000), com 1.600 amostras e 35 falhas; **nenhum uso do TEST**.
- Limitação: esse conjunto já foi observado no desenvolvimento e não é uma confirmação independente.

## Resultados de referência

| TN | FP | FN | TP | Recall | Precision | F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 1565 | 0 | 9 | 26 | 74,29% | 100% | 85,25% |

Scores dos FN: 0,0071 a 0,0304 (todos muito abaixo do threshold 0,50).
Cinco FN são do tipo L, três do tipo M e um do tipo H.

## Descoberta 1: mecanismo de falha

As flags auxiliares do dataset `TWF`, `HDF`, `PWF`, `OSF` e `RNF` foram cruzadas **somente depois** de selecionar os erros pela previsão. Flags podem se sobrepor; sua soma não representa necessariamente eventos exclusivos.

| Indicador auxiliar | 9 FN | 26 TP |
| --- | ---: | ---: |
| TWF (desgaste da ferramenta) | **7** | **0** |
| HDF (dissipação de calor) | 0 | 0 |
| PWF (potência) | 0 | 16 |
| OSF (sobrecarga) | 1 | 11 |
| RNF (falha aleatória) | 0 | 0 |
| Sem flag auxiliar marcada | 1 | 0 |

Interpretação: a maioria dos casos não reconhecidos está associada ao indicador TWF. Isso **não prova causalidade**, mas sugere investigar se o modelo está aprendendo melhor PWF/OSF do que TWF, ou se os sensores disponíveis são pouco informativos para esse modo.

**Atenção ao leakage:** nunca adicionar TWF/HDF/PWF/OSF/RNF às features. São rótulos auxiliares relacionados ao desfecho, não medições que estarão disponíveis para previsão.

## Descoberta 2: densidade de exemplos semelhantes no FIT_SEARCH

Método exploratório:
1. usar apenas as 4.800 linhas FIT_SEARCH (UDI <= 4800), com 221 falhas;
2. padronizar cinco sensores numéricos com média e desvio calculados somente no FIT_SEARCH;
3. buscar os 20 vizinhos mais próximos para cada validação, restringindo ao mesmo `Type`;
4. contar quantos vizinhos possuem `Machine failure = 1`. Nenhum modelo é ajustado nesses rótulos de validação.

| Grupo | Quantidade de registros | Mediana de falhas nos 20 vizinhos |
| --- | ---: | ---: |
| FN | 9 | **1** |
| TP | 26 | **7** |

Falhas entre os 20 vizinhos para FN (por UDI): 6420→1; 6479→0; 6760→1; 6925→4; 7088→1; 7427→1; 7511→2; 7688→2; 7850→2.

Mesmo na vizinhança padronizada, muitos FN são cercados por exemplos rotulados como normais. Isso apoia a **hipótese de sobreposição entre classes, baixa representatividade local de falhas ou informação sensorial insuficiente**. Não é evidência isolada de que SMOTENC, mais árvores ou redução do threshold resolverão o problema; a distância euclidiana é apenas um proxy e pode não refletir a geometria aprendida pela floresta.

## Descoberta 3: sinais operacionais

Medianas descritivas na validação (FN versus TP):
- Torque: 38,9 versus 60,8 Nm.
- Rotação: 1.524 versus 1.349 rpm.
- Desgaste: 214 versus 146 minutos.

A amostra é pequena; não generalizar essas medianas nem interpretar como relação causal.

## Próximos passos sugeridos

1. Investigar TWF especificamente: observar distribuição de desgaste no treinamento por tipo, verificar o papel de potência/carga e sobreposição com exemplos normais.
2. Elaborar e comparar features operacionalmente disponíveis relacionadas à história de desgaste e carga, em ablação dentro de CV do bloco FIT_SEARCH, evitando leakage.
3. Medir Recall e Precision com intervalos de incerteza e por mecanismo em **nova coorte independente**.
4. Não usar o mesmo bloco VALIDATION como base de escolha e posterior confirmação; não reutilizar o TEST previamente revelado.

**Reprodução:** notebook `notebooks/04_error_analysis.ipynb`, usando o artefato original das previsões. Nenhuma alteração de modelo/threshold foi realizada aqui.

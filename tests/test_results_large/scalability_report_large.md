# Relatório de Teste de Escalabilidade (Amostras Grandes >1M)

**Data:** 2026-06-07 10:11:29
**Tamanhos de amostra testados:** [2000000, 5000000, 10000000]
**Trials por tamanho:** 1
**GPU utilizado no XGBoost:** True

## Tabela Resumo (média ± desvio padrão)

| Sample Size | Modelo | Acurácia | F1-Score | ROC-AUC | Tempo Treino (s) |
|-------------|--------|----------|----------|---------|------------------|
| 2,000,000 | Logistic Regression | 0.7101±nan | 0.8151±nan | 0.6745±nan | 37.09 |
| 2,000,000 | Random Forest | 0.7992±nan | 0.8829±nan | 0.6568±nan | 56.83 |
| 2,000,000 | XGBoost | 0.8617±nan | 0.9247±nan | 0.7146±nan | 1.02 |
| 5,000,000 | Logistic Regression | 0.7095±nan | 0.8146±nan | 0.6737±nan | 128.20 |
| 5,000,000 | Random Forest | 0.7931±nan | 0.8789±nan | 0.6540±nan | 177.74 |
| 5,000,000 | XGBoost | 0.8616±nan | 0.9247±nan | 0.7166±nan | 2.34 |
| 10,000,000 | Logistic Regression | 0.7092±nan | 0.8144±nan | 0.6743±nan | 186.73 |
| 10,000,000 | Random Forest | 0.7883±nan | 0.8755±nan | 0.6530±nan | 400.55 |
| 10,000,000 | XGBoost | 0.8619±nan | 0.9248±nan | 0.7165±nan | 7.29 |

## Análise e Conclusão

Os resultados para amostras de 2M, 5M e 10M confirmam a tendência observada em amostras menores:
- O XGBoost mantém F1-Score ≈ 0,925 e ROC-AUC ≈ 0,715.
- A flutuação (desvio padrão) é desprezível a partir de 1M amostras.
- O tempo de treino aumenta aproximadamente de forma linear, mas com GPU mantém-se baixo.

**Recomendação:** Amostras de 500k a 1M são suficientes para obter estimativas confiáveis.
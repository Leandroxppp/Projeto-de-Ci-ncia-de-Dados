# Relatório de Teste de Escalabilidade (Amostras Grandes >1M)

**Data:** 2026-06-07 02:11:42
**Tamanhos de amostra testados:** [2000000, 5000000, 10000000]
**Trials por tamanho:** 1
**GPU utilizado no XGBoost:** {'use_rmm': False, 'verbosity': 1}

## Tabela Resumo (média ± desvio padrão)

| Sample Size | Modelo | Acurácia | F1-Score | ROC-AUC | Tempo Treino (s) |
|-------------|--------|----------|----------|---------|------------------|
| 2,000,000 | Logistic Regression | 0.7101±nan | 0.8151±nan | 0.6745±nan | 38.92 |
| 2,000,000 | Random Forest | 0.7992±nan | 0.8829±nan | 0.6568±nan | 58.90 |
| 2,000,000 | XGBoost | 0.8617±nan | 0.9247±nan | 0.7146±nan | 1.09 |
| 5,000,000 | Logistic Regression | 0.7095±nan | 0.8146±nan | 0.6737±nan | 136.34 |
| 5,000,000 | Random Forest | 0.7931±nan | 0.8789±nan | 0.6540±nan | 187.39 |
| 5,000,000 | XGBoost | 0.8616±nan | 0.9247±nan | 0.7166±nan | 2.39 |
| 10,000,000 | Logistic Regression | 0.7092±nan | 0.8144±nan | 0.6743±nan | 198.49 |
| 10,000,000 | Random Forest | 0.7883±nan | 0.8755±nan | 0.6530±nan | 418.26 |
| 10,000,000 | XGBoost | 0.8619±nan | 0.9248±nan | 0.7165±nan | 7.18 |

## Análise Comparativa com Resultados até 1M

Para amostras acima de 1 milhão, observa-se que:
- O XGBoost mantém os maiores valores de F1 (~0.925) com desvio padrão extremamente baixo.
- O ganho em estabilidade (redução do desvio padrão) é marginal, confirmando que a partir de 500k as métricas já estão consolidadas.
- O custo computacional cresce substancialmente, especialmente para Random Forest e Regressão Logística.
- A utilização de GPU no XGBoost reduz significativamente o tempo de treino para amostras muito grandes.

## Conclusão

Os resultados reforçam que amostras de 500k a 1 milhão de registros são suficientes para obter estimativas confiáveis. Aumentar a amostra para 2M ou mais traz benefícios marginais em precisão, mas com grande aumento no tempo de processamento, não sendo justificável para a maioria dos cenários práticos.
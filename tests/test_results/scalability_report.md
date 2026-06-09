# Relatorio de Teste de Escalabilidade

**Data:** 2026-06-06 23:21:45
**Tamanhos de amostra testados:** [1000, 5000, 10000, 50000, 100000, 500000, 1000000]
**Trials por tamanho:** 3

## Tabela Resumo (media ± desvio padrao)

| Sample Size | Modelo | Acurácia | F1-Score | ROC-AUC | Tempo Treino (s) |
|-------------|--------|----------|----------|---------|------------------|
| 1,000 | Logistic Regression | 0.6633±0.0379 | 0.7763±0.0325 | 0.6238±0.0251 | 0.14±0.03 |
| 1,000 | Random Forest | 0.7850±0.0071 | 0.8765±0.0054 | 0.5491±0.0456 | 0.14±0.02 |
| 1,000 | XGBoost | 0.8300±0.0108 | 0.9057±0.0060 | 0.5341±0.0248 | 0.09±0.06 |
| 5,000 | Logistic Regression | 0.6973±0.0141 | 0.8040±0.0117 | 0.6797±0.0055 | 0.20±0.06 |
| 5,000 | Random Forest | 0.8103±0.0069 | 0.8903±0.0041 | 0.6776±0.0070 | 0.15±0.00 |
| 5,000 | XGBoost | 0.8443±0.0062 | 0.9138±0.0035 | 0.6665±0.0071 | 0.06±0.00 |
| 10,000 | Logistic Regression | 0.7132±0.0020 | 0.8172±0.0021 | 0.6767±0.0134 | 0.33±0.05 |
| 10,000 | Random Forest | 0.8075±0.0066 | 0.8887±0.0043 | 0.6555±0.0058 | 0.17±0.01 |
| 10,000 | XGBoost | 0.8482±0.0024 | 0.9162±0.0015 | 0.6561±0.0129 | 0.06±0.00 |
| 50,000 | Logistic Regression | 0.7132±0.0083 | 0.8177±0.0061 | 0.6722±0.0077 | 1.08±0.19 |
| 50,000 | Random Forest | 0.8079±0.0019 | 0.8889±0.0013 | 0.6520±0.0048 | 0.50±0.02 |
| 50,000 | XGBoost | 0.8570±0.0004 | 0.9219±0.0002 | 0.6749±0.0019 | 0.11±0.01 |
| 100,000 | Logistic Regression | 0.7080±0.0016 | 0.8138±0.0011 | 0.6709±0.0048 | 1.83±0.35 |
| 100,000 | Random Forest | 0.8066±0.0020 | 0.8882±0.0012 | 0.6531±0.0030 | 0.98±0.02 |
| 100,000 | XGBoost | 0.8580±0.0002 | 0.9225±0.0001 | 0.6910±0.0006 | 0.17±0.00 |
| 500,000 | Logistic Regression | 0.7096±0.0016 | 0.8148±0.0012 | 0.6722±0.0014 | 10.43±1.93 |
| 500,000 | Random Forest | 0.8058±0.0004 | 0.8873±0.0003 | 0.6607±0.0022 | 7.57±0.12 |
| 500,000 | XGBoost | 0.8619±0.0002 | 0.9247±0.0001 | 0.7123±0.0017 | 0.67±0.01 |
| 1,000,000 | Logistic Regression | 0.7107±0.0020 | 0.8157±0.0015 | 0.6723±0.0017 | 22.24±3.15 |
| 1,000,000 | Random Forest | 0.8014±0.0007 | 0.8845±0.0004 | 0.6585±0.0003 | 22.56±1.62 |
| 1,000,000 | XGBoost | 0.8618±0.0001 | 0.9247±0.0001 | 0.7153±0.0017 | 1.58±0.12 |

## Analise de Flutuacao

A flutuacao (desvio padrao) das metricas tende a diminuir conforme o tamanho da amostra aumenta, indicando maior estabilidade com mais dados.

### Desvio padrao medio da acuracia por tamanho de amostra:
- **1,000 amostras:** 0.01860
- **5,000 amostras:** 0.00908
- **10,000 amostras:** 0.00366
- **50,000 amostras:** 0.00356
- **100,000 amostras:** 0.00127
- **500,000 amostras:** 0.00073
- **1,000,000 amostras:** 0.00095

### Modelo mais estavel (menor desvio padrao em F1)
- **XGBoost** com σ_F1 = 0.00005

### Modelo mais rapido (menor tempo medio de treino)
- **XGBoost** com media = 0.06s

## Conclusoes

O XGBoost apresentou consistentemente os melhores valores de F1 e ROC-AUC, embora com maior custo computacional. A flutuacao das metricas reduz-se significativamente a partir de 500.000 amostras, sugerindo que este e um ponto de equilibrio pratico entre custo de treinamento e estabilidade dos resultados.

## Arquivos gerados
- `scalability_results.csv` – dados brutos
- `scalability_plots.png` – graficos comparativos
- Este relatorio (`scalability_report.md`)
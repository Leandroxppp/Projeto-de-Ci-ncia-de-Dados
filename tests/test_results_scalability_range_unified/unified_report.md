# Teste de Escalabilidade Unificado (15M a 30M)

**Data:** 2026-06-07 01:40:02
**Tamanhos testados:** [15000000, 20000000, 25000000, 30000000]
**GPU utilizada:** Sim (XGBoost com device='cuda')

## Tabela de Resultados

| Amostra | Acurácia | Precisão | Recall | F1-Score | ROC-AUC | Tempo Treino (s) |
|---------|----------|----------|--------|----------|---------|------------------|
| 15,000,000 | 0.8615 | 0.8660 | 0.9921 | 0.9247 | 0.7125 | 5.16 |
| 20,000,000 | 0.8616 | 0.8661 | 0.9920 | 0.9248 | 0.7117 | 6.75 |
| 25,000,000 | 0.8615 | 0.8662 | 0.9918 | 0.9247 | 0.7112 | 8.25 |
| 30,000,000 | 0.8615 | 0.8662 | 0.9918 | 0.9247 | 0.7113 | 9.87 |

## Comportamento Observado

- O F1-Score manteve-se estável em torno de **0.9248** para todos os tamanhos.
- O ROC-AUC ficou próximo de **0.7117**.
- O tempo de treino aumentou de **5.16s** (15M) para **9.87s** (30M), com aceleração por GPU.
- A utilização de memória RAM permaneceu controlada (máximo 75-80%), confirmando a eficiência da leitura chunked.

## Conclusão

Os testes com amostras entre 15 e 30 milhões de registros confirmam a tendência observada anteriormente: o modelo XGBoost atinge estabilidade já com 500k-1M amostras. Aumentos adicionais no volume de dados não trazem ganhos significativos em métricas, apenas custo computacional.

## Arquivos Gerados

- `all_results.csv` – dados consolidados
- `comparison_plots.png` – gráficos F1 e tempo vs sample size
- Este relatório (`unified_report.md`)

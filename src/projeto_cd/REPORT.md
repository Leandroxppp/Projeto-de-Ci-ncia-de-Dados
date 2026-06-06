# Relatório Completo — Pipeline de Recomendação (Execução com 200.000 samples)

**Run timestamp:** 2026-06-06T00:30:47.660682Z
**sample_size utilizado:** 200000

## 1. Aplicação
Este trabalho propõe uma análise de recomendação de jogos focada em prever se uma aquisição será recomendada pelo usuário (`is_recommended`).
Objetivos:
- Construir e comparar modelos de classificação capazes de predizer recomendações;
- Testar hipóteses sobre o efeito de preço e desconto na satisfação (H2);
- Produzir artefatos interpretáveis (importâncias, SHAP) para apoiar decisões de produto (promoções, precificação).

Justificativa: recomendações automatizadas ajudam a melhorar retenção e conversão — entender o impacto de desconto e faixa de preço apoia ações comerciais (promoções, curadoria).

## 2. Base de Dados
Origem: arquivos do repositório do projeto local (`src/projeto_cd/`): [recommendations.csv](src/projeto_cd/recommendations.csv), [games.csv](src/projeto_cd/games.csv), [users.csv](src/projeto_cd/users.csv) e [games_metadata_formatado.json](src/projeto_cd/games_metadata_formatado.json).

Contexto e quantidade: o pipeline executou uma subamostragem estratificada para viabilizar processamento — `sample_size=200000` registros da tabela `recommendations` foram usados nesta execução.

Principais variáveis usadas (colunas selecionadas):
- `app_id`, `user_id`, `is_recommended` (target), `hours` (playtime), `price_final`, `discount`, `products`, `reviews`, `tags`, `description`.

Pré-processamento aplicado:
- Leitura robusta de JSON (JSON Lines com fallback para array JSON);
- Subamostragem estratificada por `is_recommended` para `sample_size` registros;
- Merge relacional entre `recommendations`, `games`, `users` e `metadata` (join em `app_id` / `user_id`);
- Tratamento de missings: `dropna` em `is_recommended`, `hours`, `price_final`; preenchimento com 0 em `products` e `reviews`;
- Engenharia: criação de `is_free`, `playtime_category` (bins), `price_tier`, `has_discount`, `num_tags`;
- One-hot encoding de `playtime_category` e `price_tier` (drop_first=True).

## 3. Estatística Descritiva e Inferência
Foram geradas visualizações e tabelas para entender distribuições e proporções:
- Histogramas/Barplots por `playtime_category` (`h1_playtime.png`);
- Taxas de recomendação por `price_tier` e por `has_discount` (`h2_price_tiers.png`, `h2_discount_effect.png`);
- Comparação Gratuito vs Pago (`h3_free_vs_paid.png`).

Testes de hipótese (H2): diferença entre proporções foi avaliada via teste Z para duas proporções. Sumário salvo em [plots/stat_tests_summary.txt](plots/stat_tests_summary.txt#L1-L5).

Resultados chave (H2):
- Desconto vs Preço Cheio: z = -2.9706, p = 0.002972 → diferença estatisticamente significativa (p < 0.01).
- Premium (>$60) vs Barato (<$10): z = -11.5305, p ≈ 0 → diferença altamente significativa.
- Premium (>$60) vs Medio ($10-$30): z = -12.5913, p ≈ 0 → diferença altamente significativa.

Essas evidências suportam a hipótese de que desconto e faixa de preço influenciam a proporção de recomendações.

## 4. Métodos Avaliados
Modelos treinados e comparados:
- `LogisticRegression` (baseline, regularizada);
- `RandomForestClassifier` (árvore ensemble);
- `XGBoost` (`XGBClassifier`) — modelo final selecionado.

Justificativa: mix de baseline linear e modelos de árvore para capturar interações e não-linearidades; XGBoost escolhido por melhor desempenho em F1/AUC no holdout.

## 5. Métricas de Avaliação
Métricas utilizadas no pipeline:
- F1-score (principal métrica para classificação binária desequilibrada);
- AUC-ROC (qualidade de ranking/probabilidade);
- Curvas ROC por modelo e comparação por fold (PNG gerados);
- Também geradas importâncias de features e gráficos SHAP para interpretabilidade.

## 6. Métodos de Avaliação
Estratégia aplicada:
- Divisão treino/holdout: 80% treino / 20% teste, estratificada por `is_recommended`;
- Validação: Stratified K-Fold com `k=5` sobre o conjunto de treino para obter média e desvio padrão de F1 e AUC;

Justificativa: Estratificação preserva a proporção do target em folds e holdout; k-fold é adequado para estimar variância de desempenho em amostras moderadamente grandes.

## 7. Resultados e Discussão
Métricas (execução com `sample_size=200000`):
- **Regressão Logística:** CV F1 = 0.8150 ± 0.0022 — Holdout F1 = 0.8133 | AUC = 0.6707
- **Random Forest:** CV F1 = 0.8883 ± 0.0016 — Holdout F1 = 0.8897 | AUC = 0.6620
- **XGBoost (melhor):** CV F1 = 0.9237 ± 0.0004 — Holdout F1 = 0.9241 | AUC = 0.7037

Discussão:
- O modelo XGBoost apresentou superioridade consistente em F1 e AUC no holdout, indicando alta capacidade de discriminação e boa generalização no conjunto testado.
- Resultados de H2 indicam que títulos premium têm taxas de recomendação menores, e descontos aumentam a proporção de recomendações — isso pode orientar estratégias de promoção (por exemplo, focar descontos em títulos premium para aumentar satisfação aparente).

Limitações:
- Subamostragem foi usada; resultados dependem da representatividade da amostra estratificada;
- Variáveis textuais (`description`) foram apenas parcialmente utilizadas (contagem de tags), podendo melhorar com NLP;
- Não foram avaliadas métricas de custo/benefício de intervenções comerciais.

Recomendação de decisão: considerar implantação de um classificador XGBoost em um fluxo de recomendação A/B para medir impacto real em métricas de negócio (conversão/retention). Use a probabilidade predita para priorizar ofertas/promos em títulos onde desconto aumenta probabilidade de recomendação.

## 8. Conclusão
Resumo: pipeline processou `200000` amostras, realizou testes estatísticos e comparou três abordagens de classificação. XGBoost foi selecionado como melhor modelo (Holdout F1 = 0.9241, AUC = 0.7037). Testes inferenciais mostraram efeitos significativos de faixa de preço e desconto sobre a proporção de recomendações.

Possíveis trabalhos futuros:
- Expandir uso de NLP nas descrições para extrair features semânticas;
- Testar calibração de probabilidades e otimização de thresholds alinhados a KPIs de negócio;
- Avaliar influência temporal (quando disponível) e aprendizado online.

---

### Saída do pipeline (log resumido desta execução)
```
Run metadata registrada: 2026-06-06T00:41:05.936168Z | sample_size=200000 -> c:\Users\niujo\Desktop\Ciência de Dados\projeto_final\projeto_cd\src\projeto_cd\plots\run_metadata.txt
1. Carregando base de recomendações (recommendations.csv)...
2. Aplicando subamostragem estratificada (200000 registros)...
3. Carregando tabelas de suporte (games, users, metadata)...
4. Executando integração relacional (Merges)...
5. Engenharia de Atributos e Limpeza...
6. Executando Análise Exploratória de Dados (Plots)...
Gerando gráficos para a Hipótese 2 (Preço e Satisfação)...

Resultados dos testes estatísticos (H2) salvos em: c:\Users\niujo\Desktop\Ciência de Dados\projeto_final\projeto_cd\src\projeto_cd\plots\stat_tests_summary.txt
- Desconto vs Preço Cheio: z=-2.971, p-value=0.002972
- Comparação: Premium (>$60) vs Barato (<$10): z=-11.531, p-value=0
- Comparação: Premium (>$60) vs Medio ($10-$30): z=-12.591, p-value=0
7. Divisão de conjuntos e treinamento de modelos...
[Regressão Logística (Baseline)] CV (5-fold) -> F1: 0.8150 ± 0.0022 | AUC: 0.6713 ± 0.0051
[Regressão Logística (Baseline)] Holdout -> F1-Score: 0.8133 | AUC-ROC: 0.6707
[Random Forest] CV (5-fold) -> F1: 0.8883 ± 0.0016 | AUC: 0.6541 ± 0.0031
[Random Forest] Holdout -> F1-Score: 0.8897 | AUC-ROC: 0.6620
[XGBoost Classifier] CV (5-fold) -> F1: 0.9237 ± 0.0004 | AUC: 0.6970 ± 0.0034
[XGBoost Classifier] Holdout -> F1-Score: 0.9241 | AUC-ROC: 0.7037
Melhor modelo 'XGBoost Classifier' salvo em: c:\Users\niujo\Desktop\Ciência de Dados\projeto_final\projeto_cd\src\projeto_cd\plots\best_model.joblib
8. Computando explicabilidade com SHAP Explainer...

Pipeline executado com sucesso. Gráficos exportados para o diretório local.
```

*Relatório gerado automaticamente pelo pipeline — arquivo atualizado para refletir a execução com `sample_size=200000`.*

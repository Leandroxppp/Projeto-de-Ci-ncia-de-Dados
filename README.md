# 🎮 Projeto de Ciência de Dados

## Predição de Satisfação e Recomendação de Jogos na Steam

Projeto de Ciência de Dados desenvolvido com foco na predição de satisfação de usuários da plataforma Steam utilizando técnicas de Machine Learning, Explainable AI e Business Intelligence.

O projeto foi desenvolvido para a disciplina de Ciência de Dados da Universidade Federal de Alagoas (UFAL).

---

## 📌 Problema

A indústria de jogos digitais enfrenta um grande desafio com mais de **50 mil jogos** na Steam, onde uma parcela significativa dos títulos comprados nunca chega a ser jogada. Isso gera impactos diretos em:

| Área | Impacto |
|------|--------|
| 👤 Experiência do usuário | Recomendações irrelevantes diminuem o engajamento |
| 📈 Retenção de jogadores | Jogadores insatisfeitos abandonam a plataforma |
| 📢 Marketing de jogos | Dificuldade em posicionar títulos para o público certo |
| 💰 Precificação | Estratégias de preço sem respaldo em dados de satisfação |
| 🔍 Visibilidade dos títulos | Jogos de qualidade soterrados por lançamentos genéricos |
| 🏢 Decisões estratégicas | Publishers sem métricas para direcionar investimentos |

Diante disso, este projeto busca **prever se um usuário recomendará ou não um jogo** com base em características comportamentais e atributos dos jogos.

---

## 🎯 Objetivos

| Objetivo | Descrição |
|----------|-----------|
| 🔮 Predizer | Recomendações positivas e negativas com modelos de classificação |
| 🔍 Identificar | Fatores que mais influenciam a satisfação (preço, tempo jogado, descontos) |
| 💡 Gerar | Insights acionáveis para publishers e desenvolvedores |
| 📊 Construir | Visualizações e dashboards interativos para análise de negócio |

---

## 📊 Base de Dados

**Dataset:** [Game Recommendations on Steam](https://www.kaggle.com/datasets/antonkozyriev/game-recommendations-on-steam)  

| Característica | Valor |
|---------------|-------|
| 📁 Arquivos | `games.csv`, `users.csv`, `recommendations.csv`, `games_metadata.json` |
| 📝 Avaliações | **+41 milhões** de recomendações |
| 👥 Usuários | Milhares de interações entre usuários e jogos |
| 🏷️ Metadados | Tags, descrições e categorias dos jogos |
| ✅ Target | `is_recommended` — binário (recomenda / não recomenda) |

---

## 🧠 Tecnologias Utilizadas

| Categoria | Tecnologias |
|-----------|-------------|
| 🐍 Linguagem | Python 3.14+ |
| 📊 Manipulação de dados | Pandas, NumPy |
| 📈 Visualização | Matplotlib, Seaborn, Plotly |
| 🤖 Machine Learning | Scikit-Learn, XGBoost, LightGBM |
| 🔬 Explainable AI | SHAP |
| 🖥️ Aceleração | CUDA (GPU) via XGBoost |
| 🎛️ Dashboard | Streamlit |
| 📦 Gerenciamento | Poetry |

---

## ⚙️ Etapas do Projeto

| Etapa | Atividades |
|-------|------------|
| **1.** Entendimento da Base | Análise das entidades, integração dos datasets, definição do problema de negócio |
| **2.** Pré-processamento | Tratamento de valores ausentes, remoção de inconsistências, transformação de variáveis, engenharia de atributos |
| **3.** Análise Exploratória | Distribuição do target, relação preço-satisfação, impacto do tempo jogado, comparação free vs pago |
| **4.** Modelagem | Regressão Logística, Random Forest, **XGBoost** (selecionado) |
| **5.** Avaliação | Accuracy, Precision, Recall, **F1-Score**, ROC-AUC — com Stratified K-Fold CV |
| **6.** Explainable AI | SHAP para explicar decisões dos modelos |
| **7.** Dashboard | Streamlit com filtros dinâmicos e gráficos interativos |

---

## 📈 Principais Insights

| Insight | Conclusão |
|---------|-----------|
| ⏱️ Tempo moderado de gameplay | Maiores taxas de recomendação |
| 💸 Jogos caros com pouco conteúdo | Maior insatisfação |
| 🆓 Jogos gratuitos (F2P) | Comportamento de recomendação diferente dos pagos |
| 🏷️ Descontos promocionais | Influenciam positivamente a percepção de valor |

---

## 💡 Recomendações Estratégicas

| Ação | Benefício |
|------|-----------|
| 💲 Otimizar estratégias de precificação | Aumentar conversão sem sacrificar satisfação |
| 🎯 Campanhas promocionais direcionadas | Melhor ROI em descontos sazonais |
| ⚙️ Melhorar sistemas de recomendação | Aumentar retenção e engajamento |
| 👑 Foco em retenção para jogos premium | Reduzir churn em títulos de alto valor |
| 🧠 Personalizar ofertas por comportamento | Maior taxa de aceitação de recomendações |

---

## 📊 Dashboard

| Funcionalidade | Descrição |
|---------------|-----------|
| 🔍 Filtros dinâmicos | Selecione por faixa de preço, categoria de horas, modelo de monetização |
| 📈 Gráficos interativos | Visualizações das hipóteses H1, H2 e H3 |
| ⭐ Análise de satisfação | Taxas de recomendação por segmento |
| 🎯 Importância das variáveis | Feature importance e SHAP summary |
| 🔎 Exploração de padrões | Correlações entre atributos e o target |

---

## 🚀 Como Executar

### 1. Clonar o Repositório

```bash
git clone https://github.com/Leandroxppp/Projeto-de-Ci-ncia-de-Dados.git
cd Projeto-de-Ci-ncia-de-Dados
```

### 2. Instalar o Poetry

Caso ainda não tenha o Poetry instalado, siga a [documentação oficial](https://python-poetry.org/docs/#installation):

<details>
<summary>Instalar o Poetry</summary>

```bash
# Linux / macOS / WSL
curl -sSL https://install.python-poetry.org | python3 -

# Windows (PowerShell)
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -
```

</details>

> ⚠️ Certifique-se de que o diretório de instalação do Poetry está no `PATH` do seu sistema.

### 3. Preparar os Dados

Baixe os arquivos do dataset [Game Recommendations on Steam](https://www.kaggle.com/datasets/antonkozyriev/game-recommendations-on-steam) e coloque-os na pasta `data/`:

```
data/
├── recommendations.csv
├── games.csv
├── users.csv
├── games_metadata.json
└── games_metadata_formatado.json   # (opcional, gerado pelo formatador)
```

Caso tenha o arquivo original `games_metadata.json` no formato JSON Lines, é possível gerar o `games_metadata_formatado.json` executando:

```bash
poetry run python -m projeto_cd.utils.formatador
```

### 4. Instalar as Dependências

```bash
poetry install
```

Isso criará um ambiente virtual isolado e instalará todas as dependências listadas no `pyproject.toml`.

### 5. Executar o Pipeline

O pipeline completo executa as etapas de carregamento, análise exploratória, testes estatísticos, treinamento de modelos e interpretabilidade SHAP.

**Pipeline completo (recomendado):**

```bash
poetry run pipeline
```

**Ou como módulo Python:**

```bash
poetry run python -m projeto_cd
```

A saída incluirá:
- Gráficos das hipóteses H1, H2 e H3 salvos em `src/projeto_cd/plots/`
- Testes estatísticos (Z-test) salvos em `stat_tests_summary.txt`
- Curvas ROC, métricas por fold e importância das features
- Gráfico SHAP de interpretabilidade global
- Modelo com melhor desempenho persistido (`best_model.joblib`)

### 6. Executar o Dashboard

```bash
poetry run streamlit run dashboard/app.py
```

---

### 📦 Caso precise adicionar novas dependências

```bash
poetry add nome-do-pacote
```

> [!NOTE]
> Todas as dependências do projeto estão gerenciadas pelo `pyproject.toml` e pelo `poetry.lock`, garantindo reprodutibilidade do ambiente.

---

## 🖥️ Suporte a GPU (CUDA)

O pipeline detecta **automaticamente** se uma GPU compatível com CUDA está disponível e ativa a aceleração no XGBoost. Nenhuma configuração manual é necessária.

Para forçar o uso de CPU mesmo com GPU disponível:

```python
from projeto_cd.pipeline import executar_pipeline
executar_pipeline()  # usa GPU se disponível

# Ou explicitamente:
from projeto_cd.modelos.treinamento import treinar_e_avaliar
treinar_e_avaliar(df_modelo, use_gpu=False)
```

### Ambiente Conda para GPU (opcional)

Caso o Poetry não encontre a biblioteca XGBoost com suporte CUDA no ambiente atual, crie um ambiente Conda dedicado:

```bash
conda create -n xgb-gpu python=3.12 -y
conda activate xgb-gpu
conda install -c conda-forge py-xgboost-gpu pandas numpy scikit-learn matplotlib seaborn shap -y
pip install poetry
poetry install
```

---

## 🧪 Testes de Escalabilidade

Scripts para avaliar o desempenho dos modelos em diferentes volumes de dados, com suporte a GPU.

### Teste de flutuação (1k a 1M amostras)

Compara Regressão Logística, Random Forest e XGBoost com 3 repetições por tamanho:

```bash
poetry run python tests/test_scalability.py
```

### Teste de amostras grandes (2M, 5M e 10M)

Utiliza leitura eficiente (chunked sampling) para não carregar todo o dataset na memória:

```bash
poetry run python tests/test_scalability_large.py
```

### Teste unificado (15M a 30M, apenas XGBoost)

Permite especificar tamanhos personalizados via argumentos:

```bash
poetry run python tests/test_scalability_range_unified.py
poetry run python tests/test_scalability_range_unified.py 15000000 20000000 25000000
```

> **Nota:** Os resultados (CSV, gráficos e relatórios) são salvos em subpastas dentro de `tests/`. Recomenda-se pelo menos 16 GB de RAM para amostras acima de 10M.

---

## 📁 Estrutura do Projeto

```
Projeto-de-Ci-ncia-de-Dados/
├── data/                           # 📁 Arquivos de dados (.csv, .json)
├── src/
│   └── projeto_cd/                 # 📦 Pacote principal
│       ├── __init__.py
│       ├── __main__.py             # Ponto de entrada (python -m projeto_cd)
│       ├── config.py               # Configurações globais (caminhos, constantes, GPU)
│       ├── pipeline.py             # Orquestrador do pipeline
│       ├── main.py                 # Entry point do Poetry
│       ├── dados/
│       │   ├── carregamento.py     # Leitura, subamostragem e merges
│       │   └── engenharia_atributos.py  # Feature engineering
│       ├── analise/
│       │   ├── exploratoria.py     # Gráficos H1, H2, H3
│       │   ├── testes_estatisticos.py  # Testes Z para proporções
│       │   └── interpretabilidade.py   # SHAP explainability
│       ├── modelos/
│       │   └── treinamento.py      # Treino, CV e avaliação com GPU
│       ├── utils/
│       │   ├── utilitarios.py      # Funções auxiliares
│       │   └── formatador.py       # Script de formatação JSON
│       └── plots/                  # 📈 Gráficos e artefatos gerados
├── tests/                          # 🧪 Scripts de escalabilidade
│   ├── test_scalability.py
│   ├── test_scalability_large.py
│   ├── test_scalability_range_unified.py
│   └── test_results*/              # Resultados das execuções
├── dashboard/                      # 📊 Dashboard Streamlit
├── pyproject.toml
└── README.md
```

---

## 👨‍💻 Equipe

* Eduardo Maciel Alexandre
* Josenilton Ferreira da Silva Junior
* Leandro Marcio Elias da Silva
* Sthefany Barboza de Lima

Universidade Federal de Alagoas (UFAL)

---

## 📄 Licença

Projeto desenvolvido para fins acadêmicos.

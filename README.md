# 🎮 Projeto de Ciência de Dados

## Predição de Satisfação e Recomendação de Jogos na Steam

Projeto de Ciência de Dados desenvolvido com foco na predição de satisfação de usuários da plataforma Steam utilizando técnicas de Machine Learning, Explainable AI e Business Intelligence.

O projeto foi desenvolvido para a disciplina de Ciência de Dados da Universidade Federal de Alagoas (UFAL).

---

# 📌 Problema

A indústria de jogos digitais enfrenta um grande desafio relacionado à satisfação dos usuários e à eficiência dos sistemas de recomendação.

Atualmente, a Steam possui mais de 50 mil jogos disponíveis, e uma parcela significativa dos títulos comprados nunca chega a ser jogada. Isso gera impactos diretos em:

* experiência do usuário;
* retenção de jogadores;
* marketing de jogos;
* precificação;
* visibilidade dos títulos;
* decisões estratégicas de publishers e desenvolvedores.

Diante disso, este projeto busca prever se um usuário recomendará ou não um jogo com base em características comportamentais e atributos dos jogos.

---

# 🎯 Objetivos

* Prever recomendações positivas e negativas;
* Identificar os fatores que mais influenciam a satisfação;
* Gerar insights acionáveis;
* Apoiar decisões estratégicas para publishers e desenvolvedores;
* Construir visualizações e dashboards interativos para análise de negócio.

---

# 📊 Base de Dados

Dataset utilizado:

**Game Recommendations on Steam**

Fonte:
https://www.kaggle.com/datasets/antonkozyriev/game-recommendations-on-steam

Arquivos principais:

* games.csv
* users.csv
* recommendations.csv

A base contém:

* mais de 41 milhões de avaliações;
* interações entre usuários e jogos;
* metadados dos jogos;
* recomendações binárias (recomenda/não recomenda).

---

# 🧠 Tecnologias Utilizadas

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-Learn
* XGBoost
* LightGBM
* SHAP
* Streamlit
* Plotly

---

# ⚙️ Etapas do Projeto

## 1. Entendimento da Base

* análise das entidades;
* integração dos datasets;
* definição do problema de negócio.

## 2. Pré-processamento

* tratamento de valores ausentes;
* remoção de inconsistências;
* transformação de variáveis;
* engenharia de atributos.

## 3. Análise Exploratória (EDA)

* distribuição da variável alvo;
* relação entre preço e satisfação;
* impacto do tempo jogado;
* comparação entre jogos gratuitos e pagos;
* análise de gêneros e padrões de recomendação.

## 4. Modelagem

Modelos avaliados:

* Regressão Logística
* Random Forest
* XGBoost

## 5. Avaliação

Métricas utilizadas:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC
* Matriz de Confusão

Estratégias:

* Train/Test Split
* Stratified K-Fold Cross Validation

## 6. Explainable AI

Utilização de SHAP (SHapley Additive exPlanations) para explicar as decisões dos modelos.

## 7. Dashboard Interativo

Dashboard desenvolvido em Streamlit para exploração visual e análise interativa dos resultados.

---

# 📈 Principais Insights

* Jogos com tempo moderado de gameplay apresentam maiores taxas de recomendação.
* Jogos caros tendem a gerar maior insatisfação quando oferecem pouco conteúdo jogável.
* Jogos gratuitos apresentam comportamento de recomendação diferente dos jogos pagos.
* Descontos influenciam positivamente a percepção de valor do usuário.

---

# 💡 Recomendações Estratégicas

Com base nas análises realizadas, o projeto sugere:

* otimização de estratégias de precificação;
* campanhas promocionais direcionadas;
* melhoria de sistemas de recomendação;
* foco em retenção para jogos premium;
* uso de comportamento do usuário para personalização de ofertas.

---

# 📊 Dashboard

O projeto inclui um dashboard interativo desenvolvido com Streamlit contendo:

* filtros dinâmicos;
* gráficos interativos;
* análise de satisfação;
* visualização de importância das variáveis;
* exploração de padrões de comportamento.

---

# 🚀 Como Executar

## 1. Clonar o Repositório

```bash
git clone https://github.com/Leandroxppp/Projeto-de-Ci-ncia-de-Dados.git
cd Projeto-de-Ci-ncia-de-Dados
```

## 2. Instalar o Poetry

Caso ainda não tenha o Poetry instalado, siga a [documentação oficial](https://python-poetry.org/docs/#installation):

```bash
# Linux / macOS / WSL
curl -sSL https://install.python-poetry.org | python3 -

# Windows (PowerShell)
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -
```

> ⚠️ Certifique-se de que o diretório de instalação do Poetry está no `PATH` do seu sistema.

## 3. Instalar as Dependências

Com o Poetry instalado, instale as dependências do projeto:

```bash
poetry install
```

Isso criará um ambiente virtual isolado e instalará todas as dependências listadas no `pyproject.toml`.

## 4. Ativar o Ambiente Virtual

```bash
poetry env activate
```

Ou execute comandos diretamente sem ativar o shell:

```bash
poetry run comando
```

## 5. Executar o Dashboard

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

## 5. Testes de Escalabilidade com GPU (Opcional)

Para executar os testes de escalabilidade com suporte a GPU (aceleração via CUDA), siga os passos abaixo.

### 5.1. Criar ambiente Conda com suporte GPU

```bash
conda create -n xgb-gpu python=3.12 -y
conda activate xgb-gpu
conda install -c conda-forge py-xgboost-gpu pandas numpy scikit-learn matplotlib seaborn shap -y
```

### 5.2. Executar os scripts de teste

Todos os scripts de teste estão na pasta tests/.

Teste de flutuação (1k a 1M amostras):
```bash
python tests/test_scalability.py
```

Teste de 2M, 5M e 10M amostras (comparativo entre modelos):
```bash
python tests/test_scalability_large.py
```

Teste unificado de 15M a 30M amostras (apenas XGBoost):
```bash
python tests/test_range_unified.py
```

Os resultados (CSV, gráficos e relatórios) serão salvos em subpastas dentro de tests/ (ex.: test_results_range_unified/, test_results_large/).

    Nota: Os scripts de teste utilizam leitura eficiente (chunked sampling) para não carregar todo o dataset na memória. Certifique‑se de ter pelo menos 16 GB de RAM disponível para amostras até 30M.

---

# 📁 Estrutura do Projeto

```bash
projeto/
├── data/
├── notebooks/
├── src/
├── dashboard/
├── reports/
├── models/
└── images/
```

---

# 👨‍💻 Equipe

* Eduardo Maciel Alexandre
* Josenilton Ferreira da Silva Junior
* Leandro Marcio Elias da Silva
* Sthefany Barboza de Lima

Universidade Federal de Alagoas (UFAL)

---

# 📄 Licença

Projeto desenvolvido para fins acadêmicos.

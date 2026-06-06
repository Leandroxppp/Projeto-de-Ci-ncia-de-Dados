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
git clone https://github.com/seuusuario/steaminsight-ai.git
```

## 2. Instalar Dependências

```bash
pip install -r requirements.txt
```

## 3. Executar Dashboard

```bash
streamlit run dashboard/app.py
```

---

# 📁 Estrutura do Projeto

```bash
steaminsight-ai/
│
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
<<<<<<< HEAD

1. Criando um novo projetoPara criar um projeto do zero já com a estrutura básica, use:

`poetry new meu-projeto`
`cd meu-projeto`

Alternativamente, se você já tem uma pasta com códigos, basta entrar na pasta e iniciar o Poetry com: `poetry init`.

2. Instalando e gerenciando dependênciasPara instalar bibliotecas, use o comando add. O Poetry vai adicionar o pacote automaticamente ao seu arquivo pyproject.toml e criar um ambiente virtual se ele não existir.

`poetry add requests pandas`

Para instalar dependências de desenvolvimento (como ferramentas de testes ou linters), adicione a flag --group dev:

`poetry add pytest --group dev`

3. Removendo dependênciasPara remover um pacote e atualizar o ambiente, use:

`poetry remove requests`

4. Rodando seu código dentro do ambiente virtualNão há necessidade de ativar e desativar o ambiente virtual manualmente com o Poetry. Para executar qualquer script dentro do ambiente isolado, use:

`poetry run python main.py`

Caso prefira abrir um terminal dentro do ambiente virtual isolado, basta digitar:

`poetry shell`

5. Configurando dependências existentesSe você clonou um projeto do GitHub que já possui o arquivo pyproject.toml, basta baixar as dependências exigidas executando:

`poetry install`

Para rodar:

`poetry run python src/projeto_cd/main.py`
=======
>>>>>>> d7b3bef088beddaa66d280b654b81470cd0528bf

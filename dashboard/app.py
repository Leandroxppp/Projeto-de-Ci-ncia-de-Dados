"""
Dashboard Interativo — Projeto de Ciência de Dados (Steam Recommendations)

Este dashboard apresenta os resultados do pipeline de predição de satisfação
de jogos na Steam, incluindo análise exploratória, testes estatísticos,
desempenho de modelos e um preditor interativo.
"""

import warnings
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

# ── Caminhos relativos ao projeto ──────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PLOTS_DIR = PROJECT_ROOT / "src" / "projeto_cd" / "plots"
MODEL_PATH = PLOTS_DIR / "best_model.joblib"
STAT_TESTS_PATH = PLOTS_DIR / "stat_tests_summary.txt"

# ── Configuração da página ────────────────────────────────────────────────
st.set_page_config(
    page_title="Steam Recommendation Insights",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Estilo customizado ────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .main > div {
        padding: 1rem 1.5rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 0.5rem 1.2rem;
        border-radius: 6px 6px 0 0;
        font-weight: 600;
        font-size: 0.95rem;
    }
    .metric-card {
        background: #1e1e2e;
        border-radius: 12px;
        padding: 1.2rem 1rem;
        text-align: center;
        border: 1px solid #333;
        box-shadow: 0 2px 6px rgba(0,0,0,0.15);
    }
    .metric-card .value {
        font-size: 2rem;
        font-weight: 700;
        color: #00d4aa;
    }
    .metric-card .label {
        font-size: 0.85rem;
        color: #aaa;
        margin-top: 0.25rem;
    }
    .insight-box {
        background: #1a1a2e;
        border-left: 4px solid #00d4aa;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin: 1rem 0;
    }
    .insight-box h4 {
        margin: 0 0 0.4rem 0;
        color: #00d4aa;
    }
    .insight-box p {
        margin: 0;
        color: #ccc;
    }
    .prediction-positive {
        background: linear-gradient(135deg, #00d4aa20, #00d4aa08);
        border: 2px solid #00d4aa;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
    }
    .prediction-negative {
        background: linear-gradient(135deg, #ff6b6b20, #ff6b6b08);
        border: 2px solid #ff6b6b;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
    }
    .section-header {
        border-bottom: 2px solid #333;
        padding-bottom: 0.5rem;
        margin-bottom: 1.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Funções auxiliares ────────────────────────────────────────────────────

@st.cache_resource
def carregar_modelo():
    """Carrega o modelo treinado (XGBoost) do disco."""
    if MODEL_PATH.exists():
        try:
            modelo = joblib.load(str(MODEL_PATH))
            return modelo
        except Exception as e:
            st.error(f"Erro ao carregar o modelo: {e}")
            return None
    return None


@st.cache_data
def ler_grafico(nome_arquivo: str):
    """Lê uma imagem PNG do diretório de plots."""
    caminho = PLOTS_DIR / nome_arquivo
    if caminho.exists():
        return str(caminho)
    return None


@st.cache_data
def ler_testes_estatisticos():
    """Lê o resumo dos testes estatísticos."""
    if STAT_TESTS_PATH.exists():
        with open(STAT_TESTS_PATH, "r", encoding="utf-8") as f:
            return f.read()
    return None


def interpretar_resultado(probabilidade: float, threshold: float = 0.5):
    """Retorna o rótulo e a confiança da predição."""
    if probabilidade >= threshold:
        return "recomendado", probabilidade
    else:
        return "não recomendado", 1.0 - probabilidade


def engenharia_para_predicao(
    hours: float,
    price_final: float,
    products: int,
    reviews: int,
    num_tags: int,
) -> pd.DataFrame:
    """
    Aplica a mesma engenharia de atributos usada no treinamento
    para gerar o vetor de features esperado pelo modelo.
    """
    # Cria um DataFrame com uma única linha
    dados = {
        "hours": hours,
        "price_final": price_final,
        "is_free": 1 if price_final == 0 else 0,
        "products": products,
        "reviews": reviews,
        "num_tags": num_tags,
    }

    df = pd.DataFrame([dados])

    # Categoriza tempo de jogo (mesma lógica do pipeline)
    if hours <= 2:
        cat_medio, cat_alto, cat_saturacao = 0, 0, 0
    elif hours <= 20:
        cat_medio, cat_alto, cat_saturacao = 1, 0, 0
    elif hours <= 100:
        cat_medio, cat_alto, cat_saturacao = 0, 1, 0
    else:
        cat_medio, cat_alto, cat_saturacao = 0, 0, 1

    df["playtime_category_Medio"] = cat_medio
    df["playtime_category_Alto"] = cat_alto
    df["playtime_category_Saturacao"] = cat_saturacao

    # Garante a ordem correta das features
    features_order = [
        "hours",
        "price_final",
        "is_free",
        "products",
        "reviews",
        "num_tags",
        "playtime_category_Medio",
        "playtime_category_Alto",
        "playtime_category_Saturacao",
    ]

    return df[features_order]


# ── Sidebar ───────────────────────────────────────────────────────────────

with st.sidebar:
    st.image(
        "https://cdn.akamai.steamstatic.com/store/home/store_home_share.jpg",
        width=300,
    )
    st.markdown("## 🎮 Steam Insights")
    st.markdown(
        "Dashboard de Ciência de Dados para predição de satisfação "
        "e recomendação de jogos na plataforma Steam."
    )
    st.divider()
    st.markdown("**Equipe:**")
    st.markdown("- Eduardo Maciel Alexandre")
    st.markdown("- Josenilton Ferreira da Silva Junior")
    st.markdown("- Leandro Marcio Elias da Silva")
    st.markdown("- Sthefany Barboza de Lima")
    st.divider()
    st.markdown("**UFAL — 2026**")
    st.divider()

    modelo = carregar_modelo()
    if modelo is not None:
        st.success("✅ Modelo carregado com sucesso!")
    else:
        st.warning("⚠️ Modelo não encontrado. Execute o pipeline primeiro.")


# ── Abas principais ───────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "📋 Visão Geral",
        "📊 Análise Exploratória",
        "🧠 Performance do Modelo",
        "🔍 Importância & SHAP",
        "🎯 Preditor Interativo",
    ]
)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — VISÃO GERAL
# ═══════════════════════════════════════════════════════════════════════════

with tab1:
    st.markdown("# 🎮 Predição de Satisfação e Recomendação de Jogos na Steam")
    st.markdown(
        "Este projeto aplica técnicas de **Machine Learning**, **Explainable AI** "
        "e **Business Intelligence** para prever se um usuário recomendará ou não "
        "um jogo na plataforma Steam, com base em características comportamentais "
        "e atributos dos jogos."
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            '<div class="metric-card"><div class="value">0.924</div>'
            '<div class="label">F1-Score (XGBoost)</div></div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            '<div class="metric-card"><div class="value">0.704</div>'
            '<div class="label">AUC-ROC (XGBoost)</div></div>',
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            '<div class="metric-card"><div class="value">~64%</div>'
            '<div class="label">Recomendação Média</div></div>',
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            '<div class="metric-card"><div class="value">3</div>'
            '<div class="label">Modelos Comparados</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("## 📌 O Problema")
    col_a, col_b = st.columns([3, 2])
    with col_a:
        st.markdown(
            """
            A indústria de jogos digitais enfrenta um grande desafio com mais de
            **50 mil jogos** na Steam, onde uma parcela significativa dos títulos
            comprados nunca chega a ser jogada. Isso gera impactos diretos em:

            - **Experiência do usuário:** recomendações irrelevantes diminuem o engajamento
            - **Retenção de jogadores:** jogadores insatisfeitos abandonam a plataforma
            - **Marketing de jogos:** dificuldade em posicionar títulos para o público certo
            - **Precificação:** estratégias de preço sem respaldo em dados de satisfação
            - **Visibilidade:** jogos de qualidade soterrados por lançamentos genéricos

            Diante disso, o projeto busca **prever se um usuário recomendará ou não um jogo**
            com base em características comportamentais e atributos dos jogos.
            """
        )
    with col_b:
        st.markdown(
            """
            <div class="insight-box">
            <h4>🎯 Objetivos</h4>
            <p>
            🔮 <strong>Predizer</strong> recomendações positivas e negativas<br>
            🔍 <strong>Identificar</strong> fatores que mais influenciam a satisfação<br>
            💡 <strong>Gerar</strong> insights acionáveis para publishers<br>
            📊 <strong>Construir</strong> dashboards interativos
            </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("## 🧠 Tecnologias Utilizadas")
    col_t1, col_t2, col_t3, col_t4 = st.columns(4)
    with col_t1:
        st.markdown("**🐍 Linguagem**\n- Python 3.14+")
    with col_t2:
        st.markdown("**📊 ML & Visualização**\n- Scikit-Learn, XGBoost\n- Matplotlib, Seaborn, Plotly")
    with col_t3:
        st.markdown("**🔬 Explainable AI**\n- SHAP Values\n- Feature Importance")
    with col_t4:
        st.markdown("**🚀 Infraestrutura**\n- GPU (CUDA) p/ XGBoost\n- Streamlit Dashboard")

    st.markdown("## 📊 Base de Dados")
    st.markdown(
        """
        **Dataset:** [Game Recommendations on Steam](https://www.kaggle.com/datasets/antonkozyriev/game-recommendations-on-steam)

        | Característica | Valor |
        |---|---|
        | Avaliações | **+41 milhões** de recomendações |
        | Arquivos | `games.csv`, `users.csv`, `recommendations.csv`, `games_metadata.json` |
        | Target | `is_recommended` — binário (recomenda / não recomenda) |
        | Features principais | horas jogadas, preço, descontos, tags, perfil do usuário |
        """
    )


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — ANÁLISE EXPLORATÓRIA
# ═══════════════════════════════════════════════════════════════════════════

with tab2:
    st.markdown("## 📊 Análise Exploratória de Dados")
    st.markdown(
        "Três hipóteses foram formuladas e testadas com dados reais da Steam. "
        "Cada gráfico abaixo foi gerado a partir de uma amostra estratificada de "
        "200.000 recomendações."
    )
    st.divider()

    # ── H1: Tempo de Jogo ────────────────────────────────────────────
    st.markdown("## 🕐 Hipótese H1 — Tempo de Jogo vs Satisfação")
    st.markdown(
        """
        <div class="insight-box">
        <h4>H1: Quanto mais tempo o usuário joga, maior a probabilidade de recomendação positiva.</h4>
        <p>Usuários que dedicam mais horas a um jogo tendem a recomendá-lo com muito mais frequência.
        Isso reflete o <strong>engajamento</strong> como principal motor da satisfação.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    h1_path = ler_grafico("h1_playtime.png")
    if h1_path:
        st.image(h1_path, use_container_width=True)
    else:
        st.info("Gráfico H1 não encontrado. Execute o pipeline para gerá-lo.")

    st.markdown(
        """
        **📌 Observações:**
        - Jogadores na categoria **"Saturação" (+100h)** apresentam as maiores taxas de recomendação.
        - O salto mais expressivo ocorre entre **Baixo (0-2h)** e **Médio (2-20h)**.
        - Jogos com menos de 2 horas jogadas têm as menores taxas — frequentemente associados a
        decepções ou títulos que não prenderam o jogador.
        """
    )
    st.divider()

    # ── H2: Preço e Desconto ─────────────────────────────────────────
    st.markdown("## 💰 Hipótese H2 — Preço e Descontos na Satisfação")

    col_h2a, col_h2b = st.columns(2)
    with col_h2a:
        st.markdown("### H2A: Faixa de Preço")
        st.markdown(
            """
            **Hipótese:** Jogos mais caros (Premium) tendem a apresentar taxas
            de recomendação menores.
            """
        )
        h2a_path = ler_grafico("h2_price_tiers.png")
        if h2a_path:
            st.image(h2a_path, use_container_width=True)
        else:
            st.info("Gráfico H2A não encontrado.")
        st.markdown(
            "📌 Jogos **Premium (>$60)** têm taxas de recomendação significativamente "
            "menores que jogos **Baratos (<$10)** e **Médios ($10-$30)**."
        )

    with col_h2b:
        st.markdown("### H2B: Impacto de Descontos")
        st.markdown(
            """
            **Hipótese:** Descontos promocionais aumentam a percepção de valor e,
            consequentemente, a taxa de recomendação.
            """
        )
        h2b_path = ler_grafico("h2_discount_effect.png")
        if h2b_path:
            st.image(h2b_path, use_container_width=True)
        else:
            st.info("Gráfico H2B não encontrado.")
        st.markdown(
            "📌 Jogos adquiridos **em promoção** apresentam taxas de recomendação "
            "ligeiramente maiores que os adquiridos **sem desconto**."
        )
    st.divider()

    # ── H3: Gratuito vs Pago ──────────────────────────────────────────
    st.markdown("## 🆓 Hipótese H3 — Gratuito vs Pago")
    st.markdown(
        """
        <div class="insight-box">
        <h4>H3: Jogos gratuitos (free-to-play) apresentam comportamento de recomendação diferente dos pagos.</h4>
        <p>O risco financeiro zero dos títulos gratuitos pode influenciar a percepção
        de valor e a disposição para recomendar.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    h3_path = ler_grafico("h3_free_vs_paid.png")
    if h3_path:
        st.image(h3_path, use_container_width=True)
    else:
        st.info("Gráfico H3 não encontrado.")

    col_h3a, col_h3b = st.columns(2)
    with col_h3a:
        st.markdown(
            "📌 Jogos **pagos** tendem a ter taxas de recomendação **mais altas** "
            "que jogos gratuitos. Isso sugere que o investimento financeiro cria "
            "um viés de confirmação no usuário."
        )
    with col_h3b:
        st.markdown(
            "📌 Jogos **gratuitos** podem ter taxas mais baixas devido à menor "
            "barreira de entrada — usuários experimentam sem compromisso e "
            "abandonam mais facilmente."
        )
    st.divider()

    # ── Testes Estatísticos ───────────────────────────────────────────
    st.markdown("## 📈 Testes Estatísticos (H2)")
    st.markdown(
        "Para validar as hipóteses H2A e H2B, realizamos um **Teste Z para duas proporções** "
        "comparando as taxas de recomendação entre grupos."
    )

    resultado_testes = ler_testes_estatisticos()
    if resultado_testes:
        linhas = resultado_testes.strip().split("\n")
        st.text("\n".join(linhas[:3]))
        dados_teste = []
        for linha in linhas[3:]:
            if linha.strip():
                partes = [p.strip() for p in linha.split("|")]
                if len(partes) >= 7:
                    dados_teste.append(
                        {
                            "Comparação": partes[0],
                            "Grupo A (recom/total)": f"{partes[1]}/{partes[2]}",
                            "Grupo B (recom/total)": f"{partes[3]}/{partes[4]}",
                            "Estatística Z": f"{float(partes[5]):.3f}",
                            "p-value": f"{float(partes[6]):.4e}",
                        }
                    )

        if dados_teste:
            df_teste = pd.DataFrame(dados_teste)
            st.dataframe(df_teste, use_container_width=True, hide_index=True)

        st.markdown(
            """
            **Conclusões dos testes:**
            - **Desconto vs Preço Cheio:** z = -2.971, p = 0.0030 → diferença **significativa** (p < 0.01).
            - **Premium vs Barato:** z = -11.531, p ≈ 0 → diferença **altamente significativa**.
            - **Premium vs Médio:** z = -12.591, p ≈ 0 → diferença **altamente significativa**.

            ✅ As evidências suportam a hipótese de que **preço e descontos influenciam**
            significativamente a satisfação e a probabilidade de recomendação.
            """
        )
    else:
        st.info("Arquivo de testes estatísticos não encontrado.")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 — PERFORMANCE DO MODELO
# ═══════════════════════════════════════════════════════════════════════════

with tab3:
    st.markdown("## 🧠 Performance dos Modelos Preditivos")
    st.markdown(
        "Três modelos foram treinados e comparados usando **Stratified K-Fold Cross-Validation** "
        "(k=5) com divisão holdout 80/20. O **XGBoost** foi selecionado como modelo final."
    )
    st.divider()

    # Métricas
    st.markdown("### 📋 Tabela Comparativa de Métricas")

    metrica_data = [
        {
            "Modelo": "Regressão Logística",
            "F1 (CV)": "0.8150 ± 0.0022",
            "AUC (CV)": "0.6713 ± 0.0051",
            "F1 (Holdout)": "0.8133",
            "AUC (Holdout)": "0.6707",
        },
        {
            "Modelo": "Random Forest",
            "F1 (CV)": "0.8883 ± 0.0016",
            "AUC (CV)": "0.6541 ± 0.0031",
            "F1 (Holdout)": "0.8897",
            "AUC (Holdout)": "0.6620",
        },
        {
            "Modelo": "XGBoost (selecionado)",
            "F1 (CV)": "0.9237 ± 0.0004",
            "AUC (CV)": "0.6970 ± 0.0034",
            "F1 (Holdout)": "0.9241",
            "AUC (Holdout)": "0.7037",
        },
    ]

    st.dataframe(pd.DataFrame(metrica_data), use_container_width=True, hide_index=True)

    st.markdown(
        """
        <div class="insight-box">
        <h4>🏆 XGBoost — Melhor Modelo</h4>
        <p>
        O XGBoost apresentou superioridade consistente em todas as métricas:
        <strong>F1 = 0.9241</strong> e <strong>AUC = 0.7037</strong> no holdout,
        com a menor variância entre folds (CV). Sua capacidade de capturar
        interações não-lineares entre features como preço, horas jogadas e perfil
        do usuário o torna ideal para este problema.
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    # Curvas ROC
    st.markdown("### 📈 Curvas ROC — Comparação dos Modelos")

    col_roc1, col_roc2 = st.columns(2)
    with col_roc1:
        roc_comp_path = ler_grafico("roc_comparado.png")
        if roc_comp_path:
            st.image(roc_comp_path, use_container_width=True)
        else:
            st.info("Gráfico ROC comparado não encontrado.")

        roc_xgb_path = ler_grafico("roc_XGBoost_Classifier.png")
        if roc_xgb_path:
            st.image(roc_xgb_path, use_container_width=True)

    with col_roc2:
        roc_rf_path = ler_grafico("roc_Random_Forest.png")
        if roc_rf_path:
            st.image(roc_rf_path, use_container_width=True)

        roc_lr_path = ler_grafico("roc_Regressão_Logística__Baseline_.png")
        if roc_lr_path:
            st.image(roc_lr_path, use_container_width=True)

    st.divider()

    # CV Metrics
    st.markdown("### 📊 Validação Cruzada — Métricas por Fold")

    col_cv1, col_cv2, col_cv3 = st.columns(3)
    with col_cv1:
        cv_xgb_path = ler_grafico("cv_metrics_XGBoost_Classifier.png")
        if cv_xgb_path:
            st.image(cv_xgb_path, use_container_width=True)

    with col_cv2:
        cv_rf_path = ler_grafico("cv_metrics_Random_Forest.png")
        if cv_rf_path:
            st.image(cv_rf_path, use_container_width=True)

    with col_cv3:
        cv_lr_path = ler_grafico("cv_metrics_Regressão_Logística__Baseline_.png")
        if cv_lr_path:
            st.image(cv_lr_path, use_container_width=True)

    st.divider()

    # Holdout Comparison
    st.markdown("### 🏅 Comparação Holdout: F1 vs AUC")
    holdout_path = ler_grafico("holdout_comparison.png")
    if holdout_path:
        st.image(holdout_path, use_container_width=True, width=700)

    st.markdown(
        """
        **Análise:** O XGBoost se destaca com o maior F1-Score (0.9241) e AUC-ROC (0.7037),
        indicando excelente equilíbrio entre precisão e recall. A Regressão Logística,
        apesar de mais simples, serve como baseline competitiva. O Random Forest
        apresenta bom F1 mas menor capacidade de discriminação (AUC).
        """
    )


# ═══════════════════════════════════════════════════════════════════════════
# TAB 4 — IMPORTÂNCIA & SHAP
# ═══════════════════════════════════════════════════════════════════════════

with tab4:
    st.markdown("## 🔍 Importância das Variáveis e Interpretabilidade")
    st.markdown(
        "Compreender quais fatores mais influenciam a recomendação é tão importante "
        "quanto fazer a predição. Abaixo, exploramos a importância das features "
        "e a interpretabilidade global com SHAP."
    )
    st.divider()

    st.markdown("### 🌲 Feature Importance — XGBoost")
    col_fi1, col_fi2 = st.columns(2)
    with col_fi1:
        fi_xgb_path = ler_grafico("feature_importances_XGBoost_Classifier.png")
        if fi_xgb_path:
            st.image(fi_xgb_path, use_container_width=True)
        else:
            st.info("Feature importance XGBoost não encontrado.")

    with col_fi2:
        fi_rf_path = ler_grafico("feature_importances_Random_Forest.png")
        if fi_rf_path:
            st.image(fi_rf_path, use_container_width=True)

    st.markdown("### 📉 Importância — Regressão Logística (Coeficientes)")
    fi_lr_path = ler_grafico("feature_importances_Regressão_Logística__Baseline_.png")
    if fi_lr_path:
        col_lr, _ = st.columns([2, 2])
        with col_lr:
            st.image(fi_lr_path, use_container_width=True)

    st.markdown(
        """
        <div class="insight-box">
        <h4>📌 Principais Insights sobre as Features</h4>
        <p>
        <strong>1. Hours (horas jogadas)</strong> — É a feature mais importante em todos os modelos.
        Quanto mais tempo o usuário investe, maior a probabilidade de recomendação positiva.<br>
        <strong>2. Price (preço final)</strong> — Preços elevados reduzem a satisfação. Jogos
        premium têm taxas de recomendação mais baixas.<br>
        <strong>3. Is Free (gratuidade)</strong> — Jogos gratuitos têm comportamento distinto
        dos pagos, com menor taxa de recomendação.<br>
        <strong>4. Products/Reviews</strong> — Usuários mais experientes (com mais produtos e
        reviews) tendem a ser mais seletivos nas recomendações.
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    # SHAP
    st.markdown("### 📊 SHAP — Explicabilidade Global")
    st.markdown(
        "O SHAP (SHapley Additive exPlanations) quantifica a contribuição de cada variável "
        "para a predição, baseado na teoria dos jogos cooperativos."
    )

    shap_path = ler_grafico("shap_summary.png")
    if shap_path:
        st.image(shap_path, use_container_width=True)
    else:
        st.info("Gráfico SHAP não encontrado. Execute o pipeline para gerá-lo.")

    st.markdown(
        """
        **Interpretação do gráfico SHAP:**
        - **Horas jogadas (hours):** valores altos (vermelho) empurram a predição para
        recomendação positiva (SHAP positivo); valores baixos (azul), para negativa.
        - **Preço final (price_final):** preços altos contribuem negativamente para a recomendação.
        - **Quantidade de reviews (reviews):** quanto mais reviews o usuário escreveu, maior o
        impacto na decisão.
        - **is_free:** jogos gratuitos têm um efeito misto, dependendo do contexto.

        💡 *Insight para publishers: invista em engajamento (horas jogadas) e precifique
        adequadamente. Descontos estratégicos podem mitigar o efeito negativo de preços altos.*
        """
    )


# ═══════════════════════════════════════════════════════════════════════════
# TAB 5 — PREDITOR INTERATIVO
# ═══════════════════════════════════════════════════════════════════════════

with tab5:
    st.markdown("## 🎯 Preditor Interativo de Recomendação")
    st.markdown(
        "Use o formulário abaixo para simular o comportamento de um usuário e prever "
        "se ele **recomendaria** ou **não recomendaria** um determinado jogo na Steam."
    )

    if modelo is None:
        st.error(
            "🚫 Modelo não encontrado. Execute o pipeline primeiro com:\n\n"
            "```bash\npoetry run pipeline\n```\n\n"
            "Ou verifique se o arquivo `best_model.joblib` existe em "
            "`src/projeto_cd/plots/`."
        )
    else:
        st.markdown("### 🎮 Parâmetros do Jogo e do Usuário")
        st.markdown(
            "Preencha as informações abaixo. Os campos com * são obrigatórios."
        )

        col_form1, col_form2 = st.columns(2)

        with col_form1:
            st.markdown("**📦 Sobre o Jogo**")
            price_final = st.number_input(
                "💵 Preço final (US$)*",
                min_value=0.0,
                max_value=200.0,
                value=29.99,
                step=0.01,
                format="%.2f",
                help="Preço de venda do jogo na loja Steam (0 para gratuito).",
            )

            num_tags = st.slider(
                "🏷️ Quantidade de tags do jogo",
                min_value=0,
                max_value=50,
                value=8,
                help="Número de categorias/tags associadas ao jogo (ex: Ação, Aventura, RPG).",
            )

        with col_form2:
            st.markdown("**👤 Perfil do Usuário**")
            hours = st.slider(
                "⏱️ Horas jogadas*",
                min_value=0.0,
                max_value=500.0,
                value=15.0,
                step=0.5,
                help="Quantidade de horas que o usuário jogou este título.",
            )

            products = st.number_input(
                "🛒 Total de produtos na conta",
                min_value=0,
                max_value=50000,
                value=50,
                step=1,
                help="Quantos jogos o usuário possui em sua biblioteca Steam.",
            )

            reviews = st.number_input(
                "✍️ Total de reviews escritos",
                min_value=0,
                max_value=10000,
                value=10,
                step=1,
                help="Quantas avaliações (reviews) o usuário já escreveu na plataforma.",
            )

        st.markdown("---")

        # Informações derivadas
        is_free = price_final == 0
        if hours <= 2:
            cat_label = "Baixo (0-2h)"
        elif hours <= 20:
            cat_label = "Médio (2-20h)"
        elif hours <= 100:
            cat_label = "Alto (20-100h)"
        else:
            cat_label = "Saturação (100h+)"

        col_info1, col_info2, col_info3 = st.columns(3)
        with col_info1:
            st.markdown(
                f"**Modelo de monetização:** {'🆓 Gratuito' if is_free else '💰 Pago'}"
            )
        with col_info2:
            st.markdown(f"**Categoria de horas:** {cat_label}")
        with col_info3:
            desconto_info = "Disponível para desconto" if price_final > 0 else "N/A"
            st.markdown(f"**Observação:** {desconto_info}")

        # ── Predição ──────────────────────────────────────────────────
        if st.button("🔮 Prever Recomendação", type="primary", use_container_width=True):
            with st.spinner("Calculando predição..."):
                try:
                    X_pred = engenharia_para_predicao(
                        hours=hours,
                        price_final=price_final,
                        products=products,
                        reviews=reviews,
                        num_tags=num_tags,
                    )

                    proba = modelo.predict_proba(X_pred)[0, 1]
                    resultado, confianca = interpretar_resultado(proba)

                    st.markdown("---")
                    st.markdown("### 📊 Resultado da Predição")

                    col_res1, col_res2, col_res3 = st.columns([1, 2, 1])

                    with col_res2:
                        if resultado == "recomendado":
                            st.markdown(
                                f"""
                                <div class="prediction-positive">
                                <h2 style="color:#00d4aa; margin:0;">✅ Recomendado</h2>
                                <p style="font-size:1.3rem; margin-top:0.5rem;">
                                Confiança: <strong>{confianca:.1%}</strong>
                                </p>
                                <p style="color:#aaa; font-size:0.9rem;">
                                O modelo indica que este usuário provavelmente
                                recomendaria este jogo.
                                </p>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                        else:
                            st.markdown(
                                f"""
                                <div class="prediction-negative">
                                <h2 style="color:#ff6b6b; margin:0;">❌ Não Recomendado</h2>
                                <p style="font-size:1.3rem; margin-top:0.5rem;">
                                Confiança: <strong>{confianca:.1%}</strong>
                                </p>
                                <p style="color:#aaa; font-size:0.9rem;">
                                O modelo indica que este usuário provavelmente
                                NÃO recomendaria este jogo.
                                </p>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                    # ── Fatores que influenciaram ──────────────────────
                    st.markdown("### 🔍 Fatores que influenciaram esta decisão")

                    fatores = []
                    if hours > 100:
                        fatores.append(("🕐 Horas jogadas", "Muito alto", "positivo"))
                    elif hours > 20:
                        fatores.append(("🕐 Horas jogadas", "Alto", "positivo"))
                    elif hours > 2:
                        fatores.append(("🕐 Horas jogadas", "Médio", "neutro"))
                    else:
                        fatores.append(("🕐 Horas jogadas", "Baixo", "negativo"))

                    if is_free:
                        fatores.append(("🆓 Gratuito", "Sim", "neutro"))
                    else:
                        if price_final > 60:
                            fatores.append(("💵 Preço", "Premium (>$60)", "negativo"))
                        elif price_final > 30:
                            fatores.append(("💵 Preço", "Caro ($30-$60)", "neutro"))
                        else:
                            fatores.append(("💵 Preço", "Acessível", "positivo"))

                    if products > 100:
                        fatores.append(("🛒 Experiência do usuário", "Muitos produtos", "neutro"))
                    elif products > 10:
                        fatores.append(("🛒 Experiência do usuário", "Moderada", "positivo"))

                    if num_tags > 15:
                        fatores.append(("🏷️ Tags", "Muitas tags", "positivo"))

                    col_f1, col_f2 = st.columns(2)
                    for i, (nome, valor, impacto) in enumerate(fatores):
                        icone = "✅" if impacto == "positivo" else ("⚠️" if impacto == "neutro" else "❌")
                        with col_f1 if i % 2 == 0 else col_f2:
                            st.markdown(
                                f"**{icone} {nome}:** {valor} "
                                f"({'👍' if impacto == 'positivo' else '👎' if impacto == 'negativo' else '➖'})"
                            )

                    st.markdown("---")
                    st.markdown(
                        f"**Probabilidade bruta:** {proba:.4f} "
                        f"(threshold de decisão: 0.5)"
                    )

                except Exception as e:
                    st.error(f"Erro ao realizar a predição: {e}")
                    st.info(
                        "Verifique se o modelo foi treinado com as mesmas "
                        "features esperadas."
                    )

    # Informações sobre as features do modelo
    with st.expander("📖 Sobre as features do modelo"):
        st.markdown(
            """
            O modelo XGBoost foi treinado com as seguintes features:

            | Feature | Descrição | Tipo |
            |---------|-----------|------|
            | `hours` | Horas jogadas pelo usuário | Contínua |
            | `price_final` | Preço final do jogo (US$) | Contínua |
            | `is_free` | Indicador de jogo gratuito | Binária (0/1) |
            | `products` | Total de produtos na conta do usuário | Inteira |
            | `reviews` | Total de reviews escritos pelo usuário | Inteira |
            | `num_tags` | Número de tags associadas ao jogo | Inteira |
            | `playtime_category_Medio` | 2h < horas ≤ 20h | Binária (0/1) |
            | `playtime_category_Alto` | 20h < horas ≤ 100h | Binária (0/1) |
            | `playtime_category_Saturacao` | horas > 100h | Binária (0/1) |

            O modelo atingiu **F1-Score de 0.9241** e **AUC-ROC de 0.7037** no holdout.
            """
        )

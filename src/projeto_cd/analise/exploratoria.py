"""
exploratoria.py — Visualizações da análise exploratória de dados (EDA).

Gera e exporta os gráficos associados às hipóteses H1 (tempo de jogo),
H2 (preço e desconto) e H3 (gratuidade).
"""

import os
from typing import Any, cast

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.axes import Axes

from projeto_cd.config import PLOTS_DIR


def _anotar_barras(ax: Axes) -> None:
    """
    Anota valores percentuais sobre cada barra de um gráfico.

    Parâmetros
    ----------
    ax : plt.Axes
        Eixo matplotlib contendo as barras.
    """
    for p in ax.patches:
        p = cast(Any, p)
        ax.annotate(
            f"{p.get_height():.1%}",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="bottom",
            fontsize=10,
            color="black",
            xytext=(0, 5),
            textcoords="offset points",
        )


def grafico_h1_playtime(df) -> None:
    """
    Gráfico H1: taxa de recomendação por categoria de horas jogadas.

    Hipotese H1 — Quanto mais tempo o usuário joga, maior a probabilidade
    de recomendação positiva.

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame com as colunas ``playtime_category`` e ``is_recommended``.
    """
    plt.figure()
    ax = sns.barplot(
        data=df,
        x="playtime_category",
        y="is_recommended",
        errorbar=None,
        palette="viridis",
    )
    plt.title("Taxa de Recomendação por Categoria de Horas Dedicadas (H1)")
    plt.ylabel("Proporção de Recomendações Positivas")
    plt.xlabel("Categoria de Tempo de Jogo")
    _anotar_barras(ax)
    plt.savefig(os.path.join(PLOTS_DIR, "h1_playtime.png"), dpi=300, bbox_inches="tight")
    plt.close()


def grafico_h2_faixa_preco(df) -> None:
    """
    Gráfico H2A: taxa de recomendação por faixa de preço do jogo.

    Hipotese H2A — Jogos mais caros (Premium) tendem a apresentar
    taxas de recomendação menores.

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame (apenas jogos pagos) com ``price_tier`` e ``is_recommended``.
    """
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        data=df,
        x="price_tier",
        y="is_recommended",
        errorbar=None,
        palette="flare",
    )
    plt.title("H2A: Taxa de Recomendação por Faixa de Preço do Jogo", fontsize=14)
    plt.xlabel("Categoria de Preço (US$)")
    plt.ylabel("Taxa de Recomendação Média")
    _anotar_barras(ax)
    plt.savefig(os.path.join(PLOTS_DIR, "h2_price_tiers.png"), dpi=300, bbox_inches="tight")
    plt.close()


def grafico_h2_desconto(df) -> None:
    """
    Gráfico H2B: impacto de descontos promocionais na satisfação.

    Hipotese H2B — Descontos aumentam a percepção de valor e,
    consequentemente, a taxa de recomendação.

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame (apenas jogos pagos) com ``has_discount`` e ``is_recommended``.
    """
    plt.figure(figsize=(8, 6))
    ax = sns.barplot(
        data=df,
        x="has_discount",
        y="is_recommended",
        errorbar=None,
        palette="crest",
    )
    plt.title("H2B: Impacto de Descontos Promocionais na Satisfação", fontsize=14)
    plt.xticks([0, 1], ["Preço Cheio (Sem Desconto)", "Adquirido em Promoção"])
    plt.ylabel("Taxa de Recomendação Média")
    plt.xlabel("Status de Promoção na Loja")
    _anotar_barras(ax)
    plt.savefig(os.path.join(PLOTS_DIR, "h2_discount_effect.png"), dpi=300, bbox_inches="tight")
    plt.close()


def grafico_h3_gratuito_vs_pago(df) -> None:
    """
    Gráfico H3: comparação da taxa de recomendação entre gratuitos e pagos.

    Hipotese H3 — Jogos gratuitos (free-to-play) apresentam comportamento
    de recomendação diferente dos jogos pagos.

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame com ``is_free`` e ``is_recommended``.
    """
    plt.figure()
    ax = sns.barplot(
        data=df,
        x="is_free",
        y="is_recommended",
        errorbar=None,
        palette="mako",
    )
    plt.title("Impacto do Risco Financeiro na Recomendação: Pagos vs Gratuitos (H3)")
    plt.xticks([0, 1], ["Título Pago", "Título Gratuito (Free-to-Play)"])
    plt.ylabel("Proporção de Recomendações Positivas")
    plt.xlabel("Modelo de Monetização")
    _anotar_barras(ax)
    plt.savefig(os.path.join(PLOTS_DIR, "h3_free_vs_paid.png"), dpi=300, bbox_inches="tight")
    plt.close()


def gerar_graficos_exploratorios(df, df_paid=None) -> None:
    """
    Gera e exporta todos os gráficos da análise exploratória (H1, H2 e H3).

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame completo (com atributos de engenharia).
    df_paid : pd.DataFrame, opcional
        DataFrame filtrado apenas para jogos pagos (price_final > 0).
        Se não informado, o filtro é aplicado internamente.
    """
    print("Gerando gráficos da análise exploratória (H1, H2, H3)...")

    grafico_h1_playtime(df)

    if df_paid is None:
        df_paid = df[df["price_final"] > 0].copy()

    grafico_h2_faixa_preco(df_paid)
    grafico_h2_desconto(df_paid)
    grafico_h3_gratuito_vs_pago(df)
